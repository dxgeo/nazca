//! MapLibre Tile (MLT) decoding, for basemaps published in it. Enabled by
//! the `mlt` feature, which adds no dependency.
//!
//! MLT v1 (frame tag `0x01`), the stable format, as specified at
//! <https://github.com/maplibre/maplibre-tile-spec> (`docs/specification/v1.md`
//! and `docs/encodings.md`). Frames with any other tag are skipped, as the
//! specification requires; the experimental v2 is not read.
//!
//! MLT is columnar, so a layer is decoded whole: [`Layer::decode`] turns its
//! geometry column and its id column into [`Features`], and
//! [`Features::read`] feeds one feature into a [`Sink`] in map units. Every
//! integer encoding a v1 geometry or id stream can name is read: fixed-width
//! words, varints and SIMD-FastPFOR; delta, RLE and delta-RLE; vertices plain,
//! componentwise-delta, dictionary and Morton. Property columns are parsed
//! only far enough to skip them, as nazca holds no attributes (`../project.org`
//! § WHAT NAZCA DOES NOT DO). Pre-tessellated triangles are skipped too: their
//! indices count vertices before nazca's own snapping and clipping, which
//! changes them.
//!
//! Reading the tile out of its container and decompressing it are the
//! embedder's I/O, as for [`crate::mvt`].

use crate::grid::Rect;
use crate::sink::Sink;
use std::fmt;
use std::ops::Range;

/// The most values one stream may decode to: a bound on what a few bytes of
/// RLE or FastPFOR can ask to allocate.
const MAX_VALUES: usize = 1 << 26;

/// Why a tile could not be read.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The bytes end inside a header or a stream.
    Truncated,
    /// A varint longer than its type allows.
    Varint,
    /// A layer name that is empty or not UTF-8.
    Name,
    /// A layer whose extent is zero.
    Extent,
    /// A column type byte the specification leaves unassigned.
    ColumnType(u8),
    /// A layer without exactly one geometry column, or with two id columns.
    Columns,
    /// A stream type byte that is not valid, or not allowed where it stands.
    StreamType(u8),
    /// An encoding byte naming no valid combination.
    Encoding(u8),
    /// A stream whose payload does not hold what its header says, or that
    /// would decode to more than 2^26 values.
    StreamSize,
    /// Malformed SIMD-FastPFOR data.
    FastPfor(&'static str),
    /// Morton parameters out of range.
    Morton,
    /// A geometry type outside MLT's six.
    GeomType(u64),
    /// Topology that indexes past its own streams, or offsets that overflow.
    Topology,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated => write!(f, "MLT tile ends inside a header or stream"),
            Error::Varint => write!(f, "MLT varint too long"),
            Error::Name => write!(f, "MLT layer name is empty or not UTF-8"),
            Error::Extent => write!(f, "MLT layer has an extent of zero"),
            Error::ColumnType(t) => write!(f, "MLT column type {t} is unassigned"),
            Error::Columns => write!(f, "MLT layer needs exactly one geometry column and at most one id column"),
            Error::StreamType(t) => write!(f, "MLT stream type 0x{t:02x} is not valid here"),
            Error::Encoding(e) => write!(f, "MLT encoding byte 0x{e:02x} names no valid encoding"),
            Error::StreamSize => write!(f, "MLT stream does not hold what its header says"),
            Error::FastPfor(why) => write!(f, "MLT FastPFOR stream is malformed: {why}"),
            Error::Morton => write!(f, "MLT Morton parameters are out of range"),
            Error::GeomType(t) => write!(f, "MLT geometry type {t} is not one of the six"),
            Error::Topology => write!(f, "MLT geometry topology indexes past its streams"),
        }
    }
}

impl std::error::Error for Error {}

/// A feature's geometry type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GeomType {
    Point = 0,
    LineString = 1,
    Polygon = 2,
    MultiPoint = 3,
    MultiLineString = 4,
    MultiPolygon = 5,
}

impl GeomType {
    fn from_u64(v: u64) -> Result<GeomType, Error> {
        Ok(match v {
            0 => GeomType::Point,
            1 => GeomType::LineString,
            2 => GeomType::Polygon,
            3 => GeomType::MultiPoint,
            4 => GeomType::MultiLineString,
            5 => GeomType::MultiPolygon,
            t => return Err(Error::GeomType(t)),
        })
    }

    fn is_line(self) -> bool {
        matches!(self, GeomType::LineString | GeomType::MultiLineString)
    }

    fn is_polygon(self) -> bool {
        matches!(self, GeomType::Polygon | GeomType::MultiPolygon)
    }
}

// --- Bytes --------------------------------------------------------------

struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn u8(&mut self) -> Result<u8, Error> {
        let (&b, rest) = self.bytes.split_first().ok_or(Error::Truncated)?;
        self.bytes = rest;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        if self.bytes.len() < n {
            return Err(Error::Truncated);
        }
        let (head, rest) = self.bytes.split_at(n);
        self.bytes = rest;
        Ok(head)
    }

    fn varint(&mut self) -> Result<u64, Error> {
        let mut value = 0u64;
        for i in 0..10 {
            let b = self.u8()?;
            value |= u64::from(b & 0x7f) << (7 * i);
            if b < 0x80 {
                return Ok(value);
            }
        }
        Err(Error::Varint)
    }

    fn varint32(&mut self) -> Result<u32, Error> {
        u32::try_from(self.varint()?).map_err(|_| Error::Varint)
    }

    fn string(&mut self) -> Result<&'a str, Error> {
        let len = self.varint32()? as usize;
        std::str::from_utf8(self.take(len)?).map_err(|_| Error::Name)
    }
}

fn unzigzag32(u: u32) -> i32 {
    ((u >> 1) as i32) ^ -((u & 1) as i32)
}

fn unzigzag64(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

// --- Frames and layers --------------------------------------------------

/// The v1 layers of a tile, in order. Frames of any other format are
/// skipped.
pub fn layers(tile: &[u8]) -> Layers<'_> {
    Layers { reader: Reader { bytes: tile } }
}

/// See [`layers`].
pub struct Layers<'a> {
    reader: Reader<'a>,
}

impl<'a> Iterator for Layers<'a> {
    type Item = Result<Layer<'a>, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.reader.bytes.is_empty() {
                return None;
            }
            let frame = (|| {
                let size = self.reader.varint32()? as usize;
                if size == 0 {
                    return Err(Error::Truncated);
                }
                let frame = self.reader.take(size)?;
                Ok((frame[0], &frame[1..]))
            })();
            match frame {
                Err(e) => {
                    self.reader.bytes = &[];
                    return Some(Err(e));
                }
                Ok((0x01, body)) => return Some(Layer::parse(body)),
                Ok(_) => {}
            }
        }
    }
}

/// A column's type and, for a shared dictionary, its children's.
#[derive(Debug, Clone)]
struct Column {
    kind: u8,
    children: Vec<u8>,
}

/// One v1 layer, its header parsed and its columns not yet decoded.
#[derive(Debug, Clone)]
pub struct Layer<'a> {
    name: &'a str,
    extent: u32,
    columns: Vec<Column>,
    data: &'a [u8],
}

impl<'a> Layer<'a> {
    fn parse(body: &'a [u8]) -> Result<Layer<'a>, Error> {
        let mut r = Reader { bytes: body };
        let name = r.string()?;
        if name.is_empty() {
            return Err(Error::Name);
        }
        let extent = r.varint32()?;
        if extent == 0 {
            return Err(Error::Extent);
        }
        let count = r.varint32()? as usize;
        let mut columns = Vec::with_capacity(count.min(body.len()));
        let (mut geometry, mut id) = (0, 0);
        for _ in 0..count {
            let kind = r.u8()?;
            match kind {
                0..=3 => id += 1,
                4 => geometry += 1,
                10 | 11 | 16..=30 => {
                    r.string()?;
                }
                t => return Err(Error::ColumnType(t)),
            }
            let mut children = Vec::new();
            if kind == 30 {
                let n = r.varint32()? as usize;
                for _ in 0..n {
                    let child = r.u8()?;
                    if !matches!(child, 28 | 29) {
                        return Err(Error::ColumnType(child));
                    }
                    r.string()?;
                    children.push(child);
                }
            }
            columns.push(Column { kind, children });
        }
        if geometry != 1 || id > 1 {
            return Err(Error::Columns);
        }
        Ok(Layer { name, extent, columns, data: r.bytes })
    }

    pub fn name(&self) -> &'a str {
        self.name
    }

    /// Tile units across the tile.
    pub fn extent(&self) -> u32 {
        self.extent
    }

    /// Decodes the layer's geometry and ids, skipping its properties.
    pub fn decode(&self) -> Result<Features, Error> {
        let mut r = Reader { bytes: self.data };
        let mut geometry = None;
        let mut ids = None;
        for column in &self.columns {
            match column.kind {
                0..=3 => {
                    let wide = column.kind >= 2;
                    let present = if column.kind & 1 == 1 { Some(bools(&Stream::parse(&mut r, true)?)?) } else { None };
                    let values = ints(&Stream::parse(&mut r, false)?, wide)?;
                    ids = Some(with_presence(values, present)?);
                }
                4 => geometry = Some(decode_geometry(&mut r)?),
                10 | 11 | 16..=27 => {
                    if column.kind & 1 == 1 {
                        Stream::parse(&mut r, true)?;
                    }
                    Stream::parse(&mut r, column.kind <= 11)?;
                }
                28 | 29 => {
                    let n = r.varint32()?;
                    for _ in 0..n {
                        Stream::skip(&mut r)?;
                    }
                }
                30 => {
                    r.varint32()?;
                    // The dictionary's streams end at its data stream.
                    loop {
                        let s = Stream::skip(&mut r)?;
                        if s == 0x11 || s == 0x12 {
                            break;
                        }
                    }
                    for _ in &column.children {
                        let n = r.varint32()?;
                        for _ in 0..n {
                            Stream::skip(&mut r)?;
                        }
                    }
                }
                _ => unreachable!("column types are checked when the layer is parsed"),
            }
        }
        let mut features = geometry.ok_or(Error::Columns)?;
        if let Some(ids) = ids {
            if ids.len() != features.types.len() {
                return Err(Error::StreamSize);
            }
            features.ids = ids;
        } else {
            features.ids = vec![None; features.types.len()];
        }
        features.extent = self.extent;
        Ok(features)
    }
}

/// `values` laid out over `present`: each set bit takes the next value.
fn with_presence(values: Vec<u64>, present: Option<Vec<bool>>) -> Result<Vec<Option<u64>>, Error> {
    let Some(present) = present else { return Ok(values.into_iter().map(Some).collect()) };
    let mut values = values.into_iter();
    let out: Vec<Option<u64>> = present.iter().map(|&p| if p { values.next() } else { None }).collect();
    let set = present.iter().filter(|&&p| p).count();
    if values.next().is_some() || out.iter().filter(|v| v.is_some()).count() != set {
        return Err(Error::StreamSize);
    }
    Ok(out)
}

// --- Streams ------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum Logical {
    None,
    Delta,
    DeltaRle,
    Componentwise,
    Rle,
    Morton { bits: u32, shift: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Physical {
    Fixed,
    FastPfor,
    Varint,
}

/// A stream: its header and payload.
#[derive(Debug, Clone, Copy)]
struct Stream<'a> {
    kind: u8,
    logical: Logical,
    physical: Physical,
    /// Physical words in the payload (for RLE, the runs and their values).
    words: usize,
    /// Values after an RLE expands, from the header.
    rle_values: usize,
    data: &'a [u8],
}

impl<'a> Stream<'a> {
    /// Parses a stream header and takes its payload. A boolean stream
    /// (presence, or a `Bool` column) carries no RLE counts in its header.
    fn parse(r: &mut Reader<'a>, boolean: bool) -> Result<Stream<'a>, Error> {
        let kind = r.u8()?;
        let category = kind >> 4;
        let valid = match category {
            0 => kind == 0,
            1 => kind & 15 <= 5,
            2 => kind & 15 <= 3,
            3 => kind & 15 <= 6,
            _ => false,
        };
        if !valid {
            return Err(Error::StreamType(kind));
        }
        let encoding = r.u8()?;
        let physical = match encoding & 3 {
            0 => Physical::Fixed,
            1 => Physical::FastPfor,
            2 => Physical::Varint,
            _ => return Err(Error::Encoding(encoding)),
        };
        let words = r.varint32()? as usize;
        let byte_length = r.varint32()? as usize;
        let mut rle_values = 0;
        let logical = match (encoding >> 5, (encoding >> 2) & 7) {
            (0, 0) => Logical::None,
            (1, 0) => Logical::Delta,
            (2, 0) => Logical::Componentwise,
            (1, 3) | (3, 0) => {
                if boolean {
                    rle_values = byte_length;
                } else {
                    r.varint32()?;
                    rle_values = r.varint32()? as usize;
                }
                if encoding >> 5 == 1 { Logical::DeltaRle } else { Logical::Rle }
            }
            (4, 1) => {
                let bits = r.varint32()?;
                let shift = r.varint32()?;
                if bits > 16 {
                    return Err(Error::Morton);
                }
                Logical::Morton { bits, shift }
            }
            _ => return Err(Error::Encoding(encoding)),
        };
        let data = r.take(byte_length)?;
        Ok(Stream { kind, logical, physical, words, rle_values, data })
    }

    /// Skips a stream, returning its type byte.
    fn skip(r: &mut Reader<'a>) -> Result<u8, Error> {
        let boolean = r.bytes.first() == Some(&0);
        Ok(Stream::parse(r, boolean)?.kind)
    }
}

/// The payload's physical words, widened to `u64`.
fn words(s: &Stream, wide: bool) -> Result<Vec<u64>, Error> {
    if s.words > MAX_VALUES {
        return Err(Error::StreamSize);
    }
    match s.physical {
        Physical::Fixed => {
            let w = if wide { 8 } else { 4 };
            if s.data.len() != s.words * w {
                return Err(Error::StreamSize);
            }
            Ok(s.data
                .chunks_exact(w)
                .map(|c| if wide { u64::from_le_bytes(c.try_into().unwrap()) } else { u64::from(u32::from_le_bytes(c.try_into().unwrap())) })
                .collect())
        }
        Physical::Varint => {
            let mut r = Reader { bytes: s.data };
            let mut out = Vec::with_capacity(s.words.min(s.data.len()));
            for _ in 0..s.words {
                let v = r.varint()?;
                if !wide && v > u64::from(u32::MAX) {
                    return Err(Error::Varint);
                }
                out.push(v);
            }
            if !r.bytes.is_empty() {
                return Err(Error::StreamSize);
            }
            Ok(out)
        }
        Physical::FastPfor => {
            if wide {
                return Err(Error::FastPfor("v1 has no 64-bit FastPFOR"));
            }
            Ok(fastpfor(s.data, s.words)?.into_iter().map(u64::from).collect())
        }
    }
}

/// An unsigned integer stream's values (lengths, offsets, types, ids).
/// Deltas wrap at the word width.
fn ints(s: &Stream, wide: bool) -> Result<Vec<u64>, Error> {
    let w = words(s, wide)?;
    let mask = if wide { u64::MAX } else { u64::from(u32::MAX) };
    let undelta = |v: Vec<u64>| -> Vec<u64> {
        let mut prev = 0u64;
        v.into_iter()
            .map(|word| {
                let d = if wide { unzigzag64(word) as u64 } else { unzigzag32(word as u32) as i64 as u64 };
                prev = prev.wrapping_add(d) & mask;
                prev
            })
            .collect()
    };
    Ok(match s.logical {
        Logical::None => w,
        Logical::Delta => undelta(w),
        Logical::Rle => expand_runs(&w, s.rle_values)?,
        Logical::DeltaRle => undelta(expand_runs(&w, s.rle_values)?),
        Logical::Componentwise | Logical::Morton { .. } => return Err(Error::Encoding(0)),
    })
}

/// v1 RLE: all run lengths, then all values.
fn expand_runs(w: &[u64], total: usize) -> Result<Vec<u64>, Error> {
    if !w.len().is_multiple_of(2) || total > MAX_VALUES {
        return Err(Error::StreamSize);
    }
    let (runs, values) = w.split_at(w.len() / 2);
    let mut out = Vec::with_capacity(total);
    for (&n, &v) in runs.iter().zip(values) {
        if n as usize > total - out.len() {
            return Err(Error::StreamSize);
        }
        out.extend(std::iter::repeat_n(v, n as usize));
    }
    if out.len() != total {
        return Err(Error::StreamSize);
    }
    Ok(out)
}

/// A boolean stream's `words` bits: a bitmap, raw or under ORC byte RLE.
fn bools(s: &Stream) -> Result<Vec<bool>, Error> {
    if s.words > MAX_VALUES || s.physical != Physical::Fixed {
        return Err(Error::StreamSize);
    }
    let n_bytes = s.words.div_ceil(8);
    let bytes: Vec<u8> = match s.logical {
        Logical::None => {
            if s.data.len() < n_bytes {
                return Err(Error::StreamSize);
            }
            s.data[..n_bytes].to_vec()
        }
        Logical::Rle => {
            let mut r = Reader { bytes: s.data };
            let mut out = Vec::with_capacity(n_bytes);
            while out.len() < n_bytes {
                let c = r.u8()?;
                if c < 128 {
                    let b = r.u8()?;
                    out.extend(std::iter::repeat_n(b, c as usize + 3));
                } else {
                    out.extend_from_slice(r.take(256 - c as usize)?);
                }
            }
            if out.len() != n_bytes {
                return Err(Error::StreamSize);
            }
            out
        }
        _ => return Err(Error::Encoding(0)),
    };
    Ok((0..s.words).map(|i| bytes[i / 8] >> (i % 8) & 1 == 1).collect())
}

/// A vertex stream's vertices.
fn vertices(s: &Stream) -> Result<Vec<[i32; 2]>, Error> {
    let w = words(s, false)?;
    let pairs = |v: Vec<i32>| -> Result<Vec<[i32; 2]>, Error> {
        if !v.len().is_multiple_of(2) {
            return Err(Error::StreamSize);
        }
        Ok(v.chunks_exact(2).map(|c| [c[0], c[1]]).collect())
    };
    match s.logical {
        Logical::None => pairs(w.iter().map(|&u| unzigzag32(u as u32)).collect()),
        Logical::Delta => {
            let mut prev = 0i32;
            pairs(w.iter().map(|&u| {
                prev = prev.wrapping_add(unzigzag32(u as u32));
                prev
            }).collect())
        }
        Logical::Componentwise => {
            let mut prev = [0i32; 2];
            pairs(w.iter().enumerate().map(|(i, &u)| {
                prev[i % 2] = prev[i % 2].wrapping_add(unzigzag32(u as u32));
                prev[i % 2]
            }).collect())
        }
        Logical::Morton { bits, shift } => {
            // Plain differences between ascending codes, not zigzag.
            let mut code = 0u32;
            Ok(w.iter()
                .map(|&u| {
                    code = code.wrapping_add(u as u32);
                    let (x, y) = (compact(code), compact(code >> 1));
                    let mask = if bits == 0 { 0 } else { u32::MAX >> (32 - bits) };
                    [(x & mask).wrapping_sub(shift) as i32, (y & mask).wrapping_sub(shift) as i32]
                })
                .collect())
        }
        Logical::Rle | Logical::DeltaRle => Err(Error::Encoding(0)),
    }
}

/// Bit `2i` of `v` to bit `i`.
fn compact(mut v: u32) -> u32 {
    v &= 0x5555_5555;
    v = (v | (v >> 1)) & 0x3333_3333;
    v = (v | (v >> 2)) & 0x0F0F_0F0F;
    v = (v | (v >> 4)) & 0x00FF_00FF;
    (v | (v >> 8)) & 0x0000_FFFF
}

// --- SIMD-FastPFOR, the v1 variant ----------------------------------------
//
// Big-endian 32-bit words; pages of up to 65536 values in 256-value blocks,
// each packed LSB-first at its own width, 32 values per `width` words;
// exceptions patched from per-width arrays; the values past the last whole
// block in a variable-byte tail. As `encodings.md` § FastPFOR and the
// reference TypeScript decoder (`ts/src/decoding/fastPforDecoder.ts`) define.

const BLOCK: usize = 256;
const PAGE: usize = 65536;

fn fastpfor(bytes: &[u8], n: usize) -> Result<Vec<u32>, Error> {
    if !bytes.len().is_multiple_of(4) {
        return Err(Error::FastPfor("payload is not whole words"));
    }
    let w: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_be_bytes(c.try_into().unwrap())).collect();
    let mut out = vec![0u32; n];
    let mut pos = 0;
    let mut done = 0;
    if !w.is_empty() {
        let aligned = w[0] as usize;
        pos = 1;
        if !aligned.is_multiple_of(BLOCK) || aligned > n {
            return Err(Error::FastPfor("block count does not fit the stream"));
        }
        while done < aligned {
            let this = PAGE.min(aligned - done);
            pos = fastpfor_page(&w, pos, &mut out[done..done + this])?;
            done += this;
        }
    }
    // The tail: variable-byte, bit 7 set on each value's last byte, read from
    // each word's low byte up.
    let mut value = 0u32;
    let mut shift = 0;
    let tail = w.get(pos..).unwrap_or(&[]).iter().flat_map(|word| word.to_le_bytes());
    for b in tail {
        if done == n {
            break;
        }
        value |= u32::from(b & 0x7f).checked_shl(shift).unwrap_or(0);
        if b & 0x80 != 0 {
            out[done] = value;
            done += 1;
            value = 0;
            shift = 0;
        } else {
            shift += 7;
            if shift > 28 {
                return Err(Error::FastPfor("tail value longer than five bytes"));
            }
        }
    }
    if done != n {
        return Err(Error::FastPfor("fewer values than the stream holds"));
    }
    Ok(out)
}

/// Value `j` of a run packed LSB-first at `width` bits from `words`.
fn unpacked(words: &[u32], j: usize, width: u32) -> Result<u32, Error> {
    if width == 0 {
        return Ok(0);
    }
    let bit = j * width as usize;
    let (i, off) = (bit / 32, bit % 32);
    let lo = u64::from(*words.get(i).ok_or(Error::FastPfor("packed values run past the page"))?);
    let hi = if off + width as usize > 32 { u64::from(*words.get(i + 1).ok_or(Error::FastPfor("packed values run past the page"))?) } else { 0 };
    let v = ((hi << 32 | lo) >> off) & ((1u64 << width) - 1);
    Ok(v as u32)
}

/// Decodes one page into `out` (a whole number of blocks), returning the
/// word after it.
fn fastpfor_page(w: &[u32], start: usize, out: &mut [u32]) -> Result<usize, Error> {
    let where_meta = w[start] as usize;
    if where_meta == 0 || start + where_meta >= w.len() {
        return Err(Error::FastPfor("page metadata offset out of range"));
    }
    let packed = &w[start + 1..start + where_meta];
    let meta_bytes = w[start + where_meta] as usize;
    let meta_start = start + where_meta + 1;
    let bitmap_at = meta_start + meta_bytes.div_ceil(4);
    if bitmap_at >= w.len() {
        return Err(Error::FastPfor("page metadata runs past the payload"));
    }
    let meta: Vec<u8> = w[meta_start..bitmap_at].iter().flat_map(|x| x.to_le_bytes()).take(meta_bytes).collect();

    // The page's exception arrays, one per extra width k.
    let bitmap = w[bitmap_at];
    let mut p = bitmap_at + 1;
    let mut arrays: Vec<Vec<u32>> = vec![Vec::new(); 33];
    for k in 2..=32u32 {
        if bitmap >> (k - 1) & 1 == 0 {
            continue;
        }
        let size = *w.get(p).ok_or(Error::FastPfor("exception array header past the payload"))? as usize;
        p += 1;
        if size > PAGE {
            return Err(Error::FastPfor("exception array larger than a page"));
        }
        let need = (size * k as usize).div_ceil(32);
        let src = w.get(p..p + need).ok_or(Error::FastPfor("exception array past the payload"))?;
        arrays[k as usize] = (0..size).map(|j| unpacked(src, j, k)).collect::<Result<_, _>>()?;
        p += need;
    }

    let mut read = [0usize; 33];
    let mut m = 0;
    let mut q = 0;
    let meta_byte = |m: &mut usize| -> Result<u8, Error> {
        let b = *meta.get(*m).ok_or(Error::FastPfor("block metadata past its end"))?;
        *m += 1;
        Ok(b)
    };
    for block in out.chunks_exact_mut(BLOCK) {
        let width = u32::from(meta_byte(&mut m)?);
        let exceptions = meta_byte(&mut m)? as usize;
        if width > 32 {
            return Err(Error::FastPfor("block width above 32"));
        }
        let words_used = width as usize * BLOCK / 32;
        let src = packed.get(q..q + words_used).ok_or(Error::FastPfor("packed block past the page"))?;
        for (j, v) in block.iter_mut().enumerate() {
            *v = unpacked(src, j, width)?;
        }
        q += words_used;
        if exceptions > 0 {
            let max = u32::from(meta_byte(&mut m)?);
            if max <= width || max > 32 {
                return Err(Error::FastPfor("exception width out of range"));
            }
            let k = (max - width) as usize;
            for _ in 0..exceptions {
                let pos = meta_byte(&mut m)? as usize;
                let high = if k == 1 {
                    1
                } else {
                    let v = *arrays[k].get(read[k]).ok_or(Error::FastPfor("exception without a value"))?;
                    read[k] += 1;
                    v
                };
                block[pos] |= high << width;
            }
        }
    }
    if q != packed.len() {
        return Err(Error::FastPfor("packed blocks do not fill the page"));
    }
    Ok(p)
}

// --- Geometry -----------------------------------------------------------

/// A decoded layer: every feature's geometry type, the topology offsets, the
/// vertices, and ids.
#[derive(Debug, Clone, Default)]
pub struct Features {
    types: Vec<GeomType>,
    geometries: Option<Vec<u32>>,
    parts: Option<Vec<u32>>,
    rings: Option<Vec<u32>>,
    vertices: Vec<[i32; 2]>,
    ids: Vec<Option<u64>>,
    extent: u32,
}

fn decode_geometry(r: &mut Reader) -> Result<Features, Error> {
    let count = r.varint32()?;
    if count == 0 {
        return Ok(Features::default());
    }
    let types_stream = Stream::parse(r, false)?;
    let types = ints(&types_stream, false)?.into_iter().map(GeomType::from_u64).collect::<Result<Vec<_>, _>>()?;
    let (mut geometries, mut parts, mut rings, mut offsets) = (None, None, None, None);
    let mut vertex = None;
    let set_once = |slot: &mut Option<Vec<u64>>, s: &Stream| -> Result<(), Error> {
        if slot.is_some() {
            return Err(Error::StreamType(s.kind));
        }
        *slot = Some(ints(s, false)?);
        Ok(())
    };
    for _ in 1..count {
        let s = Stream::parse(r, false)?;
        match s.kind {
            0x00 => {}
            0x31 => set_once(&mut geometries, &s)?,
            0x32 => set_once(&mut parts, &s)?,
            0x33 => set_once(&mut rings, &s)?,
            0x20 => set_once(&mut offsets, &s)?,
            // Pre-tessellated triangles and their indices: not used.
            0x34 | 0x21 => {}
            0x13 | 0x14 => {
                if vertex.is_some() {
                    return Err(Error::StreamType(s.kind));
                }
                vertex = Some(vertices(&s)?);
            }
            k => return Err(Error::StreamType(k)),
        }
    }
    let to32 = |v: Option<Vec<u64>>| v.map(|v| v.into_iter().map(|x| x as u32).collect::<Vec<u32>>());
    let (geometries, parts, rings) = topology(&types, to32(geometries), to32(parts), to32(rings))?;
    let mut vertices = vertex.unwrap_or_default();
    if let Some(offsets) = offsets {
        vertices = offsets
            .iter()
            .map(|&i| vertices.get(i as usize).copied().ok_or(Error::Topology))
            .collect::<Result<_, _>>()?;
    }
    Ok(Features { types, geometries, parts, rings, vertices, ids: Vec::new(), extent: 0 })
}

/// Turns the length streams into offsets, as `v1.md` § Length Stream
/// Encoding Rules defines: the topmost stream present has implicit lengths
/// of 1 for types at or below a threshold, and the streams below it store a
/// length for every item of a type that has one.
#[allow(clippy::type_complexity)]
fn topology(
    types: &[GeomType],
    geometries: Option<Vec<u32>>,
    parts: Option<Vec<u32>>,
    rings: Option<Vec<u32>>,
) -> Result<(Option<Vec<u32>>, Option<Vec<u32>>, Option<Vec<u32>>), Error> {
    if let Some(g) = geometries {
        let g = root(types, &g, GeomType::Polygon)?;
        let (p, r) = match (parts, rings) {
            (Some(p), Some(r)) => {
                let p = level1(types, &g, &p, false)?;
                let r = level2(types, &g, &p, &r)?;
                (Some(p), Some(r))
            }
            (Some(p), None) => (Some(level1_lines(types, &g, &p)?), None),
            (None, r) => (None, r),
        };
        return Ok((Some(g), p, r));
    }
    if let Some(p) = parts {
        return Ok(match rings {
            Some(r) => {
                let lines = types.iter().any(|t| t.is_line());
                let p = root(types, &p, GeomType::LineString)?;
                let r = level1(types, &p, &r, lines)?;
                (None, Some(p), Some(r))
            }
            None => (None, Some(root(types, &p, GeomType::Point)?), None),
        });
    }
    Ok((None, None, rings))
}

fn add(a: u32, b: u32) -> Result<u32, Error> {
    a.checked_add(b).ok_or(Error::Topology)
}

/// Offsets for the topmost length stream: types above `threshold` take
/// their length from the stream, the rest 1.
fn root(types: &[GeomType], lengths: &[u32], threshold: GeomType) -> Result<Vec<u32>, Error> {
    let mut out = Vec::with_capacity(types.len() + 1);
    out.push(0);
    let mut next = lengths.iter();
    let mut at = 0u32;
    for &t in types {
        let n = if t > threshold { *next.next().ok_or(Error::Topology)? } else { 1 };
        at = add(at, n)?;
        out.push(at);
    }
    Ok(out)
}

/// The level below the geometries when there are no rings: lines store a
/// vertex count per part, points take one vertex each.
fn level1_lines(types: &[GeomType], top: &[u32], lengths: &[u32]) -> Result<Vec<u32>, Error> {
    let mut out = vec![0];
    let mut next = lengths.iter();
    let mut at = 0u32;
    for (&t, w) in types.iter().zip(top.windows(2)) {
        for _ in w[0]..w[1] {
            at = add(at, if t.is_line() { *next.next().ok_or(Error::Topology)? } else { 1 })?;
            out.push(at);
        }
    }
    Ok(out)
}

/// The level below `top`: polygons (and lines, where rings hold their
/// vertex counts) store a length per item, the rest 1.
fn level1(types: &[GeomType], top: &[u32], lengths: &[u32], lines: bool) -> Result<Vec<u32>, Error> {
    let mut out = vec![0];
    let mut next = lengths.iter();
    let mut at = 0u32;
    for (&t, w) in types.iter().zip(top.windows(2)) {
        let stored = t.is_polygon() || (lines && t.is_line());
        for _ in w[0]..w[1] {
            at = add(at, if stored { *next.next().ok_or(Error::Topology)? } else { 1 })?;
            out.push(at);
        }
    }
    Ok(out)
}

/// The rings under geometries and parts: every part of a non-point type
/// stores its vertex counts; a point takes one.
fn level2(types: &[GeomType], top: &[u32], mid: &[u32], lengths: &[u32]) -> Result<Vec<u32>, Error> {
    let mut out = vec![0];
    let mut next = lengths.iter();
    let mut at = 0u32;
    let mut part = 0usize;
    for (&t, w) in types.iter().zip(top.windows(2)) {
        let n = (w[1] - w[0]) as usize;
        if matches!(t, GeomType::Point | GeomType::MultiPoint) {
            for _ in 0..n {
                at = add(at, 1)?;
                out.push(at);
            }
            part += n;
            continue;
        }
        for _ in 0..n {
            let (a, b) = (*mid.get(part).ok_or(Error::Topology)?, *mid.get(part + 1).ok_or(Error::Topology)?);
            part += 1;
            for _ in a..b {
                at = add(at, *next.next().ok_or(Error::Topology)?)?;
                out.push(at);
            }
        }
    }
    Ok(out)
}

impl Features {
    pub fn len(&self) -> usize {
        self.types.len()
    }

    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    /// Feature `i`'s id, if the layer gives one.
    pub fn id(&self, i: usize) -> Option<u64> {
        self.ids.get(i).copied().flatten()
    }

    pub fn geom_type(&self, i: usize) -> Option<GeomType> {
        self.types.get(i).copied()
    }

    /// Decodes feature `i` into `sink`, mapping tile units to map units so
    /// that the layer's extent spans `bounds` (Y flips). Each polygon's
    /// first ring is its exterior and the rest its holes, from the topology,
    /// not from winding; rings arrive closed.
    pub fn read<S: Sink>(&self, i: usize, bounds: Rect, sink: &mut S) -> Result<(), Error> {
        let t = *self.types.get(i).ok_or(Error::Topology)?;
        let e = self.extent.max(1) as f64;
        let map = Scale {
            origin: [bounds.min[0], bounds.max[1]],
            per_unit: [(bounds.max[0] - bounds.min[0]) / e, (bounds.max[1] - bounds.min[1]) / e],
        };
        // Without a geometries stream, a feature's own index is its part.
        let part = |k: usize| if self.geometries.is_some() { at(&self.geometries, k) } else { Ok(k) };
        let mut buf = Vec::new();
        match t {
            GeomType::Point => {
                let mut k = part(i)?;
                if self.parts.is_some() {
                    k = at(&self.parts, k)?;
                }
                if self.rings.is_some() {
                    k = at(&self.rings, k)?;
                }
                sink.point(self.vertex(k, &map)?);
            }
            GeomType::MultiPoint => {
                for g in span(&self.geometries, i)? {
                    let k = match (&self.parts, &self.rings) {
                        (Some(_), Some(_)) => at(&self.rings, at(&self.parts, g)?)?,
                        (Some(_), None) => at(&self.parts, g)?,
                        (None, _) => g,
                    };
                    sink.point(self.vertex(k, &map)?);
                }
            }
            GeomType::LineString => self.line(part(i)?, &map, &mut buf, sink)?,
            GeomType::MultiLineString => {
                for g in span(&self.geometries, i)? {
                    self.line(g, &map, &mut buf, sink)?;
                }
            }
            GeomType::Polygon => self.polygon(part(i)?, &map, &mut buf, sink)?,
            GeomType::MultiPolygon => {
                for g in span(&self.geometries, i)? {
                    self.polygon(g, &map, &mut buf, sink)?;
                }
            }
        }
        Ok(())
    }

    fn vertex(&self, k: usize, map: &Scale) -> Result<[f64; 2], Error> {
        self.vertices.get(k).map(|&v| map.apply(v)).ok_or(Error::Topology)
    }

    /// Vertices `r` into `buf`, repeating the first at the end to `close`.
    fn run(&self, r: Range<usize>, close: bool, map: &Scale, buf: &mut Vec<[f64; 2]>) -> Result<(), Error> {
        buf.clear();
        for k in r {
            buf.push(self.vertex(k, map)?);
        }
        if close && !buf.is_empty() {
            buf.push(buf[0]);
        }
        Ok(())
    }

    /// Line `part`: its vertices come through the rings when polygons share
    /// the column, else straight from the parts.
    fn line<S: Sink>(&self, part: usize, map: &Scale, buf: &mut Vec<[f64; 2]>, sink: &mut S) -> Result<(), Error> {
        let r = match &self.rings {
            Some(_) => span(&self.rings, at(&self.parts, part)?)?,
            None => span(&self.parts, part)?,
        };
        self.run(r, false, map, buf)?;
        sink.begin_line();
        sink.points(buf);
        sink.end_line();
        Ok(())
    }

    fn polygon<S: Sink>(&self, part: usize, map: &Scale, buf: &mut Vec<[f64; 2]>, sink: &mut S) -> Result<(), Error> {
        sink.begin_polygon();
        for ring in span(&self.parts, part)? {
            self.run(span(&self.rings, ring)?, true, map, buf)?;
            if !buf.is_empty() {
                sink.begin_ring();
                sink.points(buf);
                sink.end_ring();
            }
        }
        sink.end_polygon();
        Ok(())
    }
}

/// Tile units to map units: the tile's top-left corner, and map units per
/// tile unit (Y flips).
struct Scale {
    origin: [f64; 2],
    per_unit: [f64; 2],
}

impl Scale {
    fn apply(&self, v: [i32; 2]) -> [f64; 2] {
        [self.origin[0] + v[0] as f64 * self.per_unit[0], self.origin[1] - v[1] as f64 * self.per_unit[1]]
    }
}

/// Entry `k` of an offset level.
fn at(level: &Option<Vec<u32>>, k: usize) -> Result<usize, Error> {
    level.as_ref().and_then(|o| o.get(k)).map(|&v| v as usize).ok_or(Error::Topology)
}

/// Item `k`'s range in the level below.
fn span(level: &Option<Vec<u32>>, k: usize) -> Result<Range<usize>, Error> {
    Ok(at(level, k)?..at(level, k + 1)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_and_zigzag() {
        let mut r = Reader { bytes: &[0xac, 0x02, 0x01] };
        assert_eq!(r.varint().unwrap(), 300);
        assert_eq!(r.varint().unwrap(), 1);
        assert_eq!([0, 1, 2, 3, 4].map(unzigzag32), [0, -1, 1, -2, 2]);
    }

    #[test]
    fn runs_expand_and_must_add_up() {
        assert_eq!(expand_runs(&[3, 4, 5, 3], 7).unwrap(), [5, 5, 5, 3, 3, 3, 3]);
        assert_eq!(expand_runs(&[3, 4, 5, 3], 8), Err(Error::StreamSize));
        assert_eq!(expand_runs(&[u64::MAX, 1], 3), Err(Error::StreamSize));
    }

    #[test]
    fn morton_codes_decode() {
        // (5, 3) is code 27 (encodings.md § Morton).
        assert_eq!((compact(27), compact(27 >> 1)), (5, 3));
    }

    #[test]
    fn length_streams_follow_the_threshold_rules() {
        use GeomType::*;
        // v1.md's example: [Point, MultiPolygon, Polygon] with geometries [3].
        assert_eq!(root(&[Point, MultiPolygon, Polygon], &[3], Polygon).unwrap(), [0, 1, 4, 5]);
        assert_eq!(root(&[MultiPolygon], &[], Polygon), Err(Error::Topology));
    }

    /// The cross-language FastPFOR vectors (`tests/fixtures/fetch_mlt_fixtures.sh`).
    #[test]
    #[ignore = "needs tests/fixtures/local/mlt/fastpfor"]
    fn fastpfor_vectors() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/local/mlt/fastpfor");
        for n in 1..=4 {
            let encoded = std::fs::read(format!("{dir}/vector{n}_encoded.bin")).unwrap();
            let decoded = std::fs::read(format!("{dir}/vector{n}_decoded.bin")).unwrap();
            let want: Vec<u32> = decoded.chunks_exact(4).map(|c| u32::from_be_bytes(c.try_into().unwrap())).collect();
            assert_eq!(fastpfor(&encoded, want.len()).unwrap(), want, "vector {n}");
        }
    }

    #[test]
    fn foreign_frames_are_skipped() {
        // A tag-0x02 frame of two bytes, then nothing else.
        assert_eq!(layers(&[0x02, 0x02, 0xff]).count(), 0);
        assert!(layers(&[0x05, 0x01]).next().unwrap().is_err());
    }
}
