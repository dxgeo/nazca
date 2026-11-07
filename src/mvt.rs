//! Mapbox Vector Tile decoding, for third-party basemaps (`../plans/rendering.org`
//! § V1 FEATURES). Enabled by the `mvt` feature, which adds no dependency.
//!
//! A tile is protocol buffers: [`layers`] walks its layers and
//! [`Layer::features`] a layer's features, borrowing the tile's bytes
//! throughout. [`Feature::read`] decodes one feature's command stream
//! straight into a [`Sink`], in map units, with no geometry built in between.
//!
//! Only geometry and feature ids are decoded. Properties are skipped, as
//! nazca holds no attributes (`../project.org` § WHAT NAZCA DOES NOT DO); an
//! embedder that needs one reads it with its own decoder.
//!
//! Reading the tile out of its container (PMTiles' directory, an MBTiles
//! row, an HTTP response) and decompressing it are the embedder's I/O.

use crate::grid::Rect;
use crate::sink::Sink;
use std::fmt;

/// Why a tile could not be read.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The bytes end inside a field.
    Truncated,
    /// A varint longer than ten bytes.
    Varint,
    /// A protocol buffer wire type that MVT never uses (a group).
    WireType(u8),
    /// A layer name that is not UTF-8.
    Name,
    /// A layer whose extent is zero.
    Extent,
    /// A geometry type outside MVT's four.
    GeomType(u64),
    /// A command other than MoveTo, LineTo and ClosePath, or one that the
    /// geometry type does not allow where it stands.
    Command(u32),
    /// A command whose parameters run past the end of the geometry.
    Parameters,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated => write!(f, "vector tile ends inside a field"),
            Error::Varint => write!(f, "vector tile varint longer than ten bytes"),
            Error::WireType(t) => write!(f, "vector tile uses protocol buffer wire type {t}, which MVT never does"),
            Error::Name => write!(f, "vector tile layer name is not UTF-8"),
            Error::Extent => write!(f, "vector tile layer has an extent of zero"),
            Error::GeomType(t) => write!(f, "vector tile feature has geometry type {t}"),
            Error::Command(c) => write!(f, "vector tile geometry has command {c} where it is not allowed"),
            Error::Parameters => write!(f, "vector tile geometry command runs past the end of its parameters"),
        }
    }
}

impl std::error::Error for Error {}

/// A feature's geometry type, as the tile declares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeomType {
    Unknown,
    Point,
    LineString,
    Polygon,
}

/// One protocol buffer field: its number and its value.
enum Value<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
    Fixed,
}

/// Reads the fields of one message in order.
struct Fields<'a> {
    bytes: &'a [u8],
}

impl<'a> Iterator for Fields<'a> {
    type Item = Result<(u32, Value<'a>), Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bytes.is_empty() {
            return None;
        }
        Some(self.field())
    }
}

impl<'a> Fields<'a> {
    fn field(&mut self) -> Result<(u32, Value<'a>), Error> {
        let key = varint(&mut self.bytes)?;
        let number = (key >> 3) as u32;
        let value = match (key & 7) as u8 {
            0 => Value::Varint(varint(&mut self.bytes)?),
            1 => {
                take(&mut self.bytes, 8)?;
                Value::Fixed
            }
            2 => {
                let len = usize::try_from(varint(&mut self.bytes)?).map_err(|_| Error::Truncated)?;
                Value::Bytes(take(&mut self.bytes, len)?)
            }
            5 => {
                take(&mut self.bytes, 4)?;
                Value::Fixed
            }
            t => return Err(Error::WireType(t)),
        };
        Ok((number, value))
    }
}

fn take<'a>(bytes: &mut &'a [u8], n: usize) -> Result<&'a [u8], Error> {
    if bytes.len() < n {
        return Err(Error::Truncated);
    }
    let (head, rest) = bytes.split_at(n);
    *bytes = rest;
    Ok(head)
}

fn varint(bytes: &mut &[u8]) -> Result<u64, Error> {
    let mut value = 0u64;
    for i in 0..10 {
        let Some((&b, rest)) = bytes.split_first() else { return Err(Error::Truncated) };
        *bytes = rest;
        value |= u64::from(b & 0x7f) << (7 * i);
        if b < 0x80 {
            return Ok(value);
        }
    }
    Err(Error::Varint)
}

/// The layers of a tile, in order.
pub fn layers(tile: &[u8]) -> Layers<'_> {
    Layers { fields: Fields { bytes: tile } }
}

/// See [`layers`].
pub struct Layers<'a> {
    fields: Fields<'a>,
}

impl<'a> Iterator for Layers<'a> {
    type Item = Result<Layer<'a>, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.fields.next()? {
                Err(e) => return Some(Err(e)),
                Ok((3, Value::Bytes(b))) => return Some(Layer::parse(b)),
                Ok(_) => {}
            }
        }
    }
}

/// One layer of a tile.
#[derive(Debug, Clone, Copy)]
pub struct Layer<'a> {
    bytes: &'a [u8],
    name: &'a str,
    extent: u32,
}

impl<'a> Layer<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Layer<'a>, Error> {
        let (mut name, mut extent) = ("", 4096u32);
        for field in (Fields { bytes }) {
            match field? {
                (1, Value::Bytes(b)) => name = std::str::from_utf8(b).map_err(|_| Error::Name)?,
                (5, Value::Varint(v)) => extent = u32::try_from(v).map_err(|_| Error::Extent)?,
                _ => {}
            }
        }
        if extent == 0 {
            return Err(Error::Extent);
        }
        Ok(Layer { bytes, name, extent })
    }

    pub fn name(&self) -> &'a str {
        self.name
    }

    /// Tile units across the tile; 4096 unless the layer says otherwise.
    pub fn extent(&self) -> u32 {
        self.extent
    }

    /// The layer's features, in order.
    pub fn features(&self) -> Features<'a> {
        Features { fields: Fields { bytes: self.bytes }, extent: self.extent }
    }
}

/// See [`Layer::features`].
pub struct Features<'a> {
    fields: Fields<'a>,
    extent: u32,
}

impl<'a> Iterator for Features<'a> {
    type Item = Result<Feature<'a>, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.fields.next()? {
                Err(e) => return Some(Err(e)),
                Ok((2, Value::Bytes(b))) => return Some(Feature::parse(b, self.extent)),
                Ok(_) => {}
            }
        }
    }
}

/// One feature: its id, type and undecoded geometry.
#[derive(Debug, Clone, Copy)]
pub struct Feature<'a> {
    id: Option<u64>,
    kind: GeomType,
    geometry: &'a [u8],
    extent: u32,
}

impl<'a> Feature<'a> {
    fn parse(bytes: &'a [u8], extent: u32) -> Result<Feature<'a>, Error> {
        let mut feature = Feature { id: None, kind: GeomType::Unknown, geometry: &[], extent };
        for field in (Fields { bytes }) {
            match field? {
                (1, Value::Varint(v)) => feature.id = Some(v),
                (3, Value::Varint(v)) => {
                    feature.kind = match v {
                        0 => GeomType::Unknown,
                        1 => GeomType::Point,
                        2 => GeomType::LineString,
                        3 => GeomType::Polygon,
                        t => return Err(Error::GeomType(t)),
                    }
                }
                (4, Value::Bytes(b)) => feature.geometry = b,
                _ => {}
            }
        }
        Ok(feature)
    }

    /// The feature's id, if the tile gives one.
    pub fn id(&self) -> Option<u64> {
        self.id
    }

    pub fn geom_type(&self) -> GeomType {
        self.kind
    }

    /// Decodes the geometry into `sink`, mapping tile units to map units so
    /// that the tile's extent spans `bounds` (Y flips: tile rows run down,
    /// map Y up). Polygon rings are grouped by winding, as MVT 2 defines:
    /// a ring of positive area in tile units starts a polygon, and rings of
    /// negative area are its holes. A ring of no area is dropped. Rings
    /// arrive closed.
    ///
    /// A feature of unknown type draws nothing.
    pub fn read<S: Sink>(&self, bounds: Rect, sink: &mut S) -> Result<(), Error> {
        let scale = [(bounds.max[0] - bounds.min[0]) / self.extent as f64, (bounds.max[1] - bounds.min[1]) / self.extent as f64];
        let to_map = |p: [i64; 2]| [bounds.min[0] + p[0] as f64 * scale[0], bounds.max[1] - p[1] as f64 * scale[1]];
        let mut commands = Commands { bytes: self.geometry, cursor: [0, 0] };
        match self.kind {
            GeomType::Unknown => Ok(()),
            GeomType::Point => {
                while let Some((id, count)) = commands.command()? {
                    if id != MOVE_TO {
                        return Err(Error::Command(id));
                    }
                    for _ in 0..count {
                        sink.point(to_map(commands.point()?));
                    }
                }
                Ok(())
            }
            GeomType::LineString => {
                let mut line = Vec::new();
                loop {
                    let start = match commands.command()? {
                        None => return Ok(()),
                        Some((MOVE_TO, 1)) => commands.point()?,
                        Some((id, _)) => return Err(Error::Command(id)),
                    };
                    line.clear();
                    line.push(to_map(start));
                    match commands.command()? {
                        Some((LINE_TO, count)) => {
                            for _ in 0..count {
                                line.push(to_map(commands.point()?));
                            }
                        }
                        Some((id, _)) => return Err(Error::Command(id)),
                        None => return Err(Error::Parameters),
                    }
                    sink.begin_line();
                    sink.points(&line);
                    sink.end_line();
                }
            }
            GeomType::Polygon => {
                let mut ring: Vec<[i64; 2]> = Vec::new();
                let mut mapped = Vec::new();
                let mut open = false;
                loop {
                    let start = match commands.command()? {
                        None => break,
                        Some((MOVE_TO, 1)) => commands.point()?,
                        Some((id, _)) => return Err(Error::Command(id)),
                    };
                    ring.clear();
                    ring.push(start);
                    match commands.command()? {
                        Some((LINE_TO, count)) => {
                            for _ in 0..count {
                                ring.push(commands.point()?);
                            }
                        }
                        Some((id, _)) => return Err(Error::Command(id)),
                        None => return Err(Error::Parameters),
                    }
                    match commands.command()? {
                        Some((CLOSE_PATH, _)) => {}
                        Some((id, _)) => return Err(Error::Command(id)),
                        None => return Err(Error::Parameters),
                    }
                    let area = twice_area(&ring);
                    if area == 0 {
                        continue;
                    }
                    if area > 0 {
                        if open {
                            sink.end_polygon();
                        }
                        sink.begin_polygon();
                        open = true;
                    } else if !open {
                        // A hole before any exterior has nothing to cut.
                        continue;
                    }
                    mapped.clear();
                    mapped.extend(ring.iter().map(|&p| to_map(p)));
                    mapped.push(mapped[0]);
                    sink.begin_ring();
                    sink.points(&mapped);
                    sink.end_ring();
                }
                if open {
                    sink.end_polygon();
                }
                Ok(())
            }
        }
    }
}

const MOVE_TO: u32 = 1;
const LINE_TO: u32 = 2;
const CLOSE_PATH: u32 = 7;

/// Twice a ring's signed area in tile units, by the surveyor's formula MVT
/// uses: positive for a ring clockwise on screen (Y down).
fn twice_area(ring: &[[i64; 2]]) -> i128 {
    let n = ring.len();
    (0..n)
        .map(|i| {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            i128::from(a[0]) * i128::from(b[1]) - i128::from(b[0]) * i128::from(a[1])
        })
        .sum()
}

/// A geometry's command stream: packed varints, commands and zigzag deltas.
struct Commands<'a> {
    bytes: &'a [u8],
    cursor: [i64; 2],
}

impl Commands<'_> {
    /// The next command's id and count, or `None` at the end.
    fn command(&mut self) -> Result<Option<(u32, u32)>, Error> {
        if self.bytes.is_empty() {
            return Ok(None);
        }
        let c = varint(&mut self.bytes)? as u32;
        let (id, count) = (c & 7, c >> 3);
        match id {
            MOVE_TO | LINE_TO | CLOSE_PATH => Ok(Some((id, count))),
            _ => Err(Error::Command(id)),
        }
    }

    /// The next parameter pair, moving the cursor.
    fn point(&mut self) -> Result<[i64; 2], Error> {
        if self.bytes.is_empty() {
            return Err(Error::Parameters);
        }
        let dx = unzigzag(varint(&mut self.bytes)? as u32);
        let dy = unzigzag(varint(&mut self.bytes).map_err(|_| Error::Parameters)? as u32);
        self.cursor = [self.cursor[0] + i64::from(dx), self.cursor[1] + i64::from(dy)];
        Ok(self.cursor)
    }
}

fn unzigzag(u: u32) -> i32 {
    ((u >> 1) as i32) ^ -((u & 1) as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records what a sink receives, as text.
    #[derive(Default)]
    struct Log(Vec<String>);

    impl Sink for Log {
        fn point(&mut self, xy: [f64; 2]) {
            self.0.push(format!("pt {} {}", xy[0], xy[1]));
        }
        fn points(&mut self, xy: &[[f64; 2]]) {
            let s: Vec<String> = xy.iter().map(|p| format!("{} {}", p[0], p[1])).collect();
            self.0.push(s.join(","));
        }
        fn begin_line(&mut self) {
            self.0.push("line".into());
        }
        fn begin_polygon(&mut self) {
            self.0.push("poly".into());
        }
        fn begin_ring(&mut self) {
            self.0.push("ring".into());
        }
    }

    // A minimal protocol buffer writer for building tiles in tests.
    fn put_varint(out: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }
    fn put_bytes(out: &mut Vec<u8>, field: u64, b: &[u8]) {
        put_varint(out, field << 3 | 2);
        put_varint(out, b.len() as u64);
        out.extend_from_slice(b);
    }
    fn put_uint(out: &mut Vec<u8>, field: u64, v: u64) {
        put_varint(out, field << 3);
        put_varint(out, v);
    }
    fn zz(v: i32) -> u64 {
        ((v << 1) ^ (v >> 31)) as u32 as u64
    }
    fn geometry(cmds: &[(u32, &[[i32; 2]])]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut cur = [0, 0];
        for &(id, pts) in cmds {
            let count = if id == CLOSE_PATH { 1 } else { pts.len() as u32 };
            put_varint(&mut out, u64::from(count << 3 | id));
            for p in pts {
                put_varint(&mut out, zz(p[0] - cur[0]));
                put_varint(&mut out, zz(p[1] - cur[1]));
                cur = *p;
            }
        }
        out
    }
    fn tile(features: &[(Option<u64>, u64, Vec<u8>)], extent: Option<u64>) -> Vec<u8> {
        let mut layer = Vec::new();
        put_uint(&mut layer, 15, 2);
        put_bytes(&mut layer, 1, b"roads");
        for (id, kind, geom) in features {
            let mut f = Vec::new();
            if let Some(id) = id {
                put_uint(&mut f, 1, *id);
            }
            put_bytes(&mut f, 2, &[0, 0]); // tags, skipped
            put_uint(&mut f, 3, *kind);
            put_bytes(&mut f, 4, geom);
            put_bytes(&mut layer, 2, &f);
        }
        put_bytes(&mut layer, 3, b"name"); // a key, skipped
        if let Some(e) = extent {
            put_uint(&mut layer, 5, e);
        }
        let mut t = Vec::new();
        put_bytes(&mut t, 3, &layer);
        t
    }

    const UNIT: Rect = Rect { min: [0.0, 0.0], max: [8.0, 8.0] };

    fn read_all(t: &[u8]) -> Vec<String> {
        let mut log = Log::default();
        for layer in layers(t) {
            for f in layer.unwrap().features() {
                f.unwrap().read(UNIT, &mut log).unwrap();
            }
        }
        log.0
    }

    #[test]
    fn layers_name_extent_and_ids() {
        let t = tile(&[(Some(7), 1, geometry(&[(MOVE_TO, &[[1, 2]])])), (None, 1, geometry(&[(MOVE_TO, &[[3, 4]])]))], Some(8));
        let layer = layers(&t).next().unwrap().unwrap();
        assert_eq!((layer.name(), layer.extent()), ("roads", 8));
        let ids: Vec<_> = layer.features().map(|f| f.unwrap().id()).collect();
        assert_eq!(ids, [Some(7), None]);
        // No extent field: 4096.
        let t = tile(&[], None);
        assert_eq!(layers(&t).next().unwrap().unwrap().extent(), 4096);
    }

    #[test]
    fn points_and_lines_map_to_the_bounds() {
        let pts = geometry(&[(MOVE_TO, &[[0, 0], [8, 8]])]);
        let line = geometry(&[(MOVE_TO, &[[0, 8]]), (LINE_TO, &[[4, 8], [4, 4]]), (MOVE_TO, &[[1, 1]]), (LINE_TO, &[[2, 2]])]);
        let t = tile(&[(None, 1, pts), (None, 2, line)], Some(8));
        // Tile Y runs down, map Y up: tile (0, 0) is the top-left corner.
        assert_eq!(read_all(&t), ["pt 0 8", "pt 8 0", "line", "0 0,4 0,4 4", "line", "1 7,2 6"]);
    }

    #[test]
    fn polygons_group_by_winding() {
        // Clockwise on screen (positive area): an exterior. Its hole runs
        // the other way. Then a second exterior: a second polygon.
        let outer: &[[i32; 2]] = &[[0, 0], [8, 0], [8, 8], [0, 8]];
        let hole: &[[i32; 2]] = &[[2, 2], [2, 6], [6, 6], [6, 2]];
        let second: &[[i32; 2]] = &[[10, 10], [12, 10], [12, 12]];
        let ring = |r: &[[i32; 2]]| -> Vec<(u32, Vec<[i32; 2]>)> {
            vec![(MOVE_TO, r[..1].to_vec()), (LINE_TO, r[1..].to_vec()), (CLOSE_PATH, vec![])]
        };
        let cmds: Vec<(u32, Vec<[i32; 2]>)> = [ring(outer), ring(hole), ring(second)].concat();
        let cmds: Vec<(u32, &[[i32; 2]])> = cmds.iter().map(|(c, p)| (*c, p.as_slice())).collect();
        let t = tile(&[(None, 3, geometry(&cmds))], Some(8));
        let log = read_all(&t);
        assert_eq!(log.iter().filter(|s| *s == "poly").count(), 2);
        assert_eq!(&log[..5], ["poly", "ring", "0 8,8 8,8 0,0 0,0 8", "ring", "2 6,2 2,6 2,6 6,2 6"]);
    }

    #[test]
    fn broken_tiles_are_errors_not_panics() {
        let t = tile(&[(None, 2, geometry(&[(MOVE_TO, &[[1, 1]]), (LINE_TO, &[[2, 2]])]))], Some(8));
        for n in 0..t.len() {
            let mut log = Log::default();
            let _ = layers(&t[..n]).try_for_each(|l| {
                l?.features().try_for_each(|f| f?.read(UNIT, &mut log))
            });
        }
        // A command a point may not take.
        let t = tile(&[(None, 1, geometry(&[(LINE_TO, &[[1, 1]])]))], Some(8));
        let f = layers(&t).next().unwrap().unwrap().features().next().unwrap().unwrap();
        assert_eq!(f.read(UNIT, &mut Log::default()), Err(Error::Command(LINE_TO)));
        assert_eq!(layers(&[0x1b]).next().unwrap().unwrap_err(), Error::WireType(3));
    }
}
