//! A streaming Well-Known Binary reader that drives a [`Sink`].
//!
//! It never builds a geometry and never allocates: coordinates are decoded
//! from the borrowed bytes in blocks of [`BLOCK`] into a stack buffer and
//! handed to the sink a block at a time. That is the point of nazca having
//! its own reader rather than linking `geosetta`'s, which decodes into an
//! owned `Geometry` tree (a format converter needs the whole geometry in
//! hand; a renderer only needs to see each coordinate once).
//!
//! Accepted, so that bytes from any producer work:
//! - both byte orders, chosen independently by every nested geometry;
//! - ISO SFA dimension codes (`1001` = `PointZ`, `2001` = `PointM`,
//!   `3001` = `PointZM`), which DuckDB and GeoParquet producers emit;
//! - PostGIS EWKB flags (`0x80000000` Z, `0x40000000` M, `0x20000000` SRID,
//!   whose SRID value is skipped);
//! - the seven OGC types, `Point` through `GeometryCollection`.
//!
//! Z and M are read past, not passed on: the sink is 2D (`Sink`'s docs). An
//! empty point, which WKB writes as `NaN NaN`, produces no event.
//!
//! Curves, surfaces and the other SQL/MM types are rejected with
//! [`Error::UnsupportedType`] rather than skipped, so a caller is never
//! handed a silently incomplete drawing.

use crate::sink::Sink;
use std::fmt;

/// Coordinates decoded per `Sink::points` call. 256 X/Y pairs are 4 KiB of
/// stack.
pub const BLOCK: usize = 256;

/// How deeply `GeometryCollection`s may nest. Real data nests once or twice;
/// the limit exists so hostile input cannot exhaust the stack.
pub const MAX_DEPTH: usize = 32;

/// Why a WKB geometry could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The bytes ended before the geometry did.
    Truncated,
    /// A byte-order marker other than `0x00` (big-endian) or `0x01`
    /// (little-endian). Carries the byte.
    ByteOrder(u8),
    /// A type code that is not a WKB geometry type. Carries the raw code.
    UnknownType(u32),
    /// A real WKB type nazca does not draw (curves, surfaces, TIN, ...).
    /// Carries the base type number, e.g. `8` for `CircularString`.
    UnsupportedType(u32),
    /// A type code with both EWKB dimension flags and an ISO dimension
    /// offset, which no producer writes. Carries the raw code.
    MixedDimensionStyle(u32),
    /// A multi-geometry member of the wrong type, e.g. a point inside a
    /// `MultiPolygon`. Carries the parent's and the member's base types.
    MemberType { parent: u32, member: u32 },
    /// Collections nested deeper than [`MAX_DEPTH`].
    TooDeep,
    /// Bytes left over after the geometry ended. Carries how many; usually
    /// a sign the caller sliced the buffer wrongly (a GeoPackage blob still
    /// has its header, say).
    TrailingBytes(usize),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated => write!(f, "truncated WKB geometry"),
            Error::ByteOrder(b) => write!(f, "invalid WKB byte-order marker 0x{b:02x}"),
            Error::UnknownType(c) => write!(f, "unknown WKB type code {c}"),
            Error::UnsupportedType(t) => write!(f, "unsupported WKB geometry type {t}"),
            Error::MixedDimensionStyle(c) => {
                write!(f, "WKB type code 0x{c:08x} mixes EWKB flags with an ISO dimension offset")
            }
            Error::MemberType { parent, member } => {
                write!(f, "WKB type {parent} cannot contain a member of type {member}")
            }
            Error::TooDeep => write!(f, "WKB collections nested deeper than {MAX_DEPTH}"),
            Error::TrailingBytes(n) => write!(f, "{n} bytes after the end of the WKB geometry"),
        }
    }
}

impl std::error::Error for Error {}

const POINT: u32 = 1;
const LINESTRING: u32 = 2;
const POLYGON: u32 = 3;
const MULTIPOINT: u32 = 4;
const MULTILINESTRING: u32 = 5;
const MULTIPOLYGON: u32 = 6;
const GEOMETRYCOLLECTION: u32 = 7;

// The SQL/MM types after the OGC seven: CircularString (8) through
// Triangle (17). Recognized so they fail as unsupported, not unknown.
const LAST_SQL_MM: u32 = 17;

const EWKB_Z: u32 = 0x8000_0000;
const EWKB_M: u32 = 0x4000_0000;
const EWKB_SRID: u32 = 0x2000_0000;
const EWKB_FLAGS: u32 = EWKB_Z | EWKB_M | EWKB_SRID;

/// Reads one WKB geometry from `wkb` into `sink`.
///
/// `wkb` must hold exactly one geometry; leftover bytes are an error
/// ([`Error::TrailingBytes`]). On any error the sink may already have
/// received part of the geometry (see [`Sink`]).
pub fn read<S: Sink + ?Sized>(wkb: &[u8], sink: &mut S) -> Result<(), Error> {
    let mut r = Reader { buf: wkb, pos: 0 };
    r.geometry(sink, None, 0)?;
    match wkb.len() - r.pos {
        0 => Ok(()),
        n => Err(Error::TrailingBytes(n)),
    }
}

#[derive(Clone, Copy)]
enum Order {
    Big,
    Little,
}

impl Order {
    fn u32(self, b: [u8; 4]) -> u32 {
        match self {
            Order::Big => u32::from_be_bytes(b),
            Order::Little => u32::from_le_bytes(b),
        }
    }

    fn f64(self, b: [u8; 8]) -> f64 {
        match self {
            Order::Big => f64::from_be_bytes(b),
            Order::Little => f64::from_le_bytes(b),
        }
    }

    /// X and Y from the front of one coordinate's bytes.
    fn xy(self, c: &[u8]) -> [f64; 2] {
        let x = c[0..8].try_into().unwrap();
        let y = c[8..16].try_into().unwrap();
        [self.f64(x), self.f64(y)]
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.pos.checked_add(n).ok_or(Error::Truncated)?;
        let s = self.buf.get(self.pos..end).ok_or(Error::Truncated)?;
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self, order: Order) -> Result<u32, Error> {
        Ok(order.u32(self.take(4)?.try_into().unwrap()))
    }

    /// One geometry, header included. `parent` is the enclosing
    /// multi-geometry's base type, which constrains what this one may be.
    fn geometry<S: Sink + ?Sized>(
        &mut self,
        sink: &mut S,
        parent: Option<u32>,
        depth: usize,
    ) -> Result<(), Error> {
        if depth > MAX_DEPTH {
            return Err(Error::TooDeep);
        }
        let order = match self.take(1)?[0] {
            0 => Order::Big,
            1 => Order::Little,
            b => return Err(Error::ByteOrder(b)),
        };
        let code = self.u32(order)?;
        let (base, stride, has_srid) = decode_type(code)?;
        if has_srid {
            self.take(4)?;
        }
        if let Some(parent) = parent {
            let allowed = match parent {
                MULTIPOINT => base == POINT,
                MULTILINESTRING => base == LINESTRING,
                MULTIPOLYGON => base == POLYGON,
                _ => true,
            };
            if !allowed {
                return Err(Error::MemberType { parent, member: base });
            }
        }

        match base {
            POINT => {
                let c = self.take(stride)?;
                let xy = order.xy(c);
                if !(xy[0].is_nan() && xy[1].is_nan()) {
                    sink.point(xy);
                }
            }
            LINESTRING => {
                sink.begin_line();
                self.run(sink, order, stride)?;
                sink.end_line();
            }
            POLYGON => {
                sink.begin_polygon();
                for _ in 0..self.u32(order)? {
                    sink.begin_ring();
                    self.run(sink, order, stride)?;
                    sink.end_ring();
                }
                sink.end_polygon();
            }
            _ => {
                // MULTI* and GEOMETRYCOLLECTION: decode_type admits nothing else.
                for _ in 0..self.u32(order)? {
                    self.geometry(sink, Some(base), depth + 1)?;
                }
            }
        }
        Ok(())
    }

    /// A counted run of coordinates (a line, or one ring), in blocks.
    fn run<S: Sink + ?Sized>(&mut self, sink: &mut S, order: Order, stride: usize) -> Result<(), Error> {
        let n = self.u32(order)? as usize;
        let bytes = self.take(n.checked_mul(stride).ok_or(Error::Truncated)?)?;
        let mut block = [[0.0; 2]; BLOCK];
        for chunk in bytes.chunks(stride * BLOCK) {
            let mut k = 0;
            for c in chunk.chunks_exact(stride) {
                block[k] = order.xy(c);
                k += 1;
            }
            sink.points(&block[..k]);
        }
        Ok(())
    }
}

/// Splits a type code into its base type, the byte size of one coordinate,
/// and whether an EWKB SRID follows.
fn decode_type(code: u32) -> Result<(u32, usize, bool), Error> {
    let ewkb_z = code & EWKB_Z != 0;
    let ewkb_m = code & EWKB_M != 0;
    let has_srid = code & EWKB_SRID != 0;
    let iso = code & !EWKB_FLAGS;
    let (base, iso_z, iso_m) = match iso / 1000 {
        0 => (iso, false, false),
        1 => (iso - 1000, true, false),
        2 => (iso - 2000, false, true),
        3 => (iso - 3000, true, true),
        _ => return Err(Error::UnknownType(code)),
    };
    if (ewkb_z || ewkb_m) && (iso_z || iso_m) {
        return Err(Error::MixedDimensionStyle(code));
    }
    match base {
        POINT..=GEOMETRYCOLLECTION => {}
        8..=LAST_SQL_MM => return Err(Error::UnsupportedType(base)),
        _ => return Err(Error::UnknownType(code)),
    }
    let z = ewkb_z || iso_z;
    let m = ewkb_m || iso_m;
    let stride = 16 + if z { 8 } else { 0 } + if m { 8 } else { 0 };
    Ok((base, stride, has_srid))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every event a sink receives, with `points` runs merged so tests don't
    /// depend on where the reader splits them.
    #[derive(Debug, PartialEq)]
    enum Event {
        Point([f64; 2]),
        BeginLine,
        EndLine,
        BeginPolygon,
        EndPolygon,
        BeginRing,
        EndRing,
        Points(Vec<[f64; 2]>),
    }

    #[derive(Default)]
    struct Recorder {
        events: Vec<Event>,
        calls: usize,
    }

    impl Sink for Recorder {
        fn point(&mut self, xy: [f64; 2]) { self.events.push(Event::Point(xy)); }
        fn points(&mut self, xy: &[[f64; 2]]) {
            self.calls += 1;
            if let Some(Event::Points(run)) = self.events.last_mut() {
                run.extend_from_slice(xy);
            } else {
                self.events.push(Event::Points(xy.to_vec()));
            }
        }
        fn begin_line(&mut self) { self.events.push(Event::BeginLine); }
        fn end_line(&mut self) { self.events.push(Event::EndLine); }
        fn begin_polygon(&mut self) { self.events.push(Event::BeginPolygon); }
        fn end_polygon(&mut self) { self.events.push(Event::EndPolygon); }
        fn begin_ring(&mut self) { self.events.push(Event::BeginRing); }
        fn end_ring(&mut self) { self.events.push(Event::EndRing); }
    }

    fn record(wkb: &[u8]) -> Result<Vec<Event>, Error> {
        let mut r = Recorder::default();
        read(wkb, &mut r)?;
        Ok(r.events)
    }

    /// Builds WKB by hand, in either byte order, so tests state exactly
    /// which bytes they feed the reader.
    struct W {
        big: bool,
        out: Vec<u8>,
    }

    impl W {
        fn le() -> W { W { big: false, out: Vec::new() } }
        fn be() -> W { W { big: true, out: Vec::new() } }
        fn u32(mut self, v: u32) -> W {
            self.out.extend_from_slice(&if self.big { v.to_be_bytes() } else { v.to_le_bytes() });
            self
        }
        fn f64s(mut self, vs: &[f64]) -> W {
            for v in vs {
                self.out.extend_from_slice(&if self.big { v.to_be_bytes() } else { v.to_le_bytes() });
            }
            self
        }
        fn header(mut self, code: u32) -> W {
            self.out.push(if self.big { 0 } else { 1 });
            self.u32(code)
        }
        fn bytes(mut self, b: &[u8]) -> W {
            self.out.extend_from_slice(b);
            self
        }
    }

    fn square(x: f64, y: f64, size: f64) -> Vec<f64> {
        vec![x, y, x + size, y, x + size, y + size, x, y + size, x, y]
    }

    fn pairs(flat: &[f64]) -> Vec<[f64; 2]> {
        flat.chunks(2).map(|c| [c[0], c[1]]).collect()
    }

    #[test]
    fn point_in_both_byte_orders() {
        let le = W::le().header(1).f64s(&[1.5, -2.5]).out;
        let be = W::be().header(1).f64s(&[1.5, -2.5]).out;
        assert_eq!(record(&le).unwrap(), vec![Event::Point([1.5, -2.5])]);
        assert_eq!(record(&be).unwrap(), vec![Event::Point([1.5, -2.5])]);
    }

    #[test]
    fn z_and_m_are_read_past_in_iso_and_ewkb_codes() {
        let cases = [
            (1001, vec![1.0, 2.0, 3.0]),
            (2001, vec![1.0, 2.0, 4.0]),
            (3001, vec![1.0, 2.0, 3.0, 4.0]),
            (EWKB_Z | 1, vec![1.0, 2.0, 3.0]),
            (EWKB_M | 1, vec![1.0, 2.0, 4.0]),
            (EWKB_Z | EWKB_M | 1, vec![1.0, 2.0, 3.0, 4.0]),
        ];
        for (code, coords) in cases {
            let wkb = W::le().header(code).f64s(&coords).out;
            assert_eq!(record(&wkb).unwrap(), vec![Event::Point([1.0, 2.0])], "code {code:#x}");
        }
        // A ZM line: the stride must skip both extra ordinates per vertex.
        let wkb = W::le().header(3002).u32(2).f64s(&[1.0, 2.0, 9.0, 9.0, 3.0, 4.0, 9.0, 9.0]).out;
        assert_eq!(
            record(&wkb).unwrap(),
            vec![Event::BeginLine, Event::Points(vec![[1.0, 2.0], [3.0, 4.0]]), Event::EndLine]
        );
    }

    #[test]
    fn ewkb_srid_is_skipped() {
        let wkb = W::le().header(EWKB_SRID | 1).u32(4326).f64s(&[7.0, 8.0]).out;
        assert_eq!(record(&wkb).unwrap(), vec![Event::Point([7.0, 8.0])]);
    }

    #[test]
    fn empty_point_produces_no_event() {
        let wkb = W::le().header(1).f64s(&[f64::NAN, f64::NAN]).out;
        assert_eq!(record(&wkb).unwrap(), vec![]);
    }

    #[test]
    fn long_line_arrives_in_blocks_but_in_order() {
        let n = BLOCK * 2 + 17;
        let flat: Vec<f64> = (0..n).flat_map(|i| [i as f64, -(i as f64)]).collect();
        let wkb = W::le().header(2).u32(n as u32).f64s(&flat).out;
        let mut r = Recorder::default();
        read(&wkb, &mut r).unwrap();
        assert_eq!(r.calls, 3);
        assert_eq!(r.events, vec![Event::BeginLine, Event::Points(pairs(&flat)), Event::EndLine]);
    }

    #[test]
    fn polygon_with_hole() {
        let outer = square(0.0, 0.0, 4.0);
        let hole = square(1.0, 1.0, 1.0);
        let wkb = W::le().header(3).u32(2).u32(5).f64s(&outer).u32(5).f64s(&hole).out;
        assert_eq!(
            record(&wkb).unwrap(),
            vec![
                Event::BeginPolygon,
                Event::BeginRing,
                Event::Points(pairs(&outer)),
                Event::EndRing,
                Event::BeginRing,
                Event::Points(pairs(&hole)),
                Event::EndRing,
                Event::EndPolygon,
            ]
        );
    }

    #[test]
    fn empty_line_and_polygon_still_bracket() {
        let line = W::le().header(2).u32(0).out;
        assert_eq!(record(&line).unwrap(), vec![Event::BeginLine, Event::EndLine]);
        let poly = W::le().header(3).u32(0).out;
        assert_eq!(record(&poly).unwrap(), vec![Event::BeginPolygon, Event::EndPolygon]);
    }

    #[test]
    fn multipolygon_members_may_each_pick_a_byte_order() {
        let a = square(0.0, 0.0, 1.0);
        let b = square(5.0, 5.0, 1.0);
        let member_le = W::le().header(3).u32(1).u32(5).f64s(&a).out;
        let member_be = W::be().header(3).u32(1).u32(5).f64s(&b).out;
        let wkb = W::be().header(6).u32(2).bytes(&member_le).bytes(&member_be).out;
        let ring = |c: &[f64]| {
            vec![Event::BeginPolygon, Event::BeginRing, Event::Points(pairs(c)), Event::EndRing, Event::EndPolygon]
        };
        let mut expected = ring(&a);
        expected.extend(ring(&b));
        assert_eq!(record(&wkb).unwrap(), expected);
    }

    #[test]
    fn nested_collection() {
        let point = W::le().header(1).f64s(&[1.0, 1.0]).out;
        let inner = W::le().header(7).u32(1).bytes(&point).out;
        let line = W::le().header(2).u32(2).f64s(&[0.0, 0.0, 1.0, 1.0]).out;
        let wkb = W::le().header(7).u32(2).bytes(&inner).bytes(&line).out;
        assert_eq!(
            record(&wkb).unwrap(),
            vec![
                Event::Point([1.0, 1.0]),
                Event::BeginLine,
                Event::Points(vec![[0.0, 0.0], [1.0, 1.0]]),
                Event::EndLine,
            ]
        );
    }

    #[test]
    fn every_truncation_is_an_error_not_a_panic() {
        let member = W::le().header(3).u32(1).u32(5).f64s(&square(0.0, 0.0, 1.0)).out;
        let wkb = W::le().header(6).u32(2).bytes(&member).bytes(&member).out;
        assert!(record(&wkb).is_ok());
        for len in 0..wkb.len() {
            assert_eq!(record(&wkb[..len]), Err(Error::Truncated), "prefix of {len} bytes");
        }
    }

    #[test]
    fn huge_counts_fail_without_looping_or_allocating() {
        let line = W::le().header(2).u32(u32::MAX).out;
        assert_eq!(record(&line), Err(Error::Truncated));
        let poly = W::le().header(3).u32(u32::MAX).out;
        assert_eq!(record(&poly), Err(Error::Truncated));
        let multi = W::le().header(6).u32(u32::MAX).out;
        assert_eq!(record(&multi), Err(Error::Truncated));
    }

    #[test]
    fn malformed_headers() {
        assert_eq!(record(&[2, 1, 0, 0, 0]), Err(Error::ByteOrder(2)));
        assert_eq!(record(&W::le().header(99).out), Err(Error::UnknownType(99)));
        assert_eq!(record(&W::le().header(4001).out), Err(Error::UnknownType(4001)));
        assert_eq!(record(&W::le().header(8).out), Err(Error::UnsupportedType(8)));
        assert_eq!(record(&W::le().header(1017).out), Err(Error::UnsupportedType(17)));
        let mixed = EWKB_Z | 1001;
        assert_eq!(record(&W::le().header(mixed).out), Err(Error::MixedDimensionStyle(mixed)));
    }

    #[test]
    fn multi_member_types_are_checked() {
        let point = W::le().header(1).f64s(&[0.0, 0.0]).out;
        let wkb = W::le().header(6).u32(1).bytes(&point).out;
        assert_eq!(record(&wkb), Err(Error::MemberType { parent: 6, member: 1 }));
        // A Z member inside a 2D multi is still the right base type.
        let point_z = W::le().header(1001).f64s(&[0.0, 0.0, 0.0]).out;
        let wkb = W::le().header(4).u32(1).bytes(&point_z).out;
        assert_eq!(record(&wkb).unwrap(), vec![Event::Point([0.0, 0.0])]);
    }

    #[test]
    fn deep_nesting_is_refused() {
        let mut wkb = W::le().header(1).f64s(&[0.0, 0.0]).out;
        for _ in 0..=MAX_DEPTH {
            wkb = W::le().header(7).u32(1).bytes(&wkb).out;
        }
        assert_eq!(record(&wkb), Err(Error::TooDeep));
    }

    #[test]
    fn trailing_bytes_are_reported() {
        let wkb = W::le().header(1).f64s(&[0.0, 0.0]).bytes(&[0, 0, 0]).out;
        assert_eq!(record(&wkb), Err(Error::TrailingBytes(3)));
    }
}
