//! Tile grids: how a map's plane is cut into tiles at each level of detail.
//!
//! A grid is an origin plus, per level, the span of one tile in map units and
//! how many tiles fit across and down. XYZ Web Mercator is one grid
//! ([`TileGrid::web_mercator`]); any projected CRS can define its own
//! ([`TileGrid::quadtree`], or [`TileGrid::new`] for levels that do not halve),
//! which is what lets an interactive view use a per-view CRS
//! (`../plans/rendering.org` § TILE GRIDS).
//!
//! Everything here is in *map units*, the units of the coordinates the caller
//! hands nazca. Nothing converts longitude/latitude: that is CRS math, and it
//! belongs to `geoscribe` (`../project.org` § CHARTER).
//!
//! Rows count *down* from the origin, as XYZ tiles do: the origin is the grid's
//! top-left corner, row 0 is the top row, and map Y decreases as rows increase.

use std::fmt;

/// Half the width of the Web Mercator (EPSG:3857) square, in metres: pi times
/// the WGS84 semi-major axis, 6378137 m. The square spans this far either side
/// of 0 in both X and Y.
pub const WEB_MERCATOR_HALF_EXTENT: f64 = 20_037_508.342_789_244;

/// One tile: its level, column and row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileId {
    pub z: u8,
    pub x: u32,
    pub y: u32,
}

/// An axis-aligned rectangle in map units. `min` is the lower-left corner,
/// `max` the upper-right, with map Y increasing upward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub min: [f64; 2],
    pub max: [f64; 2],
}

/// One level of a grid: the span of a tile's edge in map units (tiles are
/// square) and the grid's size in tiles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Level {
    pub span: f64,
    pub cols: u32,
    pub rows: u32,
}

/// Why a grid could not be built.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// No levels at all.
    NoLevels,
    /// More levels than a [`TileId`]'s `u8` can number.
    TooManyLevels(usize),
    /// A level whose span is not a positive finite number, or that has no
    /// tiles. Carries the level.
    BadLevel(u8),
    /// A level whose tiles are not smaller than the previous level's, which
    /// would make choosing a level by resolution ambiguous. Carries the level.
    NotFiner(u8),
    /// An origin or tile size that is not usable (non-finite, or zero pixels).
    BadGeometry,
    /// A quadtree whose finest level has more tiles across than a `u32` holds.
    TooDeep(u8),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoLevels => write!(f, "a tile grid needs at least one level"),
            Error::TooManyLevels(n) => write!(f, "{n} levels is more than a tile grid can number (256)"),
            Error::BadLevel(z) => write!(f, "tile grid level {z} has no tiles or a span that is not positive"),
            Error::NotFiner(z) => write!(f, "tile grid level {z} has tiles no smaller than the level before it"),
            Error::BadGeometry => write!(f, "a tile grid's origin must be finite and its tiles at least one pixel"),
            Error::TooDeep(z) => write!(f, "a quadtree to level {z} has more tiles across than fit in a u32"),
        }
    }
}

impl std::error::Error for Error {}

/// A tile pyramid in map units. See the module docs.
#[derive(Debug, Clone, PartialEq)]
pub struct TileGrid {
    origin: [f64; 2],
    tile_px: u32,
    levels: Vec<Level>,
}

impl TileGrid {
    /// A grid with any levels, coarsest first. `origin` is the top-left
    /// corner of tile `(0, 0)` at every level; `tile_px` is a tile's edge in
    /// pixels, which with a level's span gives its resolution.
    pub fn new(origin: [f64; 2], tile_px: u32, levels: Vec<Level>) -> Result<TileGrid, Error> {
        if !origin.iter().all(|v| v.is_finite()) || tile_px == 0 {
            return Err(Error::BadGeometry);
        }
        if levels.is_empty() {
            return Err(Error::NoLevels);
        }
        if levels.len() > 256 {
            return Err(Error::TooManyLevels(levels.len()));
        }
        for (z, level) in levels.iter().enumerate() {
            let z = z as u8;
            if !(level.span.is_finite() && level.span > 0.0) || level.cols == 0 || level.rows == 0 {
                return Err(Error::BadLevel(z));
            }
            if z > 0 && level.span >= levels[z as usize - 1].span {
                return Err(Error::NotFiner(z));
            }
        }
        Ok(TileGrid { origin, tile_px, levels })
    }

    /// A grid whose tiles halve at each level, from `cols0` x `rows0` tiles of
    /// `span0` map units at level 0 down to `max_zoom`.
    pub fn quadtree(
        origin: [f64; 2],
        span0: f64,
        cols0: u32,
        rows0: u32,
        tile_px: u32,
        max_zoom: u8,
    ) -> Result<TileGrid, Error> {
        let mut levels = Vec::with_capacity(max_zoom as usize + 1);
        for z in 0..=max_zoom {
            let scale = 1u32.checked_shl(z as u32).ok_or(Error::TooDeep(z))?;
            let cols = cols0.checked_mul(scale).ok_or(Error::TooDeep(z))?;
            let rows = rows0.checked_mul(scale).ok_or(Error::TooDeep(z))?;
            levels.push(Level { span: span0 / scale as f64, cols, rows });
        }
        TileGrid::new(origin, tile_px, levels)
    }

    /// The XYZ Web Mercator grid used by OpenStreetMap-style tiles: one tile
    /// at level 0 covering the EPSG:3857 square, halving to `max_zoom`.
    /// Coordinates are EPSG:3857 metres; a caller holding longitude/latitude
    /// projects them first.
    pub fn web_mercator(tile_px: u32, max_zoom: u8) -> Result<TileGrid, Error> {
        let h = WEB_MERCATOR_HALF_EXTENT;
        TileGrid::quadtree([-h, h], 2.0 * h, 1, 1, tile_px, max_zoom)
    }

    /// The finest level.
    pub fn max_zoom(&self) -> u8 {
        (self.levels.len() - 1) as u8
    }

    pub fn tile_px(&self) -> u32 {
        self.tile_px
    }

    pub fn level(&self, z: u8) -> Option<&Level> {
        self.levels.get(z as usize)
    }

    /// Map units per pixel at level `z`.
    pub fn resolution(&self, z: u8) -> Option<f64> {
        self.level(z).map(|l| l.span / self.tile_px as f64)
    }

    /// The level to draw at `units_per_px`: the coarsest whose resolution is
    /// at least that fine, so no tile is shown magnified. Finer than the
    /// finest level gives the finest (it will be magnified, which is
    /// unavoidable); coarser than level 0 gives 0.
    pub fn level_for_resolution(&self, units_per_px: f64) -> u8 {
        // Tolerate the rounding in a resolution the caller computed from
        // this grid's own numbers.
        let target = units_per_px * (1.0 + 1e-9);
        (0..=self.max_zoom()).find(|&z| self.resolution(z).unwrap() <= target).unwrap_or(self.max_zoom())
    }

    /// A tile's extent in map units, or `None` if no such tile exists.
    pub fn bounds(&self, id: TileId) -> Option<Rect> {
        let level = self.level(id.z)?;
        if id.x >= level.cols || id.y >= level.rows {
            return None;
        }
        let min_x = self.origin[0] + id.x as f64 * level.span;
        let max_y = self.origin[1] - id.y as f64 * level.span;
        Some(Rect { min: [min_x, max_y - level.span], max: [min_x + level.span, max_y] })
    }

    /// The tiles at level `z` that `rect` touches, clipped to the grid. A
    /// rectangle edge lying exactly on a tile boundary does not pull in the
    /// tile beyond it, but a rectangle of zero width or height still touches
    /// the tile it lies in. Empty for an unknown level, a rectangle outside
    /// the grid, or one with a NaN.
    pub fn tiles_in(&self, z: u8, rect: Rect) -> TileRange {
        let empty = TileRange { z, x0: 1, x1: 0, y0: 1, y1: 0, x: 1, y: 1 };
        let Some(level) = self.level(z) else { return empty };
        let span = level.span;
        // Column and row positions, in tiles, of the rectangle's edges.
        let left = (rect.min[0] - self.origin[0]) / span;
        let right = (rect.max[0] - self.origin[0]) / span;
        let top = (self.origin[1] - rect.max[1]) / span;
        let bottom = (self.origin[1] - rect.min[1]) / span;
        let Some((x0, x1)) = covered(left, right, level.cols) else { return empty };
        let Some((y0, y1)) = covered(top, bottom, level.rows) else { return empty };
        TileRange { z, x0, x1, y0, y1, x: x0, y: y0 }
    }

    /// The frame that maps between map units and tile `id`'s local
    /// coordinates, or `None` if no such tile exists.
    pub fn frame(&self, id: TileId) -> Option<TileFrame> {
        let b = self.bounds(id)?;
        Some(TileFrame {
            center: [(b.min[0] + b.max[0]) / 2.0, (b.min[1] + b.max[1]) / 2.0],
            half_span: (b.max[0] - b.min[0]) / 2.0,
        })
    }
}

/// The first and last tile index an interval `[lo, hi]` (in tiles) touches,
/// clipped to `0..n`, or `None` if it touches none.
fn covered(lo: f64, hi: f64, n: u32) -> Option<(u32, u32)> {
    // A tile edge computed as `origin + x * span` does not always divide back
    // to exactly `x`, so a rectangle built from a tile's own bounds would
    // spill into its neighbours. An edge within a billionth of a tile of a
    // boundary counts as on it.
    let snap = |v: f64| if (v - v.round()).abs() < 1e-9 { v.round() } else { v };
    let (lo, hi) = (snap(lo), snap(hi));
    let n = n as f64;
    if lo.is_nan() || hi.is_nan() || lo > hi {
        return None;
    }
    if lo == hi {
        // No width: the point touches the tile it starts, or the last tile
        // when it sits on the grid's far edge.
        if lo < 0.0 || lo > n {
            return None;
        }
        let i = lo.floor().min(n - 1.0) as u32;
        return Some((i, i));
    }
    // Half-open: an edge exactly on a boundary stays out of the tile beyond.
    if hi <= 0.0 || lo >= n {
        return None;
    }
    let first = lo.floor().max(0.0);
    let last = (hi.ceil() - 1.0).min(n - 1.0);
    Some((first as u32, last as u32))
}

/// The tiles of one level inside a rectangle, row by row. Built by
/// [`TileGrid::tiles_in`]; holds no allocation.
#[derive(Debug, Clone)]
pub struct TileRange {
    z: u8,
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
    // The next tile to yield.
    x: u32,
    y: u32,
}

impl TileRange {
    /// How many tiles the range holds in all, yielded or not.
    pub fn count_all(&self) -> u64 {
        if self.x0 > self.x1 || self.y0 > self.y1 {
            return 0;
        }
        (self.x1 - self.x0 + 1) as u64 * (self.y1 - self.y0 + 1) as u64
    }
}

impl Iterator for TileRange {
    type Item = TileId;

    fn next(&mut self) -> Option<TileId> {
        if self.y > self.y1 || self.x0 > self.x1 {
            return None;
        }
        let id = TileId { z: self.z, x: self.x, y: self.y };
        if self.x == self.x1 {
            self.x = self.x0;
            self.y += 1;
        } else {
            self.x += 1;
        }
        Some(id)
    }
}

/// A tile's local coordinate frame: map coordinates relative to the tile's
/// centre, scaled so the tile's edges sit at -1 and +1 (Y still up).
///
/// Prepared geometry is stored in these coordinates as `f32`. Map coordinates
/// themselves cannot be: at Web Mercator's scale an `f32` steps in metres,
/// which jitters at street level. Relative to a tile's centre, the same 24-bit
/// mantissa spans one tile, so precision follows the zoom level instead
/// (`../plans/rendering.org` § TILE GRIDS). Coordinates outside the tile, such
/// as a clip guard band, simply fall outside -1..1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileFrame {
    pub center: [f64; 2],
    pub half_span: f64,
}

impl TileFrame {
    /// Map units to tile-local. The subtraction happens in `f64`, before the
    /// cast, which is what keeps the precision.
    pub fn to_local(&self, xy: [f64; 2]) -> [f32; 2] {
        [
            ((xy[0] - self.center[0]) / self.half_span) as f32,
            ((xy[1] - self.center[1]) / self.half_span) as f32,
        ]
    }

    /// Map units to tile-local, kept in `f64` for work done before storing.
    pub fn to_local_f64(&self, xy: [f64; 2]) -> [f64; 2] {
        [(xy[0] - self.center[0]) / self.half_span, (xy[1] - self.center[1]) / self.half_span]
    }

    /// Tile-local back to map units.
    pub fn to_map(&self, local: [f32; 2]) -> [f64; 2] {
        [
            self.center[0] + local[0] as f64 * self.half_span,
            self.center[1] + local[1] as f64 * self.half_span,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: f64 = WEB_MERCATOR_HALF_EXTENT;

    fn rect(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Rect {
        Rect { min: [min_x, min_y], max: [max_x, max_y] }
    }

    fn ids(range: TileRange) -> Vec<(u32, u32)> {
        range.map(|t| (t.x, t.y)).collect()
    }

    #[test]
    fn web_mercator_extent_is_pi_times_the_wgs84_radius() {
        assert!((H - std::f64::consts::PI * 6_378_137.0).abs() < 1e-6);
    }

    #[test]
    fn web_mercator_bounds_match_xyz() {
        let g = TileGrid::web_mercator(256, 22).unwrap();
        assert_eq!(g.bounds(TileId { z: 0, x: 0, y: 0 }), Some(rect(-H, -H, H, H)));
        // Level 1: (0, 0) is the north-west quarter, (1, 1) the south-east.
        assert_eq!(g.bounds(TileId { z: 1, x: 0, y: 0 }), Some(rect(-H, 0.0, 0.0, H)));
        assert_eq!(g.bounds(TileId { z: 1, x: 1, y: 1 }), Some(rect(0.0, -H, H, 0.0)));
        assert_eq!(g.bounds(TileId { z: 1, x: 2, y: 0 }), None);
        assert_eq!(g.bounds(TileId { z: 23, x: 0, y: 0 }), None);
        assert_eq!(g.max_zoom(), 22);
    }

    #[test]
    fn covering_a_rectangle() {
        let g = TileGrid::web_mercator(256, 22).unwrap();
        // The whole world at level 2 is all 16 tiles, row by row.
        let all = g.tiles_in(2, rect(-H, -H, H, H));
        assert_eq!(all.count_all(), 16);
        let all = ids(all);
        assert_eq!(all.len(), 16);
        assert_eq!(all[0], (0, 0));
        assert_eq!(all[1], (1, 0));
        assert_eq!(all[15], (3, 3));
        // A rectangle well inside one tile.
        assert_eq!(ids(g.tiles_in(1, rect(1000.0, 1000.0, 2000.0, 2000.0))), vec![(1, 0)]);
        // Larger than the grid: clipped to it.
        assert_eq!(g.tiles_in(1, rect(-10.0 * H, -10.0 * H, 10.0 * H, 10.0 * H)).count_all(), 4);
        // Outside the grid, NaN, inverted, or an unknown level: nothing.
        assert_eq!(g.tiles_in(1, rect(2.0 * H, 0.0, 3.0 * H, 1.0)).count(), 0);
        assert_eq!(g.tiles_in(1, rect(f64::NAN, 0.0, 1.0, 1.0)).count(), 0);
        assert_eq!(g.tiles_in(1, rect(1.0, 0.0, 0.0, 1.0)).count(), 0);
        assert_eq!(g.tiles_in(40, rect(-H, -H, H, H)).count(), 0);
    }

    #[test]
    fn boundaries_do_not_pull_in_the_next_tile() {
        let g = TileGrid::web_mercator(256, 22).unwrap();
        // Exactly the north-west quarter at level 1: one tile, not four.
        assert_eq!(ids(g.tiles_in(1, rect(-H, 0.0, 0.0, H))), vec![(0, 0)]);
        // Every tile's own bounds cover exactly that tile.
        for t in g.tiles_in(3, rect(-H, -H, H, H)) {
            assert_eq!(ids(g.tiles_in(3, g.bounds(t).unwrap())), vec![(t.x, t.y)], "{t:?}");
        }
        // A point on the centre still touches a tile.
        assert_eq!(ids(g.tiles_in(1, rect(0.0, 0.0, 0.0, 0.0))), vec![(1, 1)]);
        // So does a point on the grid's far edges.
        assert_eq!(ids(g.tiles_in(1, rect(H, -H, H, -H))), vec![(1, 1)]);
        // A rectangle outside the grid that only meets its edge touches
        // nothing, the same rule as between tiles.
        assert_eq!(g.tiles_in(1, rect(-2.0 * H, 0.0, -H, 1.0)).count(), 0);
        assert_eq!(g.tiles_in(1, rect(H, 0.0, 2.0 * H, 1.0)).count(), 0);
    }

    #[test]
    fn choosing_a_level_by_resolution() {
        let g = TileGrid::web_mercator(256, 22).unwrap();
        let r0 = g.resolution(0).unwrap();
        assert!((r0 - 156_543.033_928_041).abs() < 1e-6);
        assert_eq!(g.level_for_resolution(r0), 0);
        assert_eq!(g.level_for_resolution(g.resolution(5).unwrap()), 5);
        // Between levels 1 and 2: level 2, so nothing is magnified.
        assert_eq!(g.level_for_resolution(r0 / 3.0), 2);
        assert_eq!(g.level_for_resolution(r0 * 10.0), 0);
        assert_eq!(g.level_for_resolution(1e-6), 22);
    }

    #[test]
    fn a_grid_whose_levels_do_not_halve() {
        // An irregular pyramid, as some national grids are, with its origin
        // somewhere other than the CRS's.
        let levels = vec![
            Level { span: 1000.0, cols: 3, rows: 2 },
            Level { span: 400.0, cols: 8, rows: 5 },
            Level { span: 100.0, cols: 30, rows: 20 },
        ];
        let g = TileGrid::new([2_420_000.0, 1_350_000.0], 512, levels).unwrap();
        assert_eq!(
            g.bounds(TileId { z: 1, x: 2, y: 1 }),
            Some(rect(2_420_800.0, 1_349_200.0, 2_421_200.0, 1_349_600.0))
        );
        assert_eq!(g.level_for_resolution(1.0), 1);
        assert_eq!(ids(g.tiles_in(2, rect(2_420_150.0, 1_349_750.0, 2_420_250.0, 1_349_850.0))), vec![(1, 1), (2, 1), (1, 2), (2, 2)]);
    }

    #[test]
    fn grids_that_cannot_exist_are_errors() {
        let ok = Level { span: 1.0, cols: 1, rows: 1 };
        assert_eq!(TileGrid::new([0.0, 0.0], 256, vec![]), Err(Error::NoLevels));
        assert_eq!(TileGrid::new([0.0, 0.0], 0, vec![ok]), Err(Error::BadGeometry));
        assert_eq!(TileGrid::new([f64::NAN, 0.0], 256, vec![ok]), Err(Error::BadGeometry));
        assert_eq!(TileGrid::new([0.0, 0.0], 256, vec![Level { span: 0.0, ..ok }]), Err(Error::BadLevel(0)));
        assert_eq!(TileGrid::new([0.0, 0.0], 256, vec![Level { cols: 0, ..ok }]), Err(Error::BadLevel(0)));
        assert_eq!(TileGrid::new([0.0, 0.0], 256, vec![ok, ok]), Err(Error::NotFiner(1)));
        assert_eq!(TileGrid::new([0.0, 0.0], 256, vec![ok; 257]), Err(Error::TooManyLevels(257)));
        // 2^31 tiles across fits a u32; twice that from two level-0 columns
        // does not.
        assert!(TileGrid::quadtree([0.0, 0.0], 1.0, 1, 1, 256, 31).is_ok());
        assert_eq!(TileGrid::quadtree([0.0, 0.0], 1.0, 2, 1, 256, 31), Err(Error::TooDeep(31)));
        assert_eq!(TileGrid::quadtree([0.0, 0.0], 1.0, 1, 1, 256, 32), Err(Error::TooDeep(32)));
    }

    #[test]
    fn tile_local_coordinates_keep_street_level_precision() {
        let g = TileGrid::web_mercator(256, 22).unwrap();
        // A level-20 tile near the far east edge, where map X is about 2e7 m
        // and a raw f32 steps in units of 2 m.
        let t = TileId { z: 20, x: (1 << 20) - 1, y: 1 << 19 };
        let frame = g.frame(t).unwrap();
        let b = g.bounds(t).unwrap();
        assert_eq!(frame.to_local(b.min), [-1.0, -1.0]);
        assert_eq!(frame.to_local(b.max), [1.0, 1.0]);
        let p = [b.min[0] + 12.345_678, b.min[1] + 7.654_321];
        let back = frame.to_map(frame.to_local(p));
        let err = (back[0] - p[0]).abs().max((back[1] - p[1]).abs());
        assert!(err < 1e-5, "tile-local round trip off by {err} m");
        let raw = (p[0] as f32 as f64 - p[0]).abs();
        assert!(raw > 0.1, "a raw f32 would be off by {raw} m");
    }
}
