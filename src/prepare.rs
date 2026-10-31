//! Turning geometry into draw-ready buffers for one tile.
//!
//! A [`MeshBuilder`] is a [`Sink`]: feed it a tile's features and it
//! produces a [`Mesh`] in that tile's local frame. On the way through, each
//! coordinate is:
//!
//! 1. *snapped* to a sub-pixel grid ([`Detail::grid_px`], a quarter pixel by
//!    default). Every tile at a level uses the same grid, and it is fixed in
//!    map units, so two polygons sharing a border snap it to the same points
//!    whichever way each one runs. That is why this snaps rather than
//!    "dropping points that share a pixel" (`../plans/rendering.org` §
//!    PREPARE): dropping depends on direction and would open slivers between
//!    neighbours. Point features are the exception and keep their exact
//!    position (`MeshBuilder::point`).
//! 2. *decimated*: a point snapped onto its predecessor, or exactly in line
//!    with the two before it, adds nothing at this resolution and is dropped.
//!    On an integer grid both tests are exact, so neighbours still agree.
//! 3. *clipped* to the tile plus a guard band ([`Detail::guard_px`]): rings
//!    by Sutherland--Hodgman, lines segment by segment. A ring wholly inside
//!    skips clipping; one wholly outside is dropped without it. Edges that
//!    clipping creates along the clip boundary are filled but not outlined.
//! 4. *triangulated* by [`Earcut`], for polygons.
//!
//! A ring's vertices are stored once: fill triangles and outline segments are
//! both indices into the same vertex buffer, and each vertex carries the
//! feature's style class.
//!
//! Culling whole features before decoding them needs a bounding box the
//! source already stores (a GeoPackage envelope, a GeoParquet covering
//! column, a FlatGeobuf index): the caller skips those features rather than
//! hand them over. WKB carries no bbox of its own.

use crate::earcut::Earcut;
use crate::grid::{TileFrame, TileGrid, TileId};
use crate::sink::Sink;
use std::ops::Range;

/// How finely a tile is prepared.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detail {
    /// The tile's edge in output pixels: 256 or 512 for screen tiles, more
    /// for print.
    pub tile_px: f64,
    /// The snapping grid's step, in pixels.
    pub grid_px: f64,
    /// How far past the tile's edge geometry is kept, in pixels, so wide
    /// lines and their joins continue cleanly into the next tile.
    pub guard_px: f64,
}

impl Default for Detail {
    fn default() -> Detail {
        Detail { tile_px: 512.0, grid_px: 0.25, guard_px: 32.0 }
    }
}

/// Draw-ready geometry in one tile's local coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    /// The frame the vertices are relative to.
    pub frame: TileFrame,
    /// Three `f32` per vertex: tile-local X and Y, then the style class.
    pub vertices: Vec<f32>,
    /// Fill triangles, three vertex indices each, counter-clockwise.
    pub fills: Vec<u32>,
    /// Outline segments, two vertex indices each, for polygon rings and lines.
    pub outlines: Vec<u32>,
    /// Point features, one vertex index each.
    pub points: Vec<u32>,
    /// Each polygon ring's vertices, as a `[start, end)` range: a ring's
    /// vertices are stored consecutively. What a backend that fills paths
    /// rather than triangles (SVG) draws.
    pub rings: Vec<[u32; 2]>,
    /// Each polygon's rings, as a `[start, end)` range into `rings`; the
    /// first is the exterior.
    pub polygons: Vec<[u32; 2]>,
    /// One per vertex: the feature it belongs to, as
    /// [`MeshBuilder::set_feature`] set it. What picking returns.
    pub ids: Vec<u32>,
}

impl Mesh {
    pub fn vertex_count(&self) -> usize {
        self.vertices.len() / 3
    }

    /// The bytes its buffers hold, which is what uploading it costs: the
    /// size to give [`crate::TileCache::insert`].
    pub fn byte_size(&self) -> usize {
        4 * (self.vertices.len() + self.fills.len() + self.outlines.len() + self.points.len() + self.ids.len())
            + 8 * (self.rings.len() + self.polygons.len())
    }
}

/// Builds a [`Mesh`] from geometry fed to it as a [`Sink`]. See the module
/// docs.
///
/// Set the style class with [`MeshBuilder::set_class`] and the feature id
/// with [`MeshBuilder::set_feature`] before each feature.
/// The builder keeps its working buffers between features, so one builder
/// per tile allocates little beyond the mesh itself.
pub struct MeshBuilder {
    mesh: Mesh,
    class: f32,
    feature: u32,
    /// Grid steps from the tile's centre to its edge. Tile-local coordinates
    /// run -1..1, so a local coordinate times this is its grid coordinate.
    half_cells: f64,
    /// The clip square's half-width, in grid steps.
    clip: f64,
    // The polygon being received, in grid coordinates.
    pts: Vec<[f64; 2]>,
    rings: Vec<Range<usize>>,
    ring_start: usize,
    rings_seen: usize,
    // The line being received, if any, in grid coordinates.
    line: Option<Vec<[f64; 2]>>,
    scratch: [Vec<[f64; 2]>; 2],
    earcut: Earcut,
    tris: Vec<u32>,
}

impl MeshBuilder {
    /// A builder for geometry in `frame`, which spans `detail.tile_px`
    /// pixels edge to edge.
    pub fn new(frame: TileFrame, detail: Detail) -> MeshBuilder {
        // An even number of grid steps across the tile puts the tile's edges
        // on grid lines, so neighbouring tiles' grids line up.
        let cells = ((detail.tile_px / detail.grid_px / 2.0).round() * 2.0).max(2.0);
        let half_cells = cells / 2.0;
        let clip = half_cells + (detail.guard_px / detail.grid_px).round().max(0.0);
        MeshBuilder {
            mesh: Mesh {
                frame,
                vertices: Vec::new(),
                fills: Vec::new(),
                outlines: Vec::new(),
                points: Vec::new(),
                rings: Vec::new(),
                polygons: Vec::new(),
                ids: Vec::new(),
            },
            class: 0.0,
            feature: 0,
            half_cells,
            clip,
            pts: Vec::new(),
            rings: Vec::new(),
            ring_start: 0,
            rings_seen: 0,
            line: None,
            scratch: [Vec::new(), Vec::new()],
            earcut: Earcut::new(),
            tris: Vec::new(),
        }
    }

    /// A builder for everything `camera` shows, at its own resolution: one
    /// frame around the view, as a layout prepares its single view
    /// (`../plans/rendering.org` § THE PIPELINE). The camera's pixels set the
    /// detail, so a print-resolution camera keeps more vertices than a
    /// screen one of the same map.
    pub fn for_view(camera: &crate::camera::Camera) -> MeshBuilder {
        let b = camera.visible_bounds();
        let half_span = (b.max[0] - b.min[0]).max(b.max[1] - b.min[1]) / 2.0;
        let frame = TileFrame { center: b.center(), half_span: half_span.max(f64::MIN_POSITIVE) };
        let detail = Detail { tile_px: (2.0 * half_span / camera.units_per_px).max(1.0), ..Detail::default() };
        MeshBuilder::new(frame, detail)
    }

    /// A builder for tile `id` of `grid`, at the grid's tile size and the
    /// default grid step and guard band. `None` if there is no such tile.
    pub fn for_tile(grid: &TileGrid, id: TileId) -> Option<MeshBuilder> {
        let detail = Detail { tile_px: grid.tile_px() as f64, ..Detail::default() };
        Some(MeshBuilder::new(grid.frame(id)?, detail))
    }

    /// The style class for the vertices that follow: an index into the
    /// backend's palette.
    pub fn set_class(&mut self, class: u8) {
        self.class = class as f32;
    }

    /// The feature id for the vertices that follow: the embedder's own
    /// number for the feature, which picking hands back. Nazca keeps no
    /// attributes (`../project.org` § WHAT NAZCA DOES NOT DO).
    pub fn set_feature(&mut self, id: u32) {
        self.feature = id;
    }

    pub fn finish(self) -> Mesh {
        self.mesh
    }

    /// Map units to grid coordinates.
    fn snap(&self, xy: [f64; 2]) -> [f64; 2] {
        let [x, y] = self.mesh.frame.to_local_f64(xy);
        [(x * self.half_cells).round(), (y * self.half_cells).round()]
    }

    fn inside(&self, p: [f64; 2]) -> bool {
        p[0].abs() <= self.clip && p[1].abs() <= self.clip
    }

    fn push_vertex(&mut self, g: [f64; 2]) -> u32 {
        let index = self.mesh.vertex_count() as u32;
        let s = self.half_cells;
        self.mesh.vertices.extend_from_slice(&[(g[0] / s) as f32, (g[1] / s) as f32, self.class]);
        self.mesh.ids.push(self.feature);
        index
    }

    /// Whether the edge `a`--`b` lies along the clip boundary, where only
    /// clipping puts edges.
    fn on_boundary(&self, a: [f64; 2], b: [f64; 2]) -> bool {
        let c = self.clip;
        (a[0] == b[0] && a[0].abs() == c) || (a[1] == b[1] && a[1].abs() == c)
    }

    /// Clips the ring `pts[start..]` to the clip square in place.
    fn clip_ring(&mut self, start: usize) {
        let [mut a, mut b] = std::mem::take(&mut self.scratch);
        a.clear();
        a.extend_from_slice(&self.pts[start..]);
        let c = self.clip;
        for (axis, bound, keep_greater) in [(0, -c, true), (0, c, false), (1, -c, true), (1, c, false)] {
            clip_edge(&a, &mut b, axis, bound, keep_greater);
            std::mem::swap(&mut a, &mut b);
            if a.is_empty() {
                break;
            }
        }
        self.pts.truncate(start);
        for &p in &a {
            push_decimated(&mut self.pts, start, p);
        }
        self.scratch = [a, b];
    }

    fn end_line_run(&mut self, line: &[[f64; 2]]) {
        // The index of the last vertex, while the line runs on inside.
        let mut prev: Option<u32> = None;
        for w in line.windows(2) {
            let Some((p, q)) = clip_segment(w[0], w[1], self.clip) else {
                prev = None;
                continue;
            };
            let from = match prev {
                Some(i) if p == w[0] => i,
                _ => self.push_vertex(p),
            };
            let to = self.push_vertex(q);
            self.mesh.outlines.extend_from_slice(&[from, to]);
            prev = (q == w[1]).then_some(to);
        }
    }
}

/// Appends `p` to the ring or line `buf[start..]`, unless it repeats the
/// last point; and replaces the last point instead if it lies exactly on the
/// way from the one before to `p`.
fn push_decimated(buf: &mut Vec<[f64; 2]>, start: usize, p: [f64; 2]) {
    let n = buf.len() - start;
    if n >= 1 && buf[buf.len() - 1] == p {
        return;
    }
    if n >= 2 && passes_through(buf[buf.len() - 2], buf[buf.len() - 1], p) {
        *buf.last_mut().unwrap() = p;
        return;
    }
    buf.push(p);
}

/// Whether `b` lies on the segment from `a` to `c`, strictly between, so
/// dropping it changes nothing. Exact on grid coordinates.
fn passes_through(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    let (ux, uy) = (b[0] - a[0], b[1] - a[1]);
    let (vx, vy) = (c[0] - b[0], c[1] - b[1]);
    ux * vy - uy * vx == 0.0 && ux * vx + uy * vy > 0.0
}

/// One Sutherland--Hodgman pass: keeps the part of ring `input` on one side
/// of the line `p[axis] == bound`. New points sit exactly on the line, their
/// other coordinate snapped to the grid.
fn clip_edge(input: &[[f64; 2]], out: &mut Vec<[f64; 2]>, axis: usize, bound: f64, keep_greater: bool) {
    out.clear();
    let inside = |p: [f64; 2]| if keep_greater { p[axis] >= bound } else { p[axis] <= bound };
    let cross = |a: [f64; 2], b: [f64; 2]| {
        let t = (bound - a[axis]) / (b[axis] - a[axis]);
        let other = 1 - axis;
        let mut p = [0.0; 2];
        p[axis] = bound;
        p[other] = (a[other] + t * (b[other] - a[other])).round();
        p
    };
    for i in 0..input.len() {
        let cur = input[i];
        let prev = input[(i + input.len() - 1) % input.len()];
        match (inside(prev), inside(cur)) {
            (true, true) => out.push(cur),
            (true, false) => out.push(cross(prev, cur)),
            (false, true) => {
                out.push(cross(prev, cur));
                out.push(cur);
            }
            (false, false) => {}
        }
    }
}

/// The part of segment `a`--`b` inside the square `|x|, |y| <= c`
/// (Liang--Barsky), or `None`. Endpoints that move sit on the square,
/// snapped to the grid.
fn clip_segment(a: [f64; 2], b: [f64; 2], c: f64) -> Option<([f64; 2], [f64; 2])> {
    let d = [b[0] - a[0], b[1] - a[1]];
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for axis in 0..2 {
        for (p, q) in [(-d[axis], a[axis] + c), (d[axis], c - a[axis])] {
            if p == 0.0 {
                if q < 0.0 {
                    return None;
                }
            } else {
                let r = q / p;
                if p < 0.0 {
                    t0 = t0.max(r);
                } else {
                    t1 = t1.min(r);
                }
            }
        }
    }
    if t0 > t1 {
        return None;
    }
    let at = |t: f64| [(a[0] + t * d[0]).round().clamp(-c, c), (a[1] + t * d[1]).round().clamp(-c, c)];
    let p = if t0 == 0.0 { a } else { at(t0) };
    let q = if t1 == 1.0 { b } else { at(t1) };
    (p != q).then_some((p, q))
}

impl Sink for MeshBuilder {
    /// A point feature is kept where it is, not snapped: snapping serves
    /// shared borders and decimation, neither of which a lone point has, and
    /// a tile drawn as a stand-in several levels up would magnify the
    /// rounding into a visible jump when the real tile replaces it.
    fn point(&mut self, xy: [f64; 2]) {
        let [x, y] = self.mesh.frame.to_local_f64(xy);
        let g = [x * self.half_cells, y * self.half_cells];
        if self.inside(g) {
            let i = self.push_vertex(g);
            self.mesh.points.push(i);
        }
    }

    fn points(&mut self, xy: &[[f64; 2]]) {
        if let Some(mut line) = self.line.take() {
            for &p in xy {
                let g = self.snap(p);
                push_decimated(&mut line, 0, g);
            }
            self.line = Some(line);
        } else {
            for &p in xy {
                let g = self.snap(p);
                push_decimated(&mut self.pts, self.ring_start, g);
            }
        }
    }

    fn begin_line(&mut self) {
        let mut line = self.scratch[0].split_off(0);
        line.clear();
        self.line = Some(line);
    }

    fn end_line(&mut self) {
        if let Some(line) = self.line.take() {
            self.end_line_run(&line);
            self.scratch[0] = line;
        }
    }

    fn begin_polygon(&mut self) {
        self.pts.clear();
        self.rings.clear();
        self.rings_seen = 0;
    }

    fn begin_ring(&mut self) {
        self.ring_start = self.pts.len();
    }

    fn end_ring(&mut self) {
        let start = self.ring_start;
        let exterior = self.rings_seen == 0;
        self.rings_seen += 1;
        // A polygon whose exterior is gone is gone; its holes must not stand
        // in for it.
        if !exterior && self.rings.is_empty() {
            self.pts.truncate(start);
            return;
        }

        // The ring closes on itself: drop the repeated first point, then
        // anything left redundant across the join.
        if self.pts.len() - start >= 2 && self.pts.last() == self.pts.get(start) {
            self.pts.pop();
        }
        while self.pts.len() - start >= 3 {
            let n = self.pts.len();
            if passes_through(self.pts[n - 2], self.pts[n - 1], self.pts[start]) {
                self.pts.pop();
            } else if passes_through(self.pts[n - 1], self.pts[start], self.pts[start + 1]) {
                self.pts.remove(start);
            } else {
                break;
            }
        }

        if self.pts.len() - start >= 3 {
            let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
            for p in &self.pts[start..] {
                lo = [lo[0].min(p[0]), lo[1].min(p[1])];
                hi = [hi[0].max(p[0]), hi[1].max(p[1])];
            }
            let c = self.clip;
            if lo[0] > c || lo[1] > c || hi[0] < -c || hi[1] < -c {
                self.pts.truncate(start); // wholly outside
            } else if lo[0] < -c || lo[1] < -c || hi[0] > c || hi[1] > c {
                self.clip_ring(start);
            }
        }
        if self.pts.len() - start < 3 {
            self.pts.truncate(start);
            return;
        }
        self.rings.push(start..self.pts.len());
    }

    fn end_polygon(&mut self) {
        if self.rings.is_empty() {
            return;
        }
        let base = self.mesh.vertex_count() as u32;
        let first_ring = self.mesh.rings.len() as u32;
        for r in &self.rings {
            self.mesh.rings.push([base + r.start as u32, base + r.end as u32]);
        }
        self.mesh.polygons.push([first_ring, self.mesh.rings.len() as u32]);
        self.tris.clear();
        self.earcut.triangulate(&self.pts, &self.rings, &mut self.tris);
        self.mesh.fills.extend(self.tris.iter().map(|&i| base + i));
        for i in 0..self.pts.len() {
            self.push_vertex(self.pts[i]);
        }
        for r in 0..self.rings.len() {
            let ring = self.rings[r].clone();
            for i in ring.clone() {
                let next = if i + 1 == ring.end { ring.start } else { i + 1 };
                if !self.on_boundary(self.pts[i], self.pts[next]) {
                    self.mesh.outlines.extend_from_slice(&[base + i as u32, base + next as u32]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame whose local coordinates equal map coordinates, and a builder
    /// over it with 1-unit grid steps: 2 * 100 steps across, a guard of 10.
    fn builder() -> MeshBuilder {
        let frame = TileFrame { center: [0.0, 0.0], half_span: 100.0 };
        MeshBuilder::new(frame, Detail { tile_px: 200.0, grid_px: 1.0, guard_px: 10.0 })
    }

    fn polygon(b: &mut MeshBuilder, rings: &[&[[f64; 2]]]) {
        b.begin_polygon();
        for r in rings {
            b.begin_ring();
            b.points(r);
            b.points(&r[..1]); // closed, as WKB writes it
            b.end_ring();
        }
        b.end_polygon();
    }

    /// The vertex positions, in grid steps (also map units, with this
    /// frame). Stored as `f32` tile-local, so rounded back: every true
    /// position is a whole grid step.
    fn positions(m: &Mesh) -> Vec<[f64; 2]> {
        m.vertices.chunks(3).map(|v| [(v[0] as f64 * 100.0).round(), (v[1] as f64 * 100.0).round()]).collect()
    }

    fn fill_area(m: &Mesh) -> f64 {
        let p = positions(m);
        m.fills.chunks(3).map(|t| {
            let (a, b, c) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
            ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0
        }).sum()
    }

    #[test]
    fn a_square_inside_the_tile() {
        let mut b = builder();
        polygon(&mut b, &[&[[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]]);
        let m = b.finish();
        assert_eq!(m.vertex_count(), 4);
        assert_eq!(fill_area(&m), 400.0);
        assert_eq!(m.outlines.len(), 8);
    }

    #[test]
    fn polygons_record_their_rings() {
        let mut b = builder();
        let square = |c: f64, r: f64| [[c - r, c - r], [c + r, c - r], [c + r, c + r], [c - r, c + r]];
        polygon(&mut b, &[&square(0.0, 20.0), &square(0.0, 5.0)]);
        polygon(&mut b, &[&square(50.0, 5.0)]);
        let m = b.finish();
        assert_eq!(m.polygons, [[0, 2], [2, 3]]);
        assert_eq!(m.rings, [[0, 4], [4, 8], [8, 12]]);
        assert_eq!(m.vertex_count(), 12);
    }

    #[test]
    fn every_vertex_carries_its_feature() {
        let mut b = builder();
        b.set_feature(7);
        polygon(&mut b, &[&[[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]]);
        b.set_feature(u32::MAX);
        b.point([50.0, 50.0]);
        b.begin_line();
        b.points(&[[0.0, 20.0], [30.0, 20.0]]);
        b.end_line();
        let m = b.finish();
        assert_eq!(m.ids, [7, 7, 7, 7, u32::MAX, u32::MAX, u32::MAX]);
        assert_eq!(m.ids.len(), m.vertex_count());
    }

    #[test]
    fn a_polygon_over_the_whole_tile_fills_the_clip_square_without_outlines() {
        let mut b = builder();
        polygon(&mut b, &[&[[-500.0, -500.0], [500.0, -500.0], [500.0, 500.0], [-500.0, 500.0]]]);
        let m = b.finish();
        assert_eq!(fill_area(&m), 220.0 * 220.0); // the tile and its guard band
        assert!(m.outlines.is_empty(), "clipping's own edges are not outlined");
    }

    #[test]
    fn clipping_keeps_the_real_edges() {
        // Overlaps the clip square's right edge (x = 110).
        let mut b = builder();
        polygon(&mut b, &[&[[50.0, -20.0], [200.0, -20.0], [200.0, 20.0], [50.0, 20.0]]]);
        let m = b.finish();
        assert_eq!(fill_area(&m), 60.0 * 40.0);
        // Three real edges are outlined; the one at x = 110 is clipping's.
        assert_eq!(m.outlines.len() / 2, 3);
    }

    #[test]
    fn a_hole_and_clipping() {
        let mut b = builder();
        let outer: &[[f64; 2]] = &[[-300.0, -300.0], [300.0, -300.0], [300.0, 300.0], [-300.0, 300.0]];
        let hole: &[[f64; 2]] = &[[-5.0, -5.0], [-5.0, 5.0], [5.0, 5.0], [5.0, -5.0]];
        polygon(&mut b, &[outer, hole]);
        let m = b.finish();
        assert_eq!(fill_area(&m), 220.0 * 220.0 - 100.0);
        assert_eq!(m.outlines.len() / 2, 4, "only the hole is outlined");
    }

    #[test]
    fn rings_outside_the_tile() {
        let mut b = builder();
        let far: &[[f64; 2]] = &[[500.0, 500.0], [600.0, 500.0], [600.0, 600.0]];
        // An exterior outside: the polygon is gone, holes and all.
        polygon(&mut b, &[far, &[[-5.0, -5.0], [5.0, -5.0], [0.0, 5.0]]]);
        // A hole outside its exterior: ignored.
        polygon(&mut b, &[&[[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]], far]);
        let m = b.finish();
        assert_eq!(m.vertex_count(), 4);
        assert_eq!(fill_area(&m), 400.0);
    }

    #[test]
    fn decimation_keeps_shape_and_drops_redundant_points() {
        // 10,000 points on a circle 20 grid steps across: most snap onto
        // their neighbours or in line with them.
        let ring: Vec<[f64; 2]> = (0..10_000).map(|k| {
            let t = 2.0 * std::f64::consts::PI * k as f64 / 10_000.0;
            [10.0 * t.cos(), 10.0 * t.sin()]
        }).collect();
        let mut b = builder();
        polygon(&mut b, &[&ring]);
        let m = b.finish();
        assert!(m.vertex_count() < 100, "{} vertices", m.vertex_count());
        let area = fill_area(&m);
        assert!((area - std::f64::consts::PI * 100.0).abs() < 25.0, "area {area}");
        // A straight run of points collapses to its ends.
        let mut b = builder();
        let straight: Vec<[f64; 2]> = (0..=20).map(|k| [k as f64, 0.0]).chain([[20.0, 20.0]]).collect();
        polygon(&mut b, &[&straight]);
        assert_eq!(b.finish().vertex_count(), 3);
    }

    #[test]
    fn a_shared_border_snaps_the_same_both_ways() {
        // Two polygons either side of a dense, jagged border, each running
        // it in its own direction, at an offset that is not on the grid.
        let border: Vec<[f64; 2]> = (0..=400).map(|k| {
            let y = -40.0 + k as f64 * 0.2 + 0.013;
            [(k as f64 * 0.7).sin() * 3.0 + 0.37, y]
        }).collect();
        let mut left: Vec<[f64; 2]> = vec![[-60.0, 40.2], [-60.0, -40.0]];
        left.extend(border.iter().copied());
        let mut right: Vec<[f64; 2]> = vec![[60.0, -40.0], [60.0, 40.2]];
        right.extend(border.iter().rev().copied());

        let mut a = builder();
        polygon(&mut a, &[&left]);
        let mut b = builder();
        polygon(&mut b, &[&right]);
        let (ma, mb) = (a.finish(), b.finish());
        // The border proper: where it meets the box's top and bottom, each
        // side may legitimately keep or drop a point lying along that edge.
        let on_border = |m: &Mesh| -> Vec<[i64; 2]> {
            let mut v: Vec<[i64; 2]> = positions(m)
                .iter()
                .filter(|p| p[0].abs() < 50.0 && p[1].abs() < 39.5)
                .map(|p| [p[0] as i64, p[1] as i64])
                .collect();
            v.sort();
            v
        };
        assert_eq!(on_border(&ma), on_border(&mb));
        // And between them they cover the box exactly: no gap, no overlap.
        let total = fill_area(&ma) + fill_area(&mb);
        let ends = |p: [f64; 2]| p[1].round();
        assert_eq!(total, 120.0 * (ends(border[400]) - ends(border[0])));
    }

    #[test]
    fn lines_are_clipped_into_runs() {
        let mut b = builder();
        // In, out past the right edge, back in.
        b.begin_line();
        b.points(&[[0.0, 0.0], [50.0, 0.0], [300.0, 10.0], [50.0, 20.0], [0.0, 20.0]]);
        b.end_line();
        let m = b.finish();
        let p = positions(&m);
        assert_eq!(m.outlines.len() / 2, 4);
        assert!(p.iter().all(|q| q[0].abs() <= 110.0 && q[1].abs() <= 110.0));
        // Two runs of three vertices: in, on the edge; on the edge, back.
        assert_eq!(m.vertex_count(), 6);
        // Wholly outside: nothing.
        let mut b = builder();
        b.begin_line();
        b.points(&[[200.0, 200.0], [300.0, 300.0]]);
        b.end_line();
        assert!(b.finish().outlines.is_empty());
    }

    #[test]
    fn points_outside_are_culled() {
        let mut b = builder();
        b.point([5.0, 5.0]);
        b.point([105.0, -105.0]); // in the guard band
        b.point([200.0, 0.0]);
        let m = b.finish();
        assert_eq!(m.points.len(), 2);
    }

    #[test]
    fn points_are_not_snapped() {
        // A grid step is one unit here; a line's vertex rounds onto it, a
        // point feature stays put.
        let mut b = builder();
        b.point([12.3, -45.6]);
        b.begin_line();
        b.points(&[[12.3, -45.6], [30.0, 0.0]]);
        b.end_line();
        let m = b.finish();
        let at = |i: usize| [m.vertices[3 * i], m.vertices[3 * i + 1]];
        assert_eq!(at(m.points[0] as usize), [0.123, -0.456]);
        assert_eq!(at(1), [0.12, -0.46]);
    }
}
