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
//! Each vertex also carries its *distance* along its line or ring
//! ([`Mesh::distances`]), measured along the snapped and decimated geometry
//! before clipping, so a dash pattern follows the line handed in rather than
//! restarting where nazca clipped it. A line handed in as pieces starts each
//! from [`MeshBuilder::set_line_start`], measured by
//! [`MeshBuilder::line_length`] so the pieces join exactly. Distances are
//! `f64`, so a vertex far along a long line keeps its place in the pattern.
//!
//! *A border two rings share is drawn once* ([`Mesh::partners`]). Each ring
//! outlines it, running it its own way from its own start, so drawn twice it
//! would be two dash patterns out of phase, and a translucent line would blend
//! twice. When the mesh is finished, ring edges that overlap along one line of
//! the grid are split at each other's ends, so the pieces pair one to one, and
//! the earlier piece of each pair names the later. A backend draws the earlier
//! only where the later isn't drawn: hidden by a state, or in a class with no
//! line. Matched by overlap rather than by equal ends, since one county's
//! straight edge often meets two neighbours' at a T-junction.
//!
//! So a border takes the look of the last feature fed that outlines it, and
//! where two neighbours' classes differ in colour, width or dash, which one
//! that is matters. Every tile holding the border must agree, or its look
//! changes at a tile edge: feed each tile's features in one order, the same
//! for every tile (by feature id, say). Drawing emphatic features last is
//! the embedder's, by feeding them last or drawing them as a state.
//!
//! Culling whole features before decoding them needs a bounding box the
//! source already stores (a GeoPackage envelope, a GeoParquet covering
//! column, a FlatGeobuf index): the caller skips those features rather than
//! hand them over. WKB carries no bbox of its own.

use crate::earcut::Earcut;
use crate::grid::{TileFrame, TileGrid, TileId};
use crate::sink::Sink;
use std::collections::HashMap;
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
#[non_exhaustive]
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
    /// Empty, or one per vertex: how far along its line or ring it lies, in
    /// local units, where dash patterns are measured from. Empty reads as 0
    /// throughout. `f64`, since a vertex thousands of kilometres along its
    /// line at a deep level is millions of pixels along, past what an `f32`
    /// holds to a pixel.
    pub distances: Vec<f64>,
    /// Empty, or one per outline segment: the later segment, by its index
    /// into `outlines` pairs, that draws the same stretch of a border two
    /// rings share, or [`Mesh::NO_PARTNER`]. A segment is drawn only where
    /// its partner isn't. Empty when no border is shared.
    pub partners: Vec<u32>,
}

impl Mesh {
    /// A segment's partner when it has none.
    pub const NO_PARTNER: u32 = u32::MAX;

    /// An empty mesh in `frame`, for an embedder building one by hand.
    pub fn new(frame: TileFrame) -> Mesh {
        Mesh {
            frame,
            vertices: Vec::new(),
            fills: Vec::new(),
            outlines: Vec::new(),
            points: Vec::new(),
            rings: Vec::new(),
            polygons: Vec::new(),
            ids: Vec::new(),
            distances: Vec::new(),
            partners: Vec::new(),
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len() / 3
    }

    /// The bytes its buffers hold, which is what uploading it costs: the
    /// size to give [`crate::TileCache::insert`].
    pub fn byte_size(&self) -> usize {
        4 * (self.vertices.len() + self.fills.len() + self.outlines.len() + self.points.len() + self.ids.len())
            + 4 * self.partners.len()
            + 8 * (self.distances.len() + self.rings.len() + self.polygons.len())
    }

    /// Vertex `i`'s distance along its line or ring.
    pub(crate) fn distance(&self, i: u32) -> f64 {
        self.distances.get(i as usize).copied().unwrap_or(0.0)
    }

    /// Outline segment `k`'s partner, if it has one.
    pub(crate) fn partner(&self, k: usize) -> Option<usize> {
        self.partners.get(k).filter(|&&p| p != Mesh::NO_PARTNER).map(|&p| p as usize)
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
    // The polygon being received, in grid coordinates, and each kept ring
    // point's distance along its ring, in grid steps.
    pts: Vec<[f64; 2]>,
    dists: Vec<f64>,
    rings: Vec<Range<usize>>,
    ring_start: usize,
    rings_seen: usize,
    // The line being received, if any, in grid coordinates, where it starts
    // along the whole line, and its points' distances.
    line: Option<Vec<[f64; 2]>>,
    line_start: f64,
    line_dists: Vec<f64>,
    scratch: [Vec<[f64; 2]>; 2],
    ring_scratch: [Vec<[f64; 3]>; 2],
    earcut: Earcut,
    tris: Vec<u32>,
    // Every outlined ring edge so far, for `share_borders`.
    ring_edges: Vec<RingEdge>,
}

/// An outlined ring edge, in grid coordinates, and its segment's index
/// among the outline pairs.
struct RingEdge {
    a: [i64; 2],
    b: [i64; 2],
    segment: usize,
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
            mesh: Mesh::new(frame),
            class: 0.0,
            feature: 0,
            half_cells,
            clip,
            pts: Vec::new(),
            dists: Vec::new(),
            rings: Vec::new(),
            ring_start: 0,
            rings_seen: 0,
            line: None,
            line_start: 0.0,
            line_dists: Vec::new(),
            scratch: [Vec::new(), Vec::new()],
            ring_scratch: [Vec::new(), Vec::new()],
            earcut: Earcut::new(),
            tris: Vec::new(),
            ring_edges: Vec::new(),
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

    /// Where the next line's first vertex lies along the whole line it is a
    /// piece of, in map units, so a dash pattern runs on from the piece
    /// before. For the next line only; the one after starts at 0 again.
    ///
    /// The seam joins exactly when `distance` is the pieces before measured
    /// by [`MeshBuilder::line_length`], summed. Measured on the source's own
    /// geometry it is only as close as that is to nazca's measure, which
    /// runs along the line as snapped and decimated at this tile's level:
    /// snapping drops the wiggles finer than its grid and moves every
    /// vertex, so on a line of many short segments the two part at any zoom,
    /// and the seam jumps by the difference. A line handed whole to every
    /// tile needs no start.
    pub fn set_line_start(&mut self, distance: f64) {
        self.line_start = distance / self.mesh.frame.half_span * self.half_cells;
    }

    /// The length of the line through `xy`, in map units, as this builder
    /// measures it: along the line snapped to its grid and decimated, as
    /// [`Sink::points`] would take it. The pieces of a longer line, each
    /// measured so, give the next piece's start for
    /// [`MeshBuilder::set_line_start`], and the pattern runs on across the
    /// seam as if the line were handed in whole.
    ///
    /// The grid is fixed in map units for a level, so every tile at a level
    /// measures a line alike: a piece may be measured by a builder for any
    /// of them, once for the level. They part only for a point within
    /// rounding of halfway between two grid points, which each tile's own
    /// arithmetic may snap its own way.
    pub fn line_length(&self, xy: &[[f64; 2]]) -> f64 {
        let mut line = Vec::with_capacity(xy.len());
        for &p in xy {
            push_decimated(&mut line, 0, self.snap(p));
        }
        let steps = line.windows(2).fold(0.0, |d, w| d + (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1]));
        steps / self.half_cells * self.mesh.frame.half_span
    }

    /// The mesh, with the borders its rings share matched (see the module
    /// docs).
    pub fn finish(mut self) -> Mesh {
        self.share_borders();
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

    /// Adds a vertex at grid point `g`, `d` grid steps along its line or
    /// ring.
    fn push_vertex(&mut self, g: [f64; 2], d: f64) -> u32 {
        let index = self.mesh.vertex_count() as u32;
        let s = self.half_cells;
        self.mesh.vertices.extend_from_slice(&[(g[0] / s) as f32, (g[1] / s) as f32, self.class]);
        self.mesh.ids.push(self.feature);
        self.mesh.distances.push(d / s);
        index
    }

    /// Whether the edge `a`--`b` lies along the clip boundary, where only
    /// clipping puts edges.
    fn on_boundary(&self, a: [f64; 2], b: [f64; 2]) -> bool {
        let c = self.clip;
        (a[0] == b[0] && a[0].abs() == c) || (a[1] == b[1] && a[1].abs() == c)
    }

    /// Clips the ring `pts[start..]`, `length` grid steps around, to the
    /// clip square in place, its points carrying their distances through.
    fn clip_ring(&mut self, start: usize, length: f64) {
        let [mut a, mut b] = std::mem::take(&mut self.ring_scratch);
        a.clear();
        a.extend(self.pts[start..].iter().zip(&self.dists[start..]).map(|(p, &d)| [p[0], p[1], d]));
        let c = self.clip;
        for (axis, bound, keep_greater) in [(0, -c, true), (0, c, false), (1, -c, true), (1, c, false)] {
            clip_edge(&a, &mut b, axis, bound, keep_greater, length);
            std::mem::swap(&mut a, &mut b);
            if a.is_empty() {
                break;
            }
        }
        b.clear();
        for &p in &a {
            push_decimated(&mut b, 0, p);
        }
        self.pts.truncate(start);
        self.dists.truncate(start);
        self.pts.extend(b.iter().map(|p| [p[0], p[1]]));
        self.dists.extend(b.iter().map(|p| p[2]));
        self.ring_scratch = [a, b];
    }

    /// Outlines `line`, whose points lie `dists` grid steps along it, in
    /// runs clipped to the clip square.
    fn end_line_run(&mut self, line: &[[f64; 2]], dists: &[f64]) {
        // The index of the last vertex, while the line runs on inside.
        let mut prev: Option<u32> = None;
        for (i, w) in line.windows(2).enumerate() {
            let Some((p, q)) = clip_segment(w[0], w[1], self.clip) else {
                prev = None;
                continue;
            };
            let along = |p: [f64; 2]| dists[i] + (p[0] - w[0][0]).hypot(p[1] - w[0][1]);
            let from = match prev {
                Some(v) if p == w[0] => v,
                _ => self.push_vertex(p, along(p)),
            };
            let to = self.push_vertex(q, along(q));
            self.mesh.outlines.extend_from_slice(&[from, to]);
            prev = (q == w[1]).then_some(to);
        }
    }

    /// Matches the borders the mesh's rings share (see the module docs):
    /// groups the outlined ring edges by the line of the grid they lie on,
    /// splits those that overlap another at every end of one inside them,
    /// and pairs each piece with the next one, in feed order, over the same
    /// stretch.
    fn share_borders(&mut self) {
        let edges = std::mem::take(&mut self.ring_edges);
        // A line of the grid: its direction, primitive and pointing right
        // (or up), and the cross product every point on it shares.
        let mut lines: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
        for (n, e) in edges.iter().enumerate() {
            let [ux, uy] = direction(e.a, e.b);
            lines.entry([ux, uy, ux * e.a[1] - uy * e.a[0]]).or_default().push(n);
        }
        // Each edge to split: its line, and the positions along it from its
        // start to its end.
        let mut plan: HashMap<usize, ([i64; 3], Vec<i64>)> = HashMap::new();
        let mut ends = Vec::new();
        for (&line, members) in &lines {
            if members.len() < 2 {
                continue;
            }
            let along = |p: [i64; 2]| line[0] * p[0] + line[1] * p[1];
            ends.clear();
            ends.extend(members.iter().flat_map(|&n| [along(edges[n].a), along(edges[n].b)]));
            ends.sort_unstable();
            ends.dedup();
            for &n in members {
                let (ta, tb) = (along(edges[n].a), along(edges[n].b));
                let inside = ends.iter().copied().filter(|&t| t > ta.min(tb) && t < ta.max(tb));
                let mut cuts: Vec<i64> = std::iter::once(ta).chain(inside).chain(std::iter::once(tb)).collect();
                if ta > tb {
                    let last = cuts.len() - 1;
                    cuts[1..last].reverse();
                }
                plan.insert(edges[n].segment, (line, cuts));
            }
        }
        if plan.is_empty() {
            return;
        }

        let old = std::mem::take(&mut self.mesh.outlines);
        let mut outlines = Vec::with_capacity(old.len());
        let mut stretches: HashMap<[i64; 5], Vec<u32>> = HashMap::new();
        let mut edge_of = edges.iter().map(|e| (e.segment, e)).collect::<HashMap<_, _>>();
        for (k, s) in old.chunks_exact(2).enumerate() {
            let Some((line, cuts)) = plan.get(&k) else {
                outlines.extend_from_slice(s);
                continue;
            };
            let e = edge_of.remove(&k).unwrap();
            let step = line[0] * line[0] + line[1] * line[1];
            let t0 = cuts[0];
            let mut from = s[0];
            for j in 1..cuts.len() {
                let to = if j + 1 == cuts.len() {
                    s[1]
                } else {
                    let m = (cuts[j] - t0) / step;
                    self.split_vertex(s[0], e.a, [e.a[0] + m * line[0], e.a[1] + m * line[1]])
                };
                let stretch = [line[0], line[1], line[2], cuts[j - 1].min(cuts[j]), cuts[j - 1].max(cuts[j])];
                stretches.entry(stretch).or_default().push((outlines.len() / 2) as u32);
                outlines.extend_from_slice(&[from, to]);
                from = to;
            }
        }
        self.mesh.partners = vec![Mesh::NO_PARTNER; outlines.len() / 2];
        for pieces in stretches.values() {
            for w in pieces.windows(2) {
                self.mesh.partners[w[0] as usize] = w[1];
            }
        }
        self.mesh.outlines = outlines;
    }

    /// A vertex at grid point `g` on the ring edge leaving vertex `from`,
    /// which lies at grid point `a`: its feature, class and distance.
    fn split_vertex(&mut self, from: u32, a: [i64; 2], g: [i64; 2]) -> u32 {
        let index = self.mesh.vertex_count() as u32;
        let f = from as usize;
        let s = self.half_cells;
        let class = self.mesh.vertices[f * 3 + 2];
        let d = self.mesh.distance(from) + ((g[0] - a[0]) as f64).hypot((g[1] - a[1]) as f64) / s;
        self.mesh.vertices.extend_from_slice(&[(g[0] as f64 / s) as f32, (g[1] as f64 / s) as f32, class]);
        let id = self.mesh.ids[f];
        self.mesh.ids.push(id);
        self.mesh.distances.push(d);
        index
    }
}

/// The primitive direction of the grid line through `a` and `b`, pointing
/// right, or up when the line is vertical.
fn direction(a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let g = gcd(dx.abs(), dy.abs()).max(1);
    let (ux, uy) = (dx / g, dy / g);
    if ux < 0 || (ux == 0 && uy < 0) { [-ux, -uy] } else { [ux, uy] }
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Appends `p` to the ring or line `buf[start..]`, unless it repeats the
/// last point; and replaces the last point instead if it lies exactly on the
/// way from the one before to `p`. Only X and Y are compared; anything after
/// them (a distance) travels with the point kept.
fn push_decimated<const N: usize>(buf: &mut Vec<[f64; N]>, start: usize, p: [f64; N]) {
    let xy = |q: &[f64; N]| [q[0], q[1]];
    let n = buf.len() - start;
    if n >= 1 && xy(&buf[buf.len() - 1]) == xy(&p) {
        return;
    }
    if n >= 2 && passes_through(xy(&buf[buf.len() - 2]), xy(&buf[buf.len() - 1]), xy(&p)) {
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
/// other coordinate snapped to the grid, their distance interpolated along
/// the edge they cut. A ring's distances rise along it but for its closing
/// edge, which ends where the ring began, `length` along; an edge whose
/// distance falls is taken to be that one. Edges along the clip boundary
/// aren't outlined, so their distances are never read.
fn clip_edge(input: &[[f64; 3]], out: &mut Vec<[f64; 3]>, axis: usize, bound: f64, keep_greater: bool, length: f64) {
    out.clear();
    let inside = |p: [f64; 3]| if keep_greater { p[axis] >= bound } else { p[axis] <= bound };
    let cross = |a: [f64; 3], b: [f64; 3]| {
        let t = (bound - a[axis]) / (b[axis] - a[axis]);
        let other = 1 - axis;
        let mut p = [0.0; 3];
        p[axis] = bound;
        p[other] = (a[other] + t * (b[other] - a[other])).round();
        let end = if b[2] < a[2] { b[2] + length } else { b[2] };
        p[2] = a[2] + t * (end - a[2]);
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
            let i = self.push_vertex(g, 0.0);
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
            let mut dists = std::mem::take(&mut self.line_dists);
            dists.clear();
            let mut d = self.line_start;
            for (i, p) in line.iter().enumerate() {
                if i > 0 {
                    d += (p[0] - line[i - 1][0]).hypot(p[1] - line[i - 1][1]);
                }
                dists.push(d);
            }
            self.end_line_run(&line, &dists);
            self.line_dists = dists;
            self.scratch[0] = line;
        }
        self.line_start = 0.0;
    }

    fn begin_polygon(&mut self) {
        self.pts.clear();
        self.dists.clear();
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
            self.dists.truncate(start);
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

        // Distances around the ring as it stands, before clipping.
        self.dists.truncate(start);
        let mut length = 0.0;
        for i in start..self.pts.len() {
            if i > start {
                length += (self.pts[i][0] - self.pts[i - 1][0]).hypot(self.pts[i][1] - self.pts[i - 1][1]);
            }
            self.dists.push(length);
        }
        if let (Some(first), Some(last)) = (self.pts.get(start), self.pts.last()) {
            length += (first[0] - last[0]).hypot(first[1] - last[1]);
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
                self.dists.truncate(start);
            } else if lo[0] < -c || lo[1] < -c || hi[0] > c || hi[1] > c {
                self.clip_ring(start, length);
            }
        }
        if self.pts.len() - start < 3 {
            self.pts.truncate(start);
            self.dists.truncate(start);
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
            self.push_vertex(self.pts[i], self.dists[i]);
        }
        let grid = |p: [f64; 2]| [p[0] as i64, p[1] as i64];
        for r in 0..self.rings.len() {
            let ring = self.rings[r].clone();
            for i in ring.clone() {
                let next = if i + 1 == ring.end { ring.start } else { i + 1 };
                if !self.on_boundary(self.pts[i], self.pts[next]) {
                    let segment = self.mesh.outlines.len() / 2;
                    self.ring_edges.push(RingEdge { a: grid(self.pts[i]), b: grid(self.pts[next]), segment });
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

    /// The distances, in grid steps (also map units, with this frame).
    fn distances(m: &Mesh) -> Vec<f64> {
        m.distances.iter().map(|&d| (d * 100.0 * 1000.0).round() / 1000.0).collect()
    }

    /// Outline segment `k`'s ends, in grid steps.
    fn segment(m: &Mesh, k: usize) -> [[f64; 2]; 2] {
        let p = positions(m);
        [p[m.outlines[2 * k] as usize], p[m.outlines[2 * k + 1] as usize]]
    }

    #[test]
    fn lines_carry_their_distance_from_their_start() {
        let mut b = builder();
        b.set_line_start(5.0);
        b.begin_line();
        b.points(&[[0.0, 0.0], [30.0, 0.0], [30.0, 40.0]]);
        b.end_line();
        // The start was for that line only.
        b.begin_line();
        b.points(&[[0.0, 50.0], [10.0, 50.0]]);
        b.end_line();
        assert_eq!(distances(&b.finish()), [5.0, 35.0, 75.0, 0.0, 10.0]);

        // Out past the right edge and back: each cut lies as far along as
        // the line it was cut from.
        let mut b = builder();
        b.begin_line();
        b.points(&[[0.0, 0.0], [150.0, 0.0], [150.0, 20.0], [0.0, 20.0]]);
        b.end_line();
        let m = b.finish();
        assert_eq!(positions(&m), [[0.0, 0.0], [110.0, 0.0], [110.0, 20.0], [0.0, 20.0]]);
        assert_eq!(distances(&m), [0.0, 110.0, 210.0, 320.0]);
    }

    #[test]
    fn a_line_in_measured_pieces_matches_it_whole() {
        // Steps of 1.3 wiggling 2.5 snap to steps of 1 or 2 wiggling 3, so
        // the source's own length is far from nazca's; none lies near
        // halfway between grid points, nor in line with its neighbours.
        let line: Vec<[f64; 2]> = (0..40).map(|i| [i as f64 * 1.3 - 40.05, if i % 2 == 0 { 0.2 } else { 2.7 }]).collect();
        let mut b = builder();
        b.begin_line();
        b.points(&line);
        b.end_line();
        let whole = distances(&b.finish());

        // The pieces meet at vertices 7 and 15, measured by the tile to the
        // right, at the same level.
        let right = MeshBuilder::new(TileFrame { center: [200.0, 0.0], half_span: 100.0 }, Detail { tile_px: 200.0, grid_px: 1.0, guard_px: 10.0 });
        let mut b = builder();
        let mut start = 0.0;
        for piece in [&line[..=7], &line[7..=15], &line[15..]] {
            b.set_line_start(start);
            b.begin_line();
            b.points(piece);
            b.end_line();
            start += right.line_length(piece);
        }
        let expected: Vec<f64> = whole[..=7].iter().chain(&whole[7..=15]).chain(&whole[15..]).copied().collect();
        assert_eq!(distances(&b.finish()), expected);
        let source: f64 = line.windows(2).map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1])).sum();
        assert!((start - whole[39]).abs() < 1e-3 && (source - start).abs() > 5.0, "{start} {source}");
    }

    #[test]
    fn rings_carry_their_distance_through_clipping() {
        let mut b = builder();
        polygon(&mut b, &[&[[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]]);
        assert_eq!(distances(&b.finish()), [0.0, 20.0, 40.0, 60.0]);

        // Cut by the right edge: where it re-enters lies 200 + 20 + 40
        // along.
        let mut b = builder();
        polygon(&mut b, &[&[[-50.0, -10.0], [150.0, -10.0], [150.0, 10.0], [-50.0, 10.0]]]);
        let m = b.finish();
        assert_eq!(positions(&m), [[-50.0, -10.0], [110.0, -10.0], [110.0, 10.0], [-50.0, 10.0]]);
        assert_eq!(distances(&m), [0.0, 160.0, 260.0, 420.0]);

        // The same ring started on the right, so the edge that closes it is
        // cut: the cut lies 400 along, not on the way back to 0.
        let mut b = builder();
        polygon(&mut b, &[&[[150.0, -10.0], [150.0, 10.0], [-50.0, 10.0], [-50.0, -10.0]]]);
        let m = b.finish();
        assert_eq!(positions(&m), [[110.0, -10.0], [110.0, 10.0], [-50.0, 10.0], [-50.0, -10.0]]);
        assert_eq!(distances(&m), [400.0, 60.0, 220.0, 240.0]);
    }

    #[test]
    fn a_shared_border_is_paired_with_its_later_copy() {
        let mut b = builder();
        b.set_feature(1);
        polygon(&mut b, &[&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]]);
        // Nothing shared yet.
        b.set_feature(2);
        polygon(&mut b, &[&[[10.0, 0.0], [20.0, 0.0], [20.0, 10.0], [10.0, 10.0]]]);
        let m = b.finish();
        // The first square's right edge, and the second's left, run the
        // other way; edges in line along y = 0 that only touch aren't paired.
        assert_eq!(m.outlines.len() / 2, 8);
        let paired: Vec<(usize, usize)> = (0..8).filter_map(|k| m.partner(k).map(|p| (k, p))).collect();
        assert_eq!(paired, [(1, 7)]);
        assert_eq!(segment(&m, 1), [[10.0, 0.0], [10.0, 10.0]]);
        assert_eq!(segment(&m, 7), [[10.0, 10.0], [10.0, 0.0]]);

        let mut b = builder();
        polygon(&mut b, &[&[[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]]);
        assert!(b.finish().partners.is_empty());
    }

    #[test]
    fn a_t_junction_splits_the_edge_it_meets() {
        let mut b = builder();
        // One wide polygon over two narrow ones: its bottom edge is one
        // edge, theirs two, meeting at (10, 10).
        b.set_feature(1);
        polygon(&mut b, &[&[[0.0, 10.0], [20.0, 10.0], [20.0, 20.0], [0.0, 20.0]]]);
        b.set_feature(2);
        polygon(&mut b, &[&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]]);
        b.set_feature(3);
        polygon(&mut b, &[&[[10.0, 0.0], [20.0, 0.0], [20.0, 10.0], [10.0, 10.0]]]);
        let m = b.finish();
        assert_eq!(m.outlines.len() / 2, 13);
        assert_eq!(segment(&m, 0), [[0.0, 10.0], [10.0, 10.0]]);
        assert_eq!(segment(&m, 1), [[10.0, 10.0], [20.0, 10.0]]);
        // The vertex the split adds is the wide polygon's, 10 along it.
        let split = m.outlines[1] as usize;
        assert_eq!((m.ids[split], distances(&m)[split]), (1, 10.0));
        let mut paired = 0;
        for k in 0..13 {
            if let Some(p) = m.partner(k) {
                assert!(p > k);
                let [a, b] = segment(&m, k);
                assert_eq!(segment(&m, p), [b, a], "segment {k}");
                paired += 1;
            }
        }
        // Both halves of the wide edge, and the narrow ones' shared side.
        assert_eq!(paired, 3);
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
