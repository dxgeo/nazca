//! Prepares real counties tile by tile and checks the tiles fit back
//! together: with no guard band the tiles partition the plane, so the fill
//! area summed over every tile must equal the counties' own area, to within
//! what snapping to the grid can move. Features are culled per tile by the
//! bounding boxes the fixture stores (DuckDB's), as a caller with stored
//! boxes would. See `fixtures/make_fixtures.sh`.
//!
//! The grid is in degrees -- nazca does not care what the map units are.

use nazca::grid::{Rect, TileGrid};
use nazca::{Detail, MeshBuilder, Sink, wkb};
use std::time::Instant;

/// The exact area and perimeter of what a geometry's polygons cover.
#[derive(Default)]
struct Measure {
    ring: Vec<[f64; 2]>,
    rings_in_polygon: usize,
    area: f64,
    perimeter: f64,
    points: usize,
}

impl Sink for Measure {
    fn point(&mut self, _: [f64; 2]) {}
    fn points(&mut self, xy: &[[f64; 2]]) {
        self.ring.extend_from_slice(xy);
        self.points += xy.len();
    }
    fn begin_polygon(&mut self) {
        self.rings_in_polygon = 0;
    }
    fn begin_ring(&mut self) {
        self.ring.clear();
    }
    fn end_ring(&mut self) {
        let r = &self.ring;
        let o = r[0]; // measured from the ring's first point, for precision
        let twice: f64 = r.windows(2).map(|w| (w[0][0] - o[0]) * (w[1][1] - o[1]) - (w[1][0] - o[0]) * (w[0][1] - o[1])).sum();
        let area = twice.abs() / 2.0;
        self.area += if self.rings_in_polygon == 0 { area } else { -area };
        self.perimeter += r.windows(2).map(|w| ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt()).sum::<f64>();
        self.rings_in_polygon += 1;
    }
}

struct County {
    bounds: Rect,
    wkb: Vec<u8>,
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn overlaps(a: &Rect, b: &Rect) -> bool {
    a.min[0] <= b.max[0] && b.min[0] <= a.max[0] && a.min[1] <= b.max[1] && b.min[1] <= a.max[1]
}

/// Prepares every tile at level `z` the counties touch; checks the area and
/// that every vertex stays in its tile; returns a line of statistics.
fn check(tsv: &str, z: u8) -> String {
    let counties: Vec<County> = tsv
        .lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            let n = |i: usize| f[i].parse::<f64>().unwrap();
            County { bounds: Rect { min: [n(6), n(7)], max: [n(8), n(9)] }, wkb: hex(f[10]) }
        })
        .collect();

    let mut exact = Measure::default();
    for c in &counties {
        wkb::read(&c.wkb, &mut exact).unwrap();
    }
    let mut all = counties[0].bounds;
    for c in &counties {
        all = Rect { min: [all.min[0].min(c.bounds.min[0]), all.min[1].min(c.bounds.min[1])], max: [all.max[0].max(c.bounds.max[0]), all.max[1].max(c.bounds.max[1])] };
    }

    let grid = TileGrid::quadtree([-180.0, 90.0], 360.0, 1, 1, 512, 12).unwrap();
    let detail = Detail { tile_px: 512.0, grid_px: 0.25, guard_px: 0.0 };
    let step = grid.level(z).unwrap().span / (512.0 / 0.25); // a grid step, in degrees
    let started = Instant::now();
    let (mut area, mut tiles, mut vertices, mut triangles) = (0.0, 0, 0, 0);
    for id in grid.tiles_in(z, all) {
        let bounds = grid.bounds(id).unwrap();
        let frame = grid.frame(id).unwrap();
        let mut builder = MeshBuilder::new(frame, detail);
        for c in counties.iter().filter(|c| overlaps(&c.bounds, &bounds)) {
            wkb::read(&c.wkb, &mut builder).unwrap();
        }
        let mesh = builder.finish();
        let v: Vec<[f64; 2]> = mesh.vertices.chunks(3).map(|v| [v[0] as f64, v[1] as f64]).collect();
        assert!(v.iter().all(|p| p[0].abs() <= 1.0 && p[1].abs() <= 1.0), "tile {id:?}: a vertex outside the tile");
        let local: f64 = mesh.fills.chunks(3).map(|t| {
            let (a, b, c) = (v[t[0] as usize], v[t[1] as usize], v[t[2] as usize]);
            ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0
        }).sum();
        assert!(local >= 0.0);
        area += local * frame.half_span * frame.half_span;
        tiles += 1;
        vertices += mesh.vertex_count();
        triangles += mesh.fills.len() / 3;
    }
    let elapsed = started.elapsed();

    // Snapping moves each vertex at most half a step on each axis, so the
    // area can move by at most about a step for every unit of perimeter.
    let tolerance = exact.perimeter * step;
    assert!(
        (area - exact.area).abs() <= tolerance,
        "level {z}: tiles hold {area} square degrees, the counties {}, more than {tolerance} apart",
        exact.area
    );
    format!(
        "level {z}: {tiles} tiles, {} points in, {vertices} vertices out ({:.1}%), {triangles} triangles, area off by {:.2e} of {:.2e} allowed, {:.1} ms",
        exact.points,
        100.0 * vertices as f64 / exact.points as f64,
        (area - exact.area).abs(),
        tolerance,
        elapsed.as_secs_f64() * 1000.0
    )
}

#[test]
fn counties_sample_tiles_fit_back_together() {
    let tsv = include_str!("fixtures/counties_sample.tsv");
    for z in [4, 7, 10] {
        eprintln!("{}", check(tsv, z));
    }
}

/// All 3,222 US counties. Not checked in; generate it with
/// `tests/fixtures/make_fixtures.sh`, then run
/// `cargo test --release -- --ignored --nocapture` for the timings.
#[test]
#[ignore = "needs tests/fixtures/local/us_counties.tsv"]
fn us_counties_tiles_fit_back_together() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/local/us_counties.tsv");
    let tsv = std::fs::read_to_string(path).expect("run tests/fixtures/make_fixtures.sh first");
    for z in [3, 5, 7, 9] {
        eprintln!("{}", check(&tsv, z));
    }
}
