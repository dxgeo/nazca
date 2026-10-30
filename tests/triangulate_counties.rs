//! Triangulates every county in the WKB fixtures with earcut and checks the
//! triangles cover exactly the polygon: all counter-clockwise, and their
//! total area equals the exterior's minus the holes'. See
//! `fixtures/make_fixtures.sh`.

use nazca::earcut::Earcut;
use nazca::{Sink, wkb};
use std::ops::Range;

/// One polygon's points and ring ranges, as `triangulate` takes them.
type Polygon = (Vec<[f64; 2]>, Vec<Range<usize>>);

/// Collects each polygon's rings.
#[derive(Default)]
struct Polygons {
    pts: Vec<[f64; 2]>,
    rings: Vec<Range<usize>>,
    ring_start: usize,
    done: Vec<Polygon>,
}

impl Sink for Polygons {
    fn point(&mut self, _: [f64; 2]) {}
    fn points(&mut self, xy: &[[f64; 2]]) {
        self.pts.extend_from_slice(xy);
    }
    fn begin_ring(&mut self) {
        self.ring_start = self.pts.len();
    }
    fn end_ring(&mut self) {
        self.pts.pop(); // the closing repeat of the first point
        self.rings.push(self.ring_start..self.pts.len());
    }
    fn end_polygon(&mut self) {
        self.done.push((std::mem::take(&mut self.pts), std::mem::take(&mut self.rings)));
    }
}

fn shoelace(pts: &[[f64; 2]]) -> f64 {
    (0..pts.len()).map(|i| {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        a[0] * b[1] - b[0] * a[1]
    }).sum::<f64>().abs() / 2.0
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn check(tsv: &str) -> usize {
    let mut polygons = 0;
    let mut earcut = Earcut::new();
    for line in tsv.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        let mut p = Polygons::default();
        wkb::read(&hex(f[10]), &mut p).unwrap();
        for (mut pts, rings) in p.done {
            // Measure relative to the polygon's first point: shoelace terms on
            // raw longitude/latitude are thousands of times the area of a
            // county, and their rounding would swamp the comparison.
            let origin = pts[0];
            for q in &mut pts {
                *q = [q[0] - origin[0], q[1] - origin[1]];
            }
            let expected = shoelace(&pts[rings[0].clone()]) - rings[1..].iter().map(|r| shoelace(&pts[r.clone()])).sum::<f64>();
            let mut tris = Vec::new();
            earcut.triangulate(&pts, &rings, &mut tris);
            let area: f64 = tris.chunks(3).map(|t| {
                let (a, b, c) = (pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]);
                ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0
            }).sum();
            assert!(tris.chunks(3).all(|t| {
                let (a, b, c) = (pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]);
                (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0.0
            }), "{} {}: a triangle is not counter-clockwise", f[0], f[1]);
            assert!((area - expected).abs() <= expected * 1e-9, "{} {}: area {area} vs {expected}", f[0], f[1]);
            polygons += 1;
        }
    }
    polygons
}

#[test]
fn counties_sample_triangulates_exactly() {
    // 28 counties, two of them multipolygons of two parts.
    assert_eq!(check(include_str!("fixtures/counties_sample.tsv")), 30);
}

#[test]
#[ignore = "needs tests/fixtures/local/us_counties.tsv"]
fn us_counties_triangulate_exactly() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/local/us_counties.tsv");
    let tsv = std::fs::read_to_string(path).expect("run tests/fixtures/make_fixtures.sh first");
    assert!(check(&tsv) >= 3222);
}
