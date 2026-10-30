//! What both spikes draw -- the native one (`main.rs`) and the browser one
//! (`spike-web/`, which includes this file): the Massachusetts counties from
//! nazca's WKB fixture, projected to Web Mercator and prepared tile by tile
//! as the camera asks for them.

use nazca::grid::Rect;
use nazca::{Detail, Mesh, MeshBuilder, Sink, TileGrid, TileId, wkb};

const TSV: &str = include_str!("../../tests/fixtures/counties_sample.tsv");

/// Projects longitude/latitude to Web Mercator metres on the way into
/// another sink.
///
/// This stands in for `geoscribe`: projection is CRS math, which nazca does
/// not do (`project.org` § CHARTER), so a caller does it inside its sink --
/// the seam `plans/rendering.org` § INPUT describes. Spherical Web Mercator
/// is simple enough to write here in two lines.
struct ToWebMercator<'a, S: Sink>(&'a mut S);

const R: f64 = 6_378_137.0;

fn project([lon, lat]: [f64; 2]) -> [f64; 2] {
    let (lon, lat) = (lon.to_radians(), lat.to_radians());
    [R * lon, R * (std::f64::consts::FRAC_PI_4 + lat / 2.0).tan().ln()]
}

impl<S: Sink> Sink for ToWebMercator<'_, S> {
    fn point(&mut self, xy: [f64; 2]) {
        self.0.point(project(xy));
    }
    fn points(&mut self, xy: &[[f64; 2]]) {
        let mut block = [[0.0; 2]; 64];
        for chunk in xy.chunks(64) {
            for (out, &p) in block.iter_mut().zip(chunk) {
                *out = project(p);
            }
            self.0.points(&block[..chunk.len()]);
        }
    }
    fn begin_line(&mut self) { self.0.begin_line() }
    fn end_line(&mut self) { self.0.end_line() }
    fn begin_polygon(&mut self) { self.0.begin_polygon() }
    fn end_polygon(&mut self) { self.0.end_polygon() }
    fn begin_ring(&mut self) { self.0.begin_ring() }
    fn end_ring(&mut self) { self.0.end_ring() }
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

/// The fixture's Massachusetts rows (GEOID prefix 25).
fn massachusetts() -> impl Iterator<Item = Vec<&'static str>> {
    TSV.lines().skip(1).map(|l| l.split('\t').collect::<Vec<_>>()).filter(|f| f[0].starts_with("25"))
}

struct County {
    /// In Web Mercator metres.
    bounds: Rect,
    wkb: Vec<u8>,
    class: u8,
}

/// The counties, held as WKB the way a tile source would hold them, each
/// with the bounds it stores (DuckDB's) for culling.
pub struct Counties {
    counties: Vec<County>,
    /// All of them, in Web Mercator metres.
    pub bounds: Rect,
}

impl Counties {
    pub fn load() -> Counties {
        let mut bounds = Rect { min: [f64::MAX; 2], max: [f64::MIN; 2] };
        let counties: Vec<County> = massachusetts()
            .enumerate()
            .map(|(i, f)| {
                // Mercator preserves the order of coordinates, so projected
                // corners bound the projected shapes.
                let lo = project([f[6].parse().unwrap(), f[7].parse().unwrap()]);
                let hi = project([f[8].parse().unwrap(), f[9].parse().unwrap()]);
                bounds.min = [bounds.min[0].min(lo[0]), bounds.min[1].min(lo[1])];
                bounds.max = [bounds.max[0].max(hi[0]), bounds.max[1].max(hi[1])];
                County { bounds: Rect { min: lo, max: hi }, wkb: hex(f[10]), class: 1 + (i % 6) as u8 }
            })
            .collect();
        Counties { counties, bounds }
    }

    /// Prepares tile `id` of `grid`, as a tile route would serve it: the
    /// counties whose stored bounds reach the tile or its guard band, clipped
    /// to it.
    pub fn prepare(&self, grid: &TileGrid, id: TileId) -> Mesh {
        let mut builder = MeshBuilder::for_tile(grid, id).unwrap();
        let reach = grid.bounds(id).unwrap().expand(Detail::default().guard_px * grid.resolution(id.z).unwrap());
        for c in self.counties.iter().filter(|c| c.bounds.intersects(&reach)) {
            builder.set_class(c.class);
            wkb::read(&c.wkb, &mut ToWebMercator(&mut builder)).unwrap();
        }
        builder.finish()
    }
}

/// Class 0 unused, then six muted fills for neighbouring counties.
pub const PALETTE: [[f32; 4]; 16] = {
    let mut p = [[0.0, 0.0, 0.0, 1.0]; 16];
    p[1] = [0.80, 0.86, 0.75, 1.0];
    p[2] = [0.93, 0.85, 0.70, 1.0];
    p[3] = [0.76, 0.84, 0.90, 1.0];
    p[4] = [0.90, 0.78, 0.78, 1.0];
    p[5] = [0.84, 0.80, 0.90, 1.0];
    p[6] = [0.95, 0.92, 0.76, 1.0];
    p
};

pub const OUTLINE: [f32; 4] = [0.25, 0.27, 0.30, 1.0];
pub const BACKGROUND: [f32; 4] = [0.97, 0.97, 0.95, 1.0];
