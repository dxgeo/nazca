//! What both spikes draw -- the native one (`main.rs`) and the browser one
//! (`spike-web/`, which includes this file): the Massachusetts counties from
//! nazca's WKB fixture, projected to Web Mercator and prepared tile by tile
//! as the camera asks for them, and a raster laid over them.

use nazca::grid::{Rect, TileFrame};
use nazca::raster::{Band, Raster, RasterMesh};
use nazca::{Colormap, Detail, Mesh, MeshBuilder, Palette, Sink, Style, TileGrid, TileId, wkb};

const TSV: &str = include_str!("../../tests/fixtures/counties_sample.tsv");

/// Projects longitude/latitude to Web Mercator metres on the way into
/// another sink.
///
/// This stands in for `geoscribe`, which plans reprojection as of 0.5.1:
/// projection is CRS math, which nazca does not do (`project.org` § CHARTER), so a caller does it inside its sink --
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
    name: &'static str,
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
                County { name: f[1], bounds: Rect { min: lo, max: hi }, wkb: hex(f[10]), class: 1 + (i % 6) as u8 }
            })
            .collect();
        Counties { counties, bounds }
    }

    /// The county a picked feature id names.
    pub fn name(&self, feature: u32) -> Option<&'static str> {
        self.counties.get(feature as usize).map(|c| c.name)
    }

    /// Prepares tile `id` of `grid`, as a tile route would serve it: the
    /// counties whose stored bounds reach the tile or its guard band, clipped
    /// to it, each with its index as its feature id.
    pub fn prepare(&self, grid: &TileGrid, id: TileId) -> Mesh {
        let mut builder = MeshBuilder::for_tile(grid, id).unwrap();
        let reach = grid.bounds(id).unwrap().expand(Detail::default().guard_px * grid.resolution(id.z).unwrap());
        self.prepare_into(&mut builder, reach);
        builder.finish()
    }

    /// Feeds `builder` the counties whose stored bounds reach `reach`.
    pub fn prepare_into(&self, builder: &mut MeshBuilder, reach: Rect) {
        for (i, c) in self.counties.iter().enumerate().filter(|(_, c)| c.bounds.intersects(&reach)) {
            builder.set_class(c.class);
            builder.set_feature(i as u32);
            wkb::read(&c.wkb, &mut ToWebMercator(&mut *builder)).unwrap();
        }
    }
}

/// Six muted fills for neighbouring counties, outlined.
pub fn palette() -> Palette {
    const FILLS: [[f32; 4]; 6] = [
        [0.80, 0.86, 0.75, 1.0],
        [0.93, 0.85, 0.70, 1.0],
        [0.76, 0.84, 0.90, 1.0],
        [0.90, 0.78, 0.78, 1.0],
        [0.84, 0.80, 0.90, 1.0],
        [0.95, 0.92, 0.76, 1.0],
    ];
    let mut p = Palette::new();
    for (i, fill) in FILLS.into_iter().enumerate() {
        p.set(1 + i as u8, Style { fill, line: [0.25, 0.27, 0.30, 1.0], line_width: 1.25, point_size: 0.0 });
    }
    p
}

pub const BACKGROUND: [f32; 4] = [0.97, 0.97, 0.95, 1.0];

/// Which raster to lay over the counties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Overlay {
    None,
    /// Distance from Boston in km, on a ramp; nothing past 150 km.
    Continuous,
    /// The same distance in 25 km bands, one class each.
    Categorical,
}

impl Overlay {
    pub fn next(self) -> Overlay {
        match self {
            Overlay::None => Overlay::Continuous,
            Overlay::Continuous => Overlay::Categorical,
            Overlay::Categorical => Overlay::None,
        }
    }
}

/// The overlay's grid: a longitude/latitude raster over the state, in
/// pixels and degrees.
const RASTER_W: u32 = 192;
const RASTER_H: u32 = 96;
const WEST: f64 = -73.6;
const EAST: f64 = -69.8;
const NORTH: f64 = 43.0;
const SOUTH: f64 = 41.2;

/// The distance overlay, as a band and the mesh it is drawn over.
///
/// The raster is on a longitude/latitude grid, which Web Mercator stretches
/// north-south, so it is warped: a 9 x 9 control grid of its positions,
/// projected the way a caller would through `geoscribe` once it can (planned
/// as of 0.5.1; `plans/rendering.org` § RASTER, mesh warp).
pub fn overlay(kind: Overlay) -> Option<(Vec<u8>, Vec<f32>, RasterMesh, Colormap)> {
    let km = |lon: f64, lat: f64| {
        let (dx, dy) = ((lon + 71.06) * 111.32 * 42.36f64.to_radians().cos(), (lat - 42.36) * 110.57);
        (dx * dx + dy * dy).sqrt()
    };
    let mut classes = Vec::new();
    let mut values = Vec::new();
    for j in 0..RASTER_H {
        for i in 0..RASTER_W {
            let lon = WEST + (EAST - WEST) * (i as f64 + 0.5) / RASTER_W as f64;
            let lat = NORTH - (NORTH - SOUTH) * (j as f64 + 0.5) / RASTER_H as f64;
            let d = km(lon, lat);
            values.push(if d > 150.0 { f32::NAN } else { d as f32 });
            classes.push((d / 25.0).min(255.0) as u8);
        }
    }
    let (cols, rows) = (9u32, 9u32);
    let points: Vec<[f64; 2]> = (0..rows * cols)
        .map(|k| {
            let (i, j) = ((k % cols) as f64 / (cols - 1) as f64, (k / cols) as f64 / (rows - 1) as f64);
            project([WEST + (EAST - WEST) * i, NORTH - (NORTH - SOUTH) * j])
        })
        .collect();
    let (nw, se) = (project([WEST, NORTH]), project([EAST, SOUTH]));
    let frame = TileFrame {
        center: [(nw[0] + se[0]) / 2.0, (nw[1] + se[1]) / 2.0],
        half_span: (se[0] - nw[0]).max(nw[1] - se[1]) / 2.0,
    };
    let mesh = RasterMesh::warped(frame, cols, rows, &points).unwrap();
    let colormap = match kind {
        Overlay::None => return None,
        Overlay::Continuous => Colormap::ramp(&[(0.0, [178, 24, 43, 255]), (60.0, [253, 219, 120, 255]), (150.0, [33, 102, 172, 255])]).unwrap(),
        Overlay::Categorical => Colormap::categorical(&[
            (0, [215, 48, 39, 255]),
            (1, [252, 141, 89, 255]),
            (2, [254, 224, 144, 255]),
            (3, [224, 243, 248, 255]),
            (4, [145, 191, 219, 255]),
            (5, [69, 117, 180, 255]),
        ]),
    };
    Some((classes, values, mesh, colormap))
}

/// The overlay's band, borrowing what [`overlay`] made.
pub fn band<'a>(kind: Overlay, classes: &'a [u8], values: &'a [f32]) -> Raster<'a> {
    let band = match kind {
        Overlay::Categorical => Band::Categorical(classes),
        _ => Band::Continuous { values, nodata: None },
    };
    Raster::new(RASTER_W, RASTER_H, band).unwrap()
}
