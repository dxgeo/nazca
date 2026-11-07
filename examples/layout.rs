//! A layout through the SVG backend (`plans/rendering.org` § V1 SLICE, step
//! 6): one view, no tiles and no cache, prepared at print resolution, the
//! same pipeline the interactive spikes run.
//!
//!   cargo run -p nazca --features mlt --example layout -- out.svg [--bearing 20]
//!
//! Two layers on an 8 x 6 inch page at 150 dpi: an MLT basemap tile
//! for Massachusetts (decoded by nazca's `mlt` feature) under the WKB
//! counties of `tests/fixtures/counties_sample.tsv`.

#[path = "spike/data.rs"]
#[allow(dead_code)]
mod data;

use nazca::grid::{Rect, TileId};
use nazca::svg::SvgWriter;
use nazca::{Camera, MeshBuilder, Palette, Style, TileGrid, mlt};

const BASEMAP: &[u8] = include_bytes!("../tests/fixtures/basemap/ma_6_19_23.mlt");

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = || -> ! {
        eprintln!("usage: layout OUT.svg [--bearing degrees]");
        std::process::exit(2);
    };
    let out = args.next().unwrap_or_else(|| usage());
    let bearing: f64 = match (args.next().as_deref(), args.next()) {
        (None, _) => 0.0,
        (Some("--bearing"), Some(v)) => v.parse().unwrap_or_else(|_| usage()),
        _ => usage(),
    };

    let counties = data::Counties::load();
    let dpi = 150.0;
    let mut camera = Camera::new([0.0, 0.0], 1.0, [(8.0 * dpi) as u32, (6.0 * dpi) as u32]);
    camera.bearing = bearing.to_radians();
    camera.fit_bounds(counties.bounds, 0.25 * dpi);

    let mut svg = SvgWriter::new(camera.size_px, Some([0.86, 0.91, 0.95, 1.0]));

    // The basemap: every county the tile holds, plain.
    let grid = TileGrid::web_mercator(512, 20).unwrap();
    let tile = TileId { z: 6, x: 19, y: 23 };
    let bounds: Rect = grid.bounds(tile).unwrap();
    let mut basemap = MeshBuilder::for_view(&camera);
    basemap.set_class(1);
    for layer in mlt::layers(BASEMAP) {
        let features = layer.unwrap().decode().unwrap();
        for i in 0..features.len() {
            features.read(i, bounds, &mut basemap).unwrap();
        }
    }
    let mut basemap_style = Palette::new();
    basemap_style.set(1, Style { fill: [0.91, 0.89, 0.85, 1.0], line: [0.72, 0.70, 0.63, 1.0], line_width: 0.8, point_size: 0.0 });
    svg.begin_layer();
    svg.draw(&basemap.finish(), &camera, &basemap_style);

    // The counties, prepared for this page.
    let mut mesh = MeshBuilder::for_view(&camera);
    counties.prepare_into(&mut mesh, camera.visible_bounds());
    let mut palette = data::palette();
    for class in 1..=6 {
        let mut s = *palette.get(class);
        s.fill[3] = 0.85;
        s.line_width = 1.5;
        palette.set(class, s);
    }
    svg.begin_layer();
    svg.draw(&mesh.finish(), &camera, &palette);

    let doc = svg.finish();
    std::fs::write(&out, &doc).unwrap();
    println!("wrote {out}: {}x{} px, {} KB", camera.size_px[0], camera.size_px[1], doc.len() / 1024);
}
