# Nazca

Nazca renders geospatial data. It is a Rust library that draws interactive
maps and writes static layouts from one pipeline, natively and in
WebAssembly. Geometry is prepared once into tiles, then either redrawn every
frame by the GPU backend or written once at print resolution by the SVG
backend.

Current version: 0.3.0.


## PHILOSOPHY

-   **Maps and layouts are equally important.** Neither is a by-product of
    the other. Both run the same preparation, so a layout shows what the
    screen shows.
-   **Lightweight, fast, memory-efficient.** These decide between two designs
    that otherwise tie. Geometry is streamed into a `Sink` and never
    collected into a feature model, and the tile cache holds prepared
    meshes under one byte budget.
-   **The default build is standard-library only.** WKB reading,
    preparation, the tile cache, the camera and the SVG backend need no
    other crate. The GPU backend adds `glow`, and symbols from SVG markup add
    `resvg`. The `mvt` and `mlt` decoders are hand-written and add nothing.
    Any dependency is pure Rust and builds for `wasm32-unknown-unknown`
    unmodified.
-   **Compose rather than embed.** Nazca does not read files, fetch tiles,
    open windows or reproject. It names the tiles a camera needs, and the
    embedder fetches them and hands the bytes in. Coordinates arrive in the
    map's CRS, already projected by the caller. Reprojection is planned for
    [Geoscribe](https://github.com/dxgeo/geoscribe), but is not implemented
    as of its 0.5.1.
-   **No sibling dependencies.** Nazca depends on neither
    [Geosetta](https://github.com/dxgeo/geosetta) nor Geoscribe, and neither
    depends on it as of Geosetta 0.27.0 and Geoscribe 0.5.1. A caller that
    links several of them wires them together.


## SETUP

Nazca needs Rust 1.88 or later (edition 2024).

To use it as a library:

```sh
cargo add nazca
cargo add nazca --features gl,symbols,mvt,mlt
```

The tests run with `cargo test --all-features`. The MLT conformance tests
are `#[ignore]`d and need the specification's fixtures, which Nazca does not
redistribute:

```sh
tests/fixtures/fetch_mlt_fixtures.sh
cargo test --features mvt,mlt -- --ignored
```


## USAGE

### A layout

A view is prepared at the camera's resolution and written as SVG:

```rust
use nazca::svg::SvgWriter;
use nazca::{Camera, MeshBuilder, Palette, Style, wkb};

let mut camera = Camera::new([0.0, 0.0], 1.0, [1200, 900]);   // 8 x 6 in at 150 dpi
camera.fit_bounds(bounds, 40.0);

let mut mesh = MeshBuilder::for_view(&camera);
mesh.set_class(1);
for geom in &wkb_polygons {
    wkb::read(geom, &mut mesh)?;
}

let mut style = Style::default();   // draws nothing until given colours
style.fill = [0.8, 0.85, 0.76, 1.0];
style.line = [0.36, 0.43, 0.33, 1.0];
style.line_width = 1.0;
let mut palette = Palette::new();
palette.set(1, style);

let mut svg = SvgWriter::new(camera.size_px, None);
svg.begin_layer();
svg.draw(&mesh.finish(), &camera, &palette);
std::fs::write("map.svg", svg.finish())?;
```

`examples/layout.rs` does the same on a page with two layers: an MLT
basemap tile under Massachusetts counties.

    cargo run --features mlt --example layout -- out.svg --bearing 20

### An interactive map

Each frame, a `TileSelector` names the tiles the `Camera` needs, and a
`TileCache` splits them into the tiles to draw and the ones to fetch, standing
in for a missing tile with its cached ancestors or children. The embedder
fetches each tile, prepares it with `MeshBuilder::for_tile`, and puts the
mesh in the cache. The `gl` backend draws the plan over the embedder's GL
context and picks the feature under the cursor. To draw chosen features
differently, `Renderer::state` uploads a byte of flags per feature, and
`Renderer::draw_state` draws only those holding a bit of a mask.
`Picker::draw_state` and `SvgWriter::draw_state` follow the same flags, so a
layer drawn only through a state, as a filter, picks and exports only what
is on screen.

The embedder owns the GL context and wraps it in a `gl::Context`, which
every call takes:

```rust
use nazca::gl::{Context, Renderer};
use nazca::{TileKey, TileSelector};

// Natively through the context's loader (glutin's here); in a browser,
// Context::from_webgl2(webgl2_context).
let gl = unsafe { Context::from_loader(|name| display.get_proc_address(name)) };
let mut renderer = Renderer::new(&gl)?;
let gpu_palette = renderer.palette(&gl, &palette)?;

// Each frame:
renderer.begin_frame(&gl, &camera, background);
cache.begin_frame();
needed.clear();
grid.select(&camera, &mut needed);
cache.plan(SOURCE, &grid, &needed, &mut plan);
// ... fetch plan.fetch, prepare each with MeshBuilder::for_tile, upload
// it with renderer.upload and cache.insert it ...
renderer.begin_layer(&gl);
for &tile in &plan.draw {
    let mesh = cache.get(TileKey { source: SOURCE, tile }).unwrap();
    renderer.draw(&gl, mesh, &camera, &gpu_palette);
}
```

The native spike draws the Massachusetts counties in a window, and a click
picks one and selects it. F shows every other county through a state, and
`--svg out.svg` writes the view as an SVG:

    cargo run --features gl,symbols --example spike
    cargo run --features gl,symbols --example spike -- --screenshot out.png
    cargo run --features gl,symbols --example spike -- --filter --svg out.svg

`spike-web/` is the same in a browser:

    cd spike-web && wasm-pack build --target web --release
    python3 -m http.server   # then open http://localhost:8000/

A browser binding belongs in its own crate, as in `spike-web/`, rather than
in Nazca.

### Symbols

A style class can draw with images: a **marker** in place of a point's dot,
a **dash** pattern on its lines and outlines, and a **pattern** repeated over
its fill in place of the flat colour. A `Symbol` is an RGBA image, and a
`Symbols` packs them into one atlas, each named by a `SymbolId`. A symbol is
one of two kinds. A colour symbol is drawn as it is, its alpha multiplied by
the class's fill alpha. A mask uses only its alpha, tinted by the class's
fill colour, so one pin or one hatch serves every colour a palette gives it.

```rust
use nazca::symbol::Kind;
use nazca::{Dash, Detail, Marker, Palette, Pattern, Style, Symbol, Symbols};

let mut symbols = Symbols::new();
// A pin 24 pixels tall, rasterized at twice that for a pixel ratio of 2
// (feature `symbols`), its tip on the point.
let mut pin = Symbol::from_svg(PIN_SVG, 24.0, 2.0, Kind::Mask)?;
pin.anchor = [0.5, 1.0];
let pin = symbols.add(pin)?;
let hatch = symbols.add(hatch)?;   // a Symbol::new of the embedder's pixels

let mut style = Style::default();
style.fill = [0.13, 0.4, 0.85, 1.0];
style.line = [0.2, 0.2, 0.2, 1.0];
style.line_width = 1.5;
style.point_size = 24.0;   // a marker's longest side
style.marker = Some(Marker::new(pin));
style.dash = Dash::new(&[6.0, 4.0], 0.0);
style.pattern = Some(Pattern::new(hatch, 16.0));
let mut palette = Palette::new();
palette.set(1, style);
palette.check_reach(&symbols, Detail::default().guard_px)?;

renderer.set_symbols(&gl, &symbols)?;   // the GPU's atlas
svg.symbols(&symbols);                  // or a layout's
```

-   **Sizes.** A symbol has no size of its own; the style gives it one, in
    pixels. Rasterize it at the device's pixel ratio or more and say so in
    its `scale`. Drawn larger than its pixels divided by its scale it
    blurs, and `Palette::blurred` names the classes that do; drawn smaller
    than a quarter of that it can shimmer. Either way, add it again at
    another size.
-   **The guard band.** A tile draws a marker cut at its edge, and the
    neighbour draws the rest from its own copy of the point, so a marker
    must reach no further than half the tiles' guard band.
    `Palette::check_reach` says whether every marker does, and what guard
    band one needs.
-   **Dashes** are in pixels and have round ends, as lines do, so each dash
    is drawn a line width longer than its on length and each gap that much
    shorter. An on length of 0 is a dot. A line handed in as pieces starts
    each piece where the ones before end, with `MeshBuilder::set_line_start`
    of their `MeshBuilder::line_length`s summed, so the pattern runs on
    across the seams.
-   **From SVG.** With the `symbols` feature, `Symbol::from_svg` rasterizes
    any static SVG with `resvg`, but its text, which needs fonts. It adds
    about 1 MB to a WebAssembly build, under 400 KB gzipped. A browser
    embedder can instead draw its icons to a canvas and hand in the pixels
    with `Symbol::new`, at no cost, though a layout then writes them as
    images.
-   **In a layout**, each symbol used is written once: as its markup if
    `Symbol::from_svg` made it, cleaned of scripts, styles and external
    references and its ids made unique, or else as a PNG. Markers are
    `<use>`s of it, patterns `<pattern>`s, and dashes `stroke-dasharray`.

### Vector tiles

The `mvt` and `mlt` features decode basemap tiles straight into a `Sink`:

```rust
for layer in nazca::mlt::layers(&tile_bytes) {
    let features = layer?.decode()?;
    for i in 0..features.len() {
        features.read(i, tile_bounds, &mut mesh)?;
    }
}
```


## SUPPORTED

| Input | Feature | Notes |
|---|---|---|
| WKB | default | Streamed into a `Sink`; any `Sink` adapter takes other sources |
| MVT | `mvt` | Mapbox Vector Tile, geometry and ids |
| MLT | `mlt` | MapLibre Tile v1, geometry and ids; passes all 366 conformance fixtures |
| Rasters | default | R8 categorical, R32F continuous with NaN or nodata, RGBA |
| Symbols | default | RGBA images, as markers, dashes' patterns and pattern fills |
| SVG symbols | `symbols` | Any static SVG but its text, rasterized by `resvg` |

| Backend | Feature | Draws |
|---|---|---|
| GPU | `gl` | WebGL2 in a browser, OpenGL 3.3 core natively, through `glow` |
| SVG | default | Layouts, one path per style class per layer |

**Preparation.** Each tile's geometry is snapped to a grid finer than its
pixels, decimated, clipped to the tile plus a guard band, and triangulated
with a port of earcut. Neighbouring tiles snap to the same grid, so shared
borders agree point for point. Point features keep their exact position.

**GPU backend.** Fills are coloured through a palette held as a lookup
texture, so restyling uploads one small texture. Lines are expanded in the
vertex shader, antialiased, with round caps and joins at every zoom. Rasters
are coloured by a colormap texture, unwarped or stretched over a
caller-projected control grid. Each pixel belongs to one tile, so tiles that
overlap never blend twice. Picking draws feature ids into a one-pixel integer
target. Feature state redraws only the features a small texture of flags
marks, so a selection or a hover rewrites that texture and no geometry; a
pick can follow the same flags, so a filtered layer picks only what it shows.
Markers, dash patterns and pattern fills come from one mipmapped atlas and
the palette texture, so restyling them uploads no image and no geometry. A
marker picks where its image is at least half opaque; dashes run on across
tile edges without crawling as the map pans; a pattern is anchored to the
map, seamless across tiles at every zoom.

**SVG backend.** Fills come from the polygon rings rather than the triangles,
and lines are exactly the segments the GPU strokes. A layer's fills and lines
are gathered into one path per class, so a polygon split along tile edges
shows no seams in an antialiasing viewer. A border two polygons share is
drawn once, on screen and in SVG. Feature state works as on the GPU, from
the same bytes of flags.

**Camera.** Pan, zoom and rotate about a point, fit to bounds, and convert
between screen and map. Matrices are composed in `f64`, so a tile's offset
cancels before the cast to `f32`.

**Styles.** Zoom-dependent values interpolate the way MapLibre's zoom curves
do (`style::interpolate`), on MapLibre's zoom (`TileGrid::fractional_level`).


## LIMITATIONS

-   **No text.** Labels, and their placement and collision, are not drawn yet,
    and an SVG symbol's text is dropped.
-   **Symbols are images.** One drawn much larger or smaller than its pixels
    blurs or shimmers, and is added again at another size. Symbols drawn as
    vector shapes are planned.
-   **No raster output.** The SVG backend does not write rasters, and there is
    no CPU rasterizer for PNG layouts.
-   **Flat maps only.** A globe is planned; v1 is designed so it is not
    foreclosed, but there is no globe camera or selector.
-   **No reprojection.** Coordinates must arrive in the map's CRS.
-   **No feature attributes.** Nazca holds geometry and a feature id. A pick
    returns the id, and the embedder looks up the rest.
-   **MVT and MLT properties are skipped.** Only geometry and ids are decoded.
-   **SVG fills by class.** Two features of one class that overlap fill as
    their union, where the GPU blends a translucent fill twice, and a layer's
    fills stack by class rather than in feature order.
-   **The API will change.** Nazca is pre-1.0.

Design notes are in [plans/rendering.org](plans/rendering.org), the scope in
[project.org](project.org), and the open items in
[plans/todo.org](plans/todo.org).


## IMPLEMENTATION

-   **`sink.rs`:** the `Sink` trait every geometry source drives.
-   **`wkb.rs`:** the streaming WKB reader.
-   **`grid.rs`:** tile grids (Web Mercator and custom quadtrees), tile ids
    and bounds.
-   **`prepare.rs`:** snapping, decimation, clipping and tessellation into a
    `Mesh`.
-   **`earcut.rs`:** a port of mapbox/earcut.
-   **`camera.rs`, `select.rs`:** the camera and the `TileSelector` trait.
-   **`cache.rs`:** the tile cache and each frame's draw and fetch plan.
-   **`style.rs`:** palettes with their markers, dashes and patterns,
    colormaps and zoom curves.
-   **`symbol.rs`:** symbols and their atlas, and `Symbol::from_svg`
    (feature `symbols`).
-   **`raster.rs`:** raster bands and the meshes they are drawn over.
-   **`gl.rs`:** the GPU backend (feature `gl`), the only module allowed
    `unsafe`.
-   **`svg.rs`:** the SVG backend.
-   **`mvt.rs`, `mlt.rs`:** the vector tile decoders (features `mvt`, `mlt`).


## PRIOR ART

-   [MapLibre GL JS and Native](https://maplibre.org/) draw vector and raster
    tiles on the GPU from a style document. Nazca's zoom curves and Web
    Mercator zoom follow MapLibre's.
-   [Mapnik](https://mapnik.org/) renders maps on the CPU for tile servers
    and print.

The difference is scope. MapLibre owns the fetching, the style document and
the DOM, and Mapnik is a C++ library with many dependencies. Nazca draws
both interactive maps and layouts from one pipeline, leaves I/O and events to
its embedder, and builds for WebAssembly with nothing but pure-Rust crates.


## LICENSE

Released under the [MIT License](LICENSE). `src/earcut.rs` is a port of
[mapbox/earcut](https://github.com/mapbox/earcut), under its ISC licence, and the
FastPFOR decoder in `src/mlt.rs` follows
[maplibre-tile-spec](https://github.com/maplibre/maplibre-tile-spec)'s reference
decoder, under its MIT licence; see [NOTICE](NOTICE).
