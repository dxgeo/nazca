//! What both spikes draw -- the native one (`main.rs`) and the browser one
//! (`spike-web/`, which includes this file): the Massachusetts counties from
//! nazca's WKB fixture, projected to Web Mercator and prepared tile by tile
//! as the camera asks for them, a marker at each county's centre, a hatch
//! or stipple over some of them, and a raster laid over them.

use nazca::grid::{Rect, TileFrame};
use nazca::raster::{Band, Raster, RasterMesh};
use nazca::symbol::Kind;
use nazca::{Align, Colormap, Dash, Detail, Marker, Mesh, MeshBuilder, Palette, Pattern, Sink, Style, Symbol, SymbolId, Symbols, TileGrid, TileId, wkb};

/// Added to a county's index for its marker's feature id, so a pick tells a
/// marker from the county under it.
pub const MARKER_BASE: u32 = 1 << 16;
/// The first of the markers' three classes, after the counties' six.
const MARKER_CLASS: u8 = 7;

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

    pub fn len(&self) -> usize {
        self.counties.len()
    }

    /// The county a picked feature id names, its own or its marker's.
    pub fn name(&self, feature: u32) -> Option<&'static str> {
        self.counties.get((feature % MARKER_BASE) as usize).map(|c| c.name)
    }

    /// The feature id of the county called `name`.
    pub fn find(&self, name: &str) -> Option<u32> {
        self.counties.iter().position(|c| c.name == name).map(|i| i as u32)
    }

    /// The longest ring of the county called `name`, in Web Mercator
    /// metres, closed.
    pub fn longest_ring(&self, name: &str) -> Option<Vec<[f64; 2]>> {
        /// Keeps the longest ring fed to it, by point count.
        #[derive(Default)]
        struct Longest {
            ring: Vec<[f64; 2]>,
            best: Vec<[f64; 2]>,
        }
        impl Sink for Longest {
            fn point(&mut self, _: [f64; 2]) {}
            fn points(&mut self, xy: &[[f64; 2]]) {
                self.ring.extend_from_slice(xy);
            }
            fn begin_ring(&mut self) {
                self.ring.clear();
            }
            fn end_ring(&mut self) {
                if self.ring.len() > self.best.len() {
                    self.best = std::mem::take(&mut self.ring);
                }
            }
        }
        let c = self.counties.iter().find(|c| c.name == name)?;
        let mut longest = Longest::default();
        wkb::read(&c.wkb, &mut ToWebMercator(&mut longest)).ok()?;
        Some(longest.best)
    }

    /// The centre of county `i`'s bounds, where its marker stands.
    pub fn centre(&self, i: usize) -> [f64; 2] {
        let b = self.counties[i].bounds;
        [(b.min[0] + b.max[0]) / 2.0, (b.min[1] + b.max[1]) / 2.0]
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

    /// Feeds `builder` the counties whose stored bounds reach `reach`, and
    /// each one's marker, a point in one of three classes. A marker's class
    /// draws nothing until [`palette`] gives it a look.
    pub fn prepare_into(&self, builder: &mut MeshBuilder, reach: Rect) {
        for (i, c) in self.counties.iter().enumerate().filter(|(_, c)| c.bounds.intersects(&reach)) {
            builder.set_class(c.class);
            builder.set_feature(i as u32);
            wkb::read(&c.wkb, &mut ToWebMercator(&mut *builder)).unwrap();
            builder.set_class(MARKER_CLASS + (i % 3) as u8);
            builder.set_feature(MARKER_BASE + i as u32);
            builder.point(self.centre(i));
        }
    }
}

/// How the counties' markers look.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Markers {
    None,
    /// Colour symbols: a badge upright on the screen, an arrow turning with
    /// the map to point north, and an arrow pointing right on the screen.
    Colour,
    /// One mask, a pin anchored at its tip, in three classes' colours: two
    /// upright on the screen, one turning with the map and translucent.
    Mask,
    /// As `Mask`, but the second class a badge, and both symbols from SVG
    /// markup ([`nazca::Symbol::from_svg`]): the pin a mask painted in
    /// `currentColor`, the badge in colour with a gradient.
    Markup,
}

impl Markers {
    pub fn next(self) -> Markers {
        match self {
            Markers::None => Markers::Colour,
            Markers::Colour => Markers::Mask,
            Markers::Mask => Markers::Markup,
            Markers::Markup => Markers::None,
        }
    }
}

/// The markers' and patterns' symbols, drawn by [`rasterize`] as an
/// embedder's own rasterizer would hand them in: larger than they are
/// drawn, transparent pixels black. The pattern check's own two are opaque
/// squares of 16: white, and four coloured quarters. Two more come from SVG
/// markup, [`PIN_SVG`] and [`BADGE_SVG`].
pub struct Icons {
    pub symbols: Symbols,
    badge: SymbolId,
    arrow: SymbolId,
    pin: SymbolId,
    pub hatch: SymbolId,
    stipple: SymbolId,
    pub solid: SymbolId,
    pub quarters: SymbolId,
    pin_svg: SymbolId,
    badge_svg: SymbolId,
}

/// The pin as SVG: the same head over a point, its outline the tangents from
/// the tip, the hole cut by `evenodd`.
pub const PIN_SVG: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 48"><path fill="currentColor" fill-rule="evenodd" d="M16 46.5 3.243 22.893A14.5 14.5 0 1 1 28.757 22.893ZM16 10.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 1 0 0-11Z"/></svg>"#;

/// A badge as SVG: a dark ring around a disc shading from white to blue, and
/// a red centre. Its gradient's id is `a`, as an exported icon's would be.
pub const BADGE_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><defs><radialGradient id="a"><stop offset="0.3" stop-color="#fff"/><stop offset="1" stop-color="#6aa0e0"/></radialGradient></defs><circle cx="32" cy="32" r="28" fill="url(#a)" stroke="#282828" stroke-width="4"/><circle cx="32" cy="32" r="9" fill="#c81e1e"/></svg>"##;

/// The pattern check's quarters: top left, top right, bottom left, bottom
/// right.
pub const QUARTERS: [[u8; 3]; 4] = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]];

/// The pin's height in its image, and its tip's distance from the top.
const PIN: (u32, f64) = (48, 46.5);

pub fn icons() -> Icons {
    let mut symbols = Symbols::new();
    // A white disc with a dark ring and a red centre, in a square whose
    // corners are transparent.
    let badge = rasterize(64, 64, |x, y| {
        let r = (x - 32.0).hypot(y - 32.0);
        (r < 30.0).then_some(if r > 26.0 { [40, 40, 40] } else if r < 9.0 { [200, 30, 30] } else { [255, 255, 255] })
    });
    // A kite pointing up, its halves two reds.
    let arrow = rasterize(64, 64, |x, y| {
        let kite = [[32.0, 4.0], [54.0, 60.0], [32.0, 46.0], [10.0, 60.0]];
        inside(&kite, x, y).then_some(if x < 32.0 { [210, 40, 40] } else { [90, 20, 20] })
    });
    // A round head over a point, with a hole through the head.
    let pin = rasterize(32, PIN.0, |x, y| {
        let r = (x - 16.0).hypot(y - 16.0);
        let body = r < 14.5 || inside(&[[2.5, 21.0], [29.5, 21.0], [16.0, PIN.1]], x, y);
        (body && r > 5.5).then_some([255, 255, 255])
    });
    let mut pin = Symbol::new(32, PIN.0, pin).unwrap();
    pin.kind = Kind::Mask;
    pin.anchor = [0.5, (PIN.1 / PIN.0 as f64) as f32];
    // A diagonal line, rising to the right, that meets its neighbours'
    // across every edge, so the repeats tile.
    let hatch = rasterize(32, 32, |x, y| {
        let m = (x + y).rem_euclid(32.0);
        (m.min(32.0 - m) / std::f64::consts::SQRT_2 < 3.0).then_some([255, 255, 255])
    });
    let mut hatch = Symbol::new(32, 32, hatch).unwrap();
    hatch.kind = Kind::Mask;
    // Two dots, a green and a brown, offset as a stipple's are.
    let stipple = rasterize(32, 32, |x, y| {
        let dot = |cx: f64, cy: f64| (x - cx).hypot(y - cy) < 5.0;
        if dot(8.0, 8.0) { Some([40, 110, 60]) } else if dot(24.0, 24.0) { Some([140, 90, 40]) } else { None }
    });
    let solid = rasterize(16, 16, |_, _| Some([255, 255, 255]));
    let quarters = rasterize(16, 16, |x, y| Some(QUARTERS[(x >= 8.0) as usize + 2 * (y >= 8.0) as usize]));
    let mut add = |pixels: Symbol| symbols.add(pixels).unwrap();
    let badge = add(Symbol::new(64, 64, badge).unwrap());
    let arrow = add(Symbol::new(64, 64, arrow).unwrap());
    let pin = add(pin);
    let hatch = add(hatch);
    let stipple = add(Symbol::new(32, 32, stipple).unwrap());
    let solid = add(Symbol::new(16, 16, solid).unwrap());
    let quarters = add(Symbol::new(16, 16, quarters).unwrap());
    // Rasterized as large as the pixel ones: three image pixels a drawn one
    // at the sizes they are drawn.
    let mut pin_svg = Symbol::from_svg(PIN_SVG, PIN_ON_MAP, 3.0, Kind::Mask).unwrap();
    pin_svg.anchor = [0.5, (PIN.1 / PIN.0 as f64) as f32];
    let pin_svg = add(pin_svg);
    let badge_svg = add(Symbol::from_svg(BADGE_SVG, 22.0, 3.0, Kind::Colour).unwrap());
    Icons { symbols, badge, arrow, pin, hatch, stipple, solid, quarters, pin_svg, badge_svg }
}

/// RGBA8 pixels of `w` x `h`, from `paint`'s colour at a position, or none,
/// sampled 4 x 4 a pixel.
fn rasterize(w: u32, h: u32, paint: impl Fn(f64, f64) -> Option<[u8; 3]>) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let (mut sum, mut n) = ([0u32; 3], 0u32);
            for k in 0..16 {
                let at = |v: u32, s: u32| v as f64 + (s as f64 + 0.5) / 4.0;
                if let Some(c) = paint(at(x, k % 4), at(y, k / 4)) {
                    (0..3).for_each(|i| sum[i] += c[i] as u32);
                    n += 1;
                }
            }
            let rgb = if n > 0 { sum.map(|s| (s / n) as u8) } else { [0; 3] };
            pixels.extend_from_slice(&[rgb[0], rgb[1], rgb[2], (n * 255 / 16) as u8]);
        }
    }
    pixels
}

/// Whether (`x`, `y`) lies inside `polygon`, by crossings.
fn inside(polygon: &[[f64; 2]], x: f64, y: f64) -> bool {
    let mut odd = false;
    for (i, a) in polygon.iter().enumerate() {
        let b = polygon[(i + 1) % polygon.len()];
        if (a[1] > y) != (b[1] > y) && x < a[0] + (y - a[1]) / (b[1] - a[1]) * (b[0] - a[0]) {
            odd = !odd;
        }
    }
    odd
}

/// A marker of `symbol`, its longest side `size`. A colour symbol's alpha is
/// multiplied by the fill's, so it is opaque white.
fn marker(symbol: SymbolId, size: f32, align: Align, rotation: f32, fill: [f32; 4]) -> Style {
    let mut marker = Marker::new(symbol);
    marker.align = align;
    marker.rotation = rotation;
    let mut style = Style::default();
    style.fill = fill;
    style.point_size = size;
    style.marker = Some(marker);
    style
}

/// How the counties are outlined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Outlines {
    Solid,
    /// Dashes 6 pixels on, 4 off.
    Dashed,
    /// Dots: on lengths of 0, round, 5 pixels apart.
    Dotted,
    /// A wide translucent line, which shows a border drawn twice darker.
    Translucent,
}

impl Outlines {
    pub fn next(self) -> Outlines {
        match self {
            Outlines::Solid => Outlines::Dashed,
            Outlines::Dashed => Outlines::Dotted,
            Outlines::Dotted => Outlines::Translucent,
            Outlines::Translucent => Outlines::Solid,
        }
    }

    /// The counties' line colour, width and dash.
    fn look(self) -> ([f32; 4], f32, Option<Dash>) {
        const INK: [f32; 4] = [0.25, 0.27, 0.30, 1.0];
        match self {
            Outlines::Solid => (INK, 1.25, None),
            Outlines::Dashed => (INK, 1.5, Dash::new(&[6.0, 4.0], 0.0)),
            Outlines::Dotted => (INK, 2.5, Dash::new(&[0.0, 5.0], 0.0)),
            Outlines::Translucent => ([0.1, 0.2, 0.7, 0.5], 4.0, None),
        }
    }

    /// The dash a chosen county's outline takes: one when the counties
    /// have one.
    pub fn highlight_dash(self) -> Option<Dash> {
        match self {
            Outlines::Dashed | Outlines::Dotted => Dash::new(&[10.0, 6.0], 0.0),
            Outlines::Solid | Outlines::Translucent => None,
        }
    }
}

/// What lies over the counties, a layer of its own drawn from their tiles
/// again: a pattern over the flat fills beneath, in a palette of its own.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Patterns {
    None,
    /// A mask hatch, a dark translucent blue, over every other county.
    Hatch,
    /// A colour stipple over the others.
    Stipple,
}

impl Patterns {
    pub fn next(self) -> Patterns {
        match self {
            Patterns::None => Patterns::Hatch,
            Patterns::Hatch => Patterns::Stipple,
            Patterns::Stipple => Patterns::None,
        }
    }
}

/// The patterns' layer: a hatch or a stipple in some of the counties'
/// classes, and nothing in the rest.
pub fn hatching(patterns: Patterns, icons: &Icons) -> Palette {
    let mut p = Palette::new();
    let (symbol, repeat, fill, classes) = match patterns {
        Patterns::None => return p,
        Patterns::Hatch => (icons.hatch, 12.0, [0.1, 0.2, 0.5, 0.6], [1, 3, 5]),
        Patterns::Stipple => (icons.stipple, 16.0, [1.0; 4], [2, 4, 6]),
    };
    for class in classes {
        p.set(class, pattern(symbol, repeat, fill));
    }
    p
}

/// A fill of `symbol` repeated every `repeat` pixels, tinted or faded by
/// `fill`.
pub fn pattern(symbol: SymbolId, repeat: f32, fill: [f32; 4]) -> Style {
    let mut style = Style::default();
    style.fill = fill;
    style.pattern = Some(Pattern::new(symbol, repeat));
    style
}

/// The pin's sizes. Checked against half of the default 32-pixel guard band
/// ([`nazca::Palette::check_reach`]): against the map the pin reaches its
/// tip's depth, 0.97 of its size, so 16 fits; upright on the screen it can
/// turn against the tiles, and reaches to its top corners, so 16 does not
/// and 15 does.
pub const PIN_ON_MAP: f32 = 16.0;
pub const PIN_ON_SCREEN: f32 = 15.0;

/// Six muted fills for neighbouring counties, outlined, and their markers.
pub fn palette(markers: Markers, outlines: Outlines, icons: &Icons) -> Palette {
    const FILLS: [[f32; 4]; 6] = [
        [0.80, 0.86, 0.75, 1.0],
        [0.93, 0.85, 0.70, 1.0],
        [0.76, 0.84, 0.90, 1.0],
        [0.90, 0.78, 0.78, 1.0],
        [0.84, 0.80, 0.90, 1.0],
        [0.95, 0.92, 0.76, 1.0],
    ];
    let mut p = Palette::new();
    let (line, width, dash) = outlines.look();
    for (i, fill) in FILLS.into_iter().enumerate() {
        let mut style = outlined(fill, line, width);
        style.dash = dash;
        p.set(1 + i as u8, style);
    }
    const WHITE: [f32; 4] = [1.0; 4];
    let looks = match markers {
        Markers::None => return p,
        // A centred square reaches half its diagonal when it turns: 22.
        Markers::Colour => [
            marker(icons.badge, 22.0, Align::Screen, 0.0, WHITE),
            marker(icons.arrow, 32.0, Align::Map, 0.0, WHITE),
            marker(icons.arrow, 22.0, Align::Screen, 90.0, WHITE),
        ],
        Markers::Mask => [
            marker(icons.pin, PIN_ON_SCREEN, Align::Screen, 0.0, [0.13, 0.4, 0.85, 1.0]),
            marker(icons.pin, PIN_ON_SCREEN, Align::Screen, 0.0, [0.1, 0.6, 0.3, 1.0]),
            marker(icons.pin, PIN_ON_MAP, Align::Map, 0.0, [0.55, 0.2, 0.7, 0.75]),
        ],
        Markers::Markup => [
            marker(icons.pin_svg, PIN_ON_SCREEN, Align::Screen, 0.0, [0.13, 0.4, 0.85, 1.0]),
            marker(icons.badge_svg, 22.0, Align::Screen, 0.0, WHITE),
            marker(icons.pin_svg, PIN_ON_MAP, Align::Map, 0.0, [0.55, 0.2, 0.7, 0.75]),
        ],
    };
    for (k, look) in looks.into_iter().enumerate() {
        p.set(MARKER_CLASS + k as u8, look);
    }
    p
}

/// The marker cost measurement's looks for class 1 at `size`: a dot, the
/// colour badge, and the mask pin.
pub fn grid_looks(icons: &Icons, size: f32) -> [(&'static str, Palette); 3] {
    let mut dot = Style::default();
    dot.fill = [0.8, 0.1, 0.1, 1.0];
    dot.point_size = size;
    let looks = [
        ("dots", dot),
        ("colour markers", marker(icons.badge, size, Align::Screen, 0.0, [1.0; 4])),
        ("mask markers", marker(icons.pin, size, Align::Screen, 0.0, [0.13, 0.4, 0.85, 1.0])),
    ];
    looks.map(|(name, style)| {
        let mut p = Palette::new();
        p.set(1, style);
        (name, p)
    })
}

/// A palette whose pin is upright on the screen at [`PIN_ON_MAP`]'s size,
/// which [`nazca::Palette::check_reach`] refuses.
pub fn too_far(icons: &Icons) -> Palette {
    let mut p = Palette::new();
    p.set(MARKER_CLASS, marker(icons.pin, PIN_ON_MAP, Align::Screen, 0.0, [1.0; 4]));
    p
}

/// A fill with an outline, and no points.
fn outlined(fill: [f32; 4], line: [f32; 4], line_width: f32) -> Style {
    let mut style = Style::default();
    style.fill = fill;
    style.line = line;
    style.line_width = line_width;
    style
}

/// The look of a chosen county, in every class the counties use.
pub fn highlight(fill: [f32; 4], line: [f32; 4], line_width: f32, dash: Option<Dash>) -> Palette {
    let mut p = Palette::new();
    for class in 1..=6 {
        let mut style = outlined(fill, line, line_width);
        style.dash = dash;
        p.set(class, style);
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
