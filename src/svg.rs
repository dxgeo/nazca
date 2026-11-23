//! The SVG backend, for vector layouts (`../plans/rendering.org` §
//! BACKENDS). Default build, no dependency.
//!
//! It replays the same prepared [`Mesh`] the `gl` backend uploads, styled by
//! the same [`Palette`], placed by the same [`Camera`]:
//!
//! - *Fills* are the mesh's polygon rings (`Mesh::rings`, `Mesh::polygons`),
//!   not the triangles, which would show their seams once antialiased and
//!   cost more.
//! - *Lines* are the mesh's outline segments, chained back into polylines.
//!   These are exactly the segments the GPU draws, so edges that clipping
//!   made are not stroked here either, and lines draw over every fill of
//!   their layer, as on screen. Each is cut to the tile's own area, half
//!   open as points are, so a line in a neighbour's guard band too is
//!   written once; a dash cut at a tile edge is two capped halves whose
//!   union looks whole. The earlier copy of a shared border is left out
//!   where its partner is written, as on screen ([`Mesh::partners`]).
//! - *Dashes* are `stroke-dasharray`, with the dash's phase as the class's
//!   `stroke-dashoffset`. SVG starts the pattern again at each subpath, so a
//!   chain that starts part-way through it, by its distance along its line,
//!   is a `<path>` of its own with its own offset; only chains that start at
//!   the pattern's start share the class's one path.
//! - *Points* are `<circle>`s of the class's point size, in its fill colour,
//!   or, where the class has a [`crate::Marker`], its symbol: a `<use>` of
//!   it placed, sized and turned as on screen, or for a mask a `<rect>` of
//!   the class's fill through a `<mask>` of it. Each symbol a marker uses is
//!   written once, in `<defs>`, from the [`Symbols`] given to
//!   [`SvgWriter::symbols`]: as a PNG of its pixels, stored uncompressed by
//!   a small encoder here. A mask's PNG is white where it is opaque, so a
//!   viewer reading the mask by luminance, without `mask-type: alpha`, sees
//!   the same mask. A point is written by the tile whose own area holds it,
//!   so a point in a neighbour's guard band too is written once, not once
//!   from each.
//!
//! *A layer is one `<path>` per class for its fills and one for its lines*,
//! gathered over every [`SvgWriter::draw`] of the layer and written when it
//! ends. A viewer antialiases each path's edges on its own, so a polygon
//! split along tile edges into separate paths shows the background through
//! every split: about half a pixel's coverage from each side, which at 0.9
//! opacity composites to 0.7. Inside one path the split is interior and
//! leaves no trace. The same holds for lines: where two tiles' pieces of a
//! line meet, their round ends overlap, and a translucent stroke would blend
//! twice there in two paths but not in one.
//!
//! Fills use `fill-rule: nonzero`, with every polygon's exterior wound one
//! way and its holes the other, so a hole stays empty while pieces of
//! neighbouring features neither cancel (as `evenodd` would make them) nor
//! blend twice. The cost: two features of one class that overlap are
//! filled as their union, where the GPU blends a translucent fill twice
//! over the overlap; and a layer's fills are stacked by class rather than in
//! feature order.
//!
//! *Feature state* works as on the GPU: [`SvgWriter::draw_state`] writes only
//! the features whose flags share a bit with a mask, taking the same bytes
//! [`crate::gl::Renderer::state`] does. So a layer drawn only through a state
//! writes only what the screen shows, and a state pass, such as a highlight
//! in a layer of its own, writes over the layer as it draws on screen. A
//! feature is written whole or not at all: each of its primitives carries its
//! id.
//!
//! Styles are written once each, as CSS classes. Rasters are not written: an
//! SVG image needs encoded image bytes, which is the `raster` backend's
//! business.

use crate::camera::Camera;
use crate::prepare::Mesh;
use crate::style::{Align, Palette, Style};
use crate::symbol::{Kind, Symbol, Symbols};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

/// Builds one SVG document. See the module docs.
pub struct SvgWriter {
    size: [u32; 2],
    background: Option<[f32; 4]>,
    body: String,
    /// CSS rules by class name.
    styles: BTreeMap<String, String>,
    layer: u32,
    /// The current layer's fill and line path data by style class, and its
    /// point elements: written to `body` when the layer ends.
    fills: BTreeMap<u8, String>,
    lines: BTreeMap<u8, String>,
    /// Dashed chains that start part-way through their pattern, by class:
    /// each one's `stroke-dashoffset` and path data.
    runs: BTreeMap<u8, Vec<(f64, String)>>,
    points: String,
    /// The symbols markers draw, by id, and the ids a marker has used.
    symbols: BTreeMap<u16, Symbol>,
    used: BTreeSet<u16>,
}

impl SvgWriter {
    /// A document `size_px` across, in the pixels the camera that draws into
    /// it uses, over `background` (none: transparent).
    pub fn new(size_px: [u32; 2], background: Option<[f32; 4]>) -> SvgWriter {
        SvgWriter {
            size: size_px,
            background,
            body: String::new(),
            styles: BTreeMap::new(),
            layer: 0,
            fills: BTreeMap::new(),
            lines: BTreeMap::new(),
            runs: BTreeMap::new(),
            points: String::new(),
            symbols: BTreeMap::new(),
            used: BTreeSet::new(),
        }
    }

    /// The symbols markers draw from, in place of any given before. Each one
    /// a marker uses is written once, in `<defs>`. Give them before drawing
    /// the layers that use them; a marker whose symbol isn't here draws
    /// nothing, as on screen.
    pub fn symbols(&mut self, symbols: &Symbols) {
        self.symbols = symbols.iter().map(|(id, s, _)| (id.0, s.clone())).collect();
    }

    /// Starts a layer: what follows draws over what came before, and may use
    /// a different palette under the same class numbers.
    pub fn begin_layer(&mut self) {
        self.end_layer();
        self.layer += 1;
    }

    /// Writes the current layer: its fills, then its lines, then its points.
    fn end_layer(&mut self) {
        let layer = self.layer;
        for (class, d) in std::mem::take(&mut self.fills) {
            let _ = writeln!(self.body, r#"<path class="f{layer}-{class}" d="{d}"/>"#);
        }
        let mut runs = std::mem::take(&mut self.runs);
        for (class, d) in std::mem::take(&mut self.lines) {
            if !d.is_empty() {
                let _ = writeln!(self.body, r#"<path class="l{layer}-{class}" d="{d}"/>"#);
            }
            for (offset, d) in runs.remove(&class).unwrap_or_default() {
                let _ = writeln!(self.body, r#"<path class="l{layer}-{class}" style="stroke-dashoffset:{}" d="{d}"/>"#, num(offset));
            }
        }
        self.body.push_str(&std::mem::take(&mut self.points));
    }

    /// Draws `mesh` where `camera` puts it, styled by `palette`, into the
    /// current layer's paths (see the module docs).
    pub fn draw(&mut self, mesh: &Mesh, camera: &Camera, palette: &Palette) {
        self.draw_shown(mesh, camera, palette, |_| true);
    }

    /// Draws only the features of `mesh` whose byte in `flags` shares a bit
    /// with `mask`, as [`SvgWriter::draw`] draws. `flags` holds the features
    /// with ids from `first` on, wrapping past the top of the range, as
    /// [`crate::gl::Renderer::state`] takes them; a feature outside the span
    /// holds no state.
    pub fn draw_state(&mut self, mesh: &Mesh, camera: &Camera, palette: &Palette, first: u32, flags: &[u8], mask: u8) {
        self.draw_shown(mesh, camera, palette, |id| flags.get(id.wrapping_sub(first) as usize).is_some_and(|f| f & mask != 0));
    }

    /// Draws the features of `mesh` for whose id `shown` holds: a polygon by
    /// its first ring's first vertex, a segment by its first, a point by its
    /// own.
    fn draw_shown(&mut self, mesh: &Mesh, camera: &Camera, palette: &Palette, shown: impl Fn(u32) -> bool) {
        let shown = |i: u32| shown(mesh.ids[i as usize]);
        let screen = |i: u32| {
            let v = &mesh.vertices[i as usize * 3..i as usize * 3 + 2];
            camera.map_to_screen(mesh.frame.to_map([v[0], v[1]]))
        };
        let class_of = |i: u32| mesh.vertices[i as usize * 3 + 2] as u8;
        let layer = self.layer;
        let mut ring = Vec::new();

        for p in &mesh.polygons {
            let rings = &mesh.rings[p[0] as usize..p[1] as usize];
            let Some(first) = rings.first() else { continue };
            if !shown(first[0]) {
                continue;
            }
            let class = class_of(first[0]);
            let style = palette.get(class);
            if style.fill[3] <= 0.0 {
                continue;
            }
            let name = format!("f{layer}-{class}");
            self.styles.entry(name).or_insert_with(|| format!("fill:{};stroke:none;fill-rule:nonzero", paint(style.fill)));
            let d = self.fills.entry(class).or_default();
            for (k, r) in rings.iter().enumerate() {
                ring.clear();
                ring.extend((r[0]..r[1]).map(screen));
                // The exterior turns clockwise on screen (Y down: a positive
                // shoelace sum), holes the other way.
                if (shoelace(&ring) > 0.0) != (k == 0) {
                    ring.reverse();
                }
                for (j, [x, y]) in ring.iter().enumerate() {
                    d.push(if j == 0 { 'M' } else { ' ' });
                    let _ = write!(d, "{},{}", num(*x), num(*y));
                }
                d.push('Z');
            }
        }

        // Outline segments chained into polylines (see the module docs).
        let draws = |style: &Style| style.line_width > 0.0 && style.line[3] > 0.0;
        let local = |i: u32| {
            let v = &mesh.vertices[i as usize * 3..i as usize * 3 + 2];
            [v[0] as f64, v[1] as f64]
        };
        let to_screen = |p: [f64; 2]| {
            let f = &mesh.frame;
            camera.map_to_screen([f.center[0] + p[0] * f.half_span, f.center[1] + p[1] * f.half_span])
        };
        let unit_px = mesh.frame.half_span / camera.units_per_px;
        let mut chain: Option<Chain> = None;
        for (k, s) in mesh.outlines.chunks_exact(2).enumerate() {
            let (a, b) = (s[0], s[1]);
            let class = class_of(a);
            if !draws(palette.get(class)) || !shown(a) {
                continue;
            }
            if let Some(p) = mesh.partner(k) {
                let pa = mesh.outlines[p * 2];
                if shown(pa) && draws(palette.get(class_of(pa))) {
                    continue;
                }
            }
            let (la, lb) = (local(a), local(b));
            // Along the tile's far edges, the neighbour's to write.
            if (la[0] == 1.0 && lb[0] == 1.0) || (la[1] == 1.0 && lb[1] == 1.0) {
                continue;
            }
            let Some((p, q)) = clip_to_own_area(la, lb) else { continue };
            self.lines.entry(class).or_default();
            if !matches!(&chain, Some(c) if c.class == class && p == la && c.last == Some(a)) {
                if let Some(c) = chain.take() {
                    c.finish(palette, &mut self.lines, &mut self.runs);
                }
                let along = mesh.distance(a) + (p[0] - la[0]).hypot(p[1] - la[1]);
                let [x, y] = to_screen(p);
                chain = Some(Chain { class, first: (p == la).then_some(a), last: None, along_px: along * unit_px, d: format!("M{},{}", num(x), num(y)) });
            }
            let c = chain.as_mut().unwrap();
            let [x, y] = to_screen(q);
            let _ = write!(c.d, " {},{}", num(x), num(y));
            c.last = (q == lb).then_some(b);
        }
        if let Some(c) = chain.take() {
            c.finish(palette, &mut self.lines, &mut self.runs);
        }
        for &class in self.lines.keys() {
            let style = palette.get(class);
            self.styles.entry(format!("l{layer}-{class}")).or_insert_with(|| {
                let mut rule = format!("fill:none;stroke:{};stroke-width:{};stroke-linecap:round;stroke-linejoin:round", paint(style.line), num(style.line_width as f64));
                if let Some(dash) = style.dash {
                    let lengths: Vec<String> = dash.lengths().iter().map(|&l| num(l as f64)).collect();
                    let _ = write!(rule, ";stroke-dasharray:{}", lengths.join(","));
                    if dash.phase != 0.0 {
                        let _ = write!(rule, ";stroke-dashoffset:{}", num(dash.phase as f64));
                    }
                }
                rule
            });
        }

        for &i in &mesh.points {
            let class = class_of(i);
            let style = palette.get(class);
            // A point in the tile's own area, half open so a point on an
            // edge belongs to one tile.
            let local = &mesh.vertices[i as usize * 3..i as usize * 3 + 2];
            let own = local.iter().all(|&v| (-1.0..1.0).contains(&v));
            if style.point_size <= 0.0 || style.fill[3] <= 0.0 || !own || !shown(i) {
                continue;
            }
            let [x, y] = screen(i);
            let Some(marker) = style.marker else {
                let name = format!("p{layer}-{class}");
                self.styles.entry(name.clone()).or_insert_with(|| format!("fill:{};stroke:none", paint(style.fill)));
                let _ = writeln!(self.points, r#"<circle class="{name}" cx="{}" cy="{}" r="{}"/>"#, num(x), num(y), num(style.point_size as f64 / 2.0));
                continue;
            };
            let Some(symbol) = self.symbols.get(&marker.symbol.0) else { continue };
            let id = marker.symbol.0;
            self.used.insert(id);
            let (w, h) = (symbol.width(), symbol.height());
            let mut turn = marker.rotation as f64;
            if marker.align == Align::Map {
                turn -= camera.bearing.to_degrees();
            }
            let mut transform = format!("translate({},{})", num(x), num(y));
            if turn.rem_euclid(360.0) != 0.0 {
                let _ = write!(transform, " rotate({})", num(turn));
            }
            let scale = style.point_size as f64 / w.max(h) as f64;
            let [ax, ay] = symbol.anchor.map(|a| a as f64);
            let _ = write!(transform, " scale({}) translate({},{})", fine(scale), num(-ax * w as f64), num(-ay * h as f64));
            let name = format!("m{layer}-{class}");
            if symbol.kind == Kind::Mask {
                self.styles.entry(name.clone()).or_insert_with(|| format!("fill:{};stroke:none", paint(style.fill)));
                let _ = writeln!(self.points, r#"<rect class="{name}" width="{w}" height="{h}" mask="url(#mask{id})" transform="{transform}"/>"#);
            } else if style.fill[3] < 1.0 {
                self.styles.entry(name.clone()).or_insert_with(|| format!("opacity:{}", num(style.fill[3] as f64)));
                let _ = writeln!(self.points, r##"<use class="{name}" href="#sym{id}" transform="{transform}"/>"##);
            } else {
                let _ = writeln!(self.points, r##"<use href="#sym{id}" transform="{transform}"/>"##);
            }
        }
    }

    /// The finished document.
    pub fn finish(mut self) -> String {
        self.end_layer();
        let [w, h] = self.size;
        let mut out = String::with_capacity(self.body.len() + 512);
        let _ = writeln!(out, r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">"#);
        if !self.styles.is_empty() {
            out.push_str("<style>\n");
            for (name, rule) in &self.styles {
                let _ = writeln!(out, ".{name}{{{rule}}}");
            }
            out.push_str("</style>\n");
        }
        if !self.used.is_empty() {
            out.push_str("<defs>\n");
            for id in &self.used {
                let symbol = &self.symbols[id];
                let mask = symbol.kind == Kind::Mask;
                let (w, h) = (symbol.width(), symbol.height());
                let png = png(w, h, symbol.pixels(), mask);
                let _ = writeln!(out, r#"<image id="sym{id}" width="{w}" height="{h}" href="data:image/png;base64,{}"/>"#, base64(&png));
                if mask {
                    let _ = writeln!(out, r##"<mask id="mask{id}" style="mask-type:alpha"><use href="#sym{id}"/></mask>"##);
                }
            }
            out.push_str("</defs>\n");
        }
        if let Some(bg) = self.background {
            let _ = writeln!(out, r#"<rect width="{w}" height="{h}" style="fill:{}"/>"#, paint(bg));
        }
        out.push_str(&self.body);
        out.push_str("</svg>\n");
        out
    }
}

/// A polyline of outline segments being written: its class, the vertex it
/// began at and the one it reached, unless a tile edge cut them, how far
/// along its line it starts in pixels, and its path data.
struct Chain {
    class: u8,
    first: Option<u32>,
    last: Option<u32>,
    along_px: f64,
    d: String,
}

impl Chain {
    /// Closes the chain if it came back to its start, and adds it to its
    /// class's path, or for a dash that it starts part-way through, to the
    /// runs that take paths of their own.
    fn finish(mut self, palette: &Palette, lines: &mut BTreeMap<u8, String>, runs: &mut BTreeMap<u8, Vec<(f64, String)>>) {
        if self.first.is_some() && self.first == self.last {
            self.d.push('Z');
        }
        if let Some(dash) = palette.get(self.class).dash {
            let sum: f64 = dash.lengths().iter().map(|&l| l as f64).sum();
            let period = if dash.lengths().len() % 2 == 1 { 2.0 * sum } else { sum };
            let into = self.along_px.rem_euclid(period);
            // Within a hundredth of a pixel, as `num` writes it.
            if into >= 0.005 && period - into >= 0.005 {
                let offset = (dash.phase as f64 + into).rem_euclid(period);
                runs.entry(self.class).or_default().push((offset, self.d));
                return;
            }
        }
        lines.get_mut(&self.class).unwrap().push_str(&self.d);
    }
}

/// The part of segment `a`--`b` inside the tile's own square, `|x|, |y| <=
/// 1` in local units (Liang--Barsky), or `None`. Ends inside stay exactly as
/// they were.
fn clip_to_own_area(a: [f64; 2], b: [f64; 2]) -> Option<([f64; 2], [f64; 2])> {
    let d = [b[0] - a[0], b[1] - a[1]];
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for axis in 0..2 {
        for (p, q) in [(-d[axis], a[axis] + 1.0), (d[axis], 1.0 - a[axis])] {
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
    if t0 >= t1 {
        return None;
    }
    let at = |t: f64| [(a[0] + t * d[0]).clamp(-1.0, 1.0), (a[1] + t * d[1]).clamp(-1.0, 1.0)];
    Some((if t0 == 0.0 { a } else { at(t0) }, if t1 == 1.0 { b } else { at(t1) }))
}

/// An RGBA8 PNG of `pixels`, its deflate blocks stored uncompressed, so it
/// needs no compressor. White where `white`, keeping only the alpha.
fn png(w: u32, h: u32, pixels: &[u8], white: bool) -> Vec<u8> {
    let mut raw = Vec::with_capacity(pixels.len() + h as usize);
    for row in pixels.chunks_exact(w as usize * 4) {
        raw.push(0); // no filter
        if white {
            raw.extend(row.chunks_exact(4).flat_map(|p| [255, 255, 255, p[3]]));
        } else {
            raw.extend_from_slice(row);
        }
    }
    // zlib: a header, stored blocks of at most 65,535 bytes, the Adler-32.
    let mut z = vec![0x78, 0x01];
    let blocks = raw.chunks(65535);
    let last = blocks.len().max(1) - 1;
    for (k, block) in raw.chunks(65535).enumerate() {
        let n = block.len() as u16;
        z.push((k == last) as u8);
        z.extend_from_slice(&n.to_le_bytes());
        z.extend_from_slice(&(!n).to_le_bytes());
        z.extend_from_slice(block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &raw {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    z.extend_from_slice(&(b << 16 | a).to_be_bytes());

    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut chunk = |kind: &[u8; 4], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let start = out.len();
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        let crc = crc32(&out[start..]);
        out.extend_from_slice(&crc.to_be_bytes());
    };
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&w.to_be_bytes());
    header.extend_from_slice(&h.to_be_bytes());
    header.extend_from_slice(&[8, 6, 0, 0, 0]); // 8 bits, RGBA, deflate, no filter, no interlace
    chunk(b"IHDR", &header);
    chunk(b"IDAT", &z);
    chunk(b"IEND", &[]);
    out
}

/// The CRC-32 PNG chunks end with.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { 0xedb8_8320 ^ (crc >> 1) } else { crc >> 1 };
        }
    }
    !crc
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for group in bytes.chunks(3) {
        let n = group.iter().enumerate().fold(0u32, |n, (i, &b)| n | (b as u32) << (16 - 8 * i));
        for i in 0..4 {
            out.push(if i <= group.len() { ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

/// Twice a ring's signed area by the shoelace formula: positive when it
/// turns from +X towards +Y.
fn shoelace(ring: &[[f64; 2]]) -> f64 {
    let n = ring.len();
    (0..n).map(|i| {
        let ([x0, y0], [x1, y1]) = (ring[i], ring[(i + 1) % n]);
        x0 * y1 - x1 * y0
    }).sum()
}

/// A coordinate to a hundredth of a pixel, without trailing zeros.
fn num(v: f64) -> String {
    let r = (v * 100.0).round() / 100.0;
    let r = if r == 0.0 { 0.0 } else { r }; // no "-0"
    let mut s = format!("{r:.2}");
    while s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.pop();
    }
    s
}

/// A scale to a millionth, without trailing zeros: a hundredth of a 48-pixel
/// image's scale would move its far edge by half a pixel.
fn fine(v: f64) -> String {
    let s = format!("{v:.6}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0".into() } else { s.into() }
}

/// A straight-alpha RGBA colour as CSS: `#rrggbb`, with an opacity when it
/// is not opaque.
fn paint(c: [f32; 4]) -> String {
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let hex = format!("#{:02x}{:02x}{:02x}", byte(c[0]), byte(c[1]), byte(c[2]));
    if c[3] >= 1.0 { hex } else { format!("{hex};opacity:{}", num(c[3] as f64)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Rect;
    use crate::prepare::MeshBuilder;
    use crate::sink::Sink;
    use crate::style::Style;

    fn camera() -> Camera {
        let mut c = Camera::new([0.0, 0.0], 1.0, [200, 100]);
        c.fit_bounds(Rect { min: [0.0, 0.0], max: [200.0, 100.0] }, 0.0);
        c
    }

    fn palette() -> Palette {
        let mut p = Palette::new();
        p.set(1, Style { fill: [1.0, 0.0, 0.0, 1.0], line: [0.0, 0.0, 1.0, 0.5], line_width: 2.0, point_size: 6.0, ..Style::default() });
        p
    }

    #[test]
    fn a_polygon_with_a_hole_a_line_and_a_point() {
        let cam = camera();
        let mut b = MeshBuilder::for_view(&cam);
        b.set_class(1);
        b.begin_polygon();
        for r in [[[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0], [10.0, 10.0]], [[40.0, 40.0], [40.0, 60.0], [60.0, 60.0], [60.0, 40.0], [40.0, 40.0]]] {
            b.begin_ring();
            b.points(&r);
            b.end_ring();
        }
        b.end_polygon();
        b.begin_line();
        b.points(&[[120.0, 20.0], [180.0, 20.0], [180.0, 80.0]]);
        b.end_line();
        b.point([150.0, 50.0]);
        let mesh = b.finish();
        let mut svg = SvgWriter::new(cam.size_px, Some([1.0, 1.0, 1.0, 1.0]));
        svg.begin_layer();
        svg.draw(&mesh, &cam, &palette());
        let out = svg.finish();
        assert!(out.starts_with(r#"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100""#));
        // Map Y up, screen Y down: map (10, 10) is screen (10, 90). The
        // exterior is turned clockwise on screen, the hole the other way.
        assert!(out.contains(r#"<path class="f1-1" d="M10,10 90,10 90,90 10,90Z"#), "{out}");
        assert!(out.contains("10,90ZM60,60 60,40 40,40 40,60Z"), "{out}");
        assert!(out.contains(".f1-1{fill:#ff0000;stroke:none;fill-rule:nonzero}"), "{out}");
        // Both rings' outlines close; the line stays open.
        assert!(out.contains(r#"<path class="l1-1" d="M10,90 90,90 90,10 10,10 10,90Z"#), "{out}");
        assert!(out.contains("M120,80 180,80 180,20"), "{out}");
        assert!(out.contains(".l1-1{fill:none;stroke:#0000ff;opacity:0.5;stroke-width:2;"), "{out}");
        assert!(out.contains(r#"<circle class="p1-1" cx="150" cy="50" r="3"/>"#), "{out}");
        assert!(out.ends_with("</svg>\n"));
    }

    /// One square fed to two tiles that split it down the middle, each
    /// clipped exactly to its own square.
    fn split_square() -> (Camera, [Mesh; 2]) {
        use crate::grid::TileFrame;
        use crate::prepare::Detail;
        let cam = camera();
        let tile = |cx: f64| {
            let frame = TileFrame { center: [cx, 50.0], half_span: 50.0 };
            let mut b = MeshBuilder::new(frame, Detail { tile_px: 100.0, grid_px: 0.25, guard_px: 0.0 });
            b.set_class(1);
            b.begin_polygon();
            b.begin_ring();
            b.points(&[[20.0, 20.0], [180.0, 20.0], [180.0, 80.0], [20.0, 80.0], [20.0, 20.0]]);
            b.end_ring();
            b.end_polygon();
            b.begin_line();
            b.points(&[[20.0, 50.0], [180.0, 50.0]]);
            b.end_line();
            b.finish()
        };
        (cam, [tile(50.0), tile(150.0)])
    }

    #[test]
    fn a_layer_drawn_tile_by_tile_is_one_path_per_class() {
        let (cam, tiles) = split_square();
        let mut svg = SvgWriter::new(cam.size_px, None);
        svg.begin_layer();
        for t in &tiles {
            svg.draw(t, &cam, &palette());
        }
        svg.begin_layer();
        svg.draw(&tiles[0], &cam, &palette());
        let out = svg.finish();
        // Both halves in one fill path, meeting exactly at x = 100.
        assert!(out.contains(r#"<path class="f1-1" d="M20,20 100,20 100,80 20,80ZM100,20 180,20 180,80 100,80Z"/>"#), "{out}");
        assert_eq!(out.matches(r#"class="f1-1""#).count(), 1, "{out}");
        assert_eq!(out.matches(r#"class="l1-1""#).count(), 1, "{out}");
        // Layer 1 is written whole before layer 2 starts.
        let (l1, f2) = (out.find(r#"class="l1-1""#).unwrap(), out.find(r#"class="f2-1""#).unwrap());
        assert!(l1 < f2, "{out}");
    }

    /// A polygon with a hole as feature 10, a line as 11 and a point as 12.
    fn three_features(cam: &Camera) -> Mesh {
        let mut b = MeshBuilder::for_view(cam);
        b.set_class(1);
        b.set_feature(10);
        b.begin_polygon();
        for r in [[[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0], [10.0, 10.0]], [[40.0, 40.0], [40.0, 60.0], [60.0, 60.0], [60.0, 40.0], [40.0, 40.0]]] {
            b.begin_ring();
            b.points(&r);
            b.end_ring();
        }
        b.end_polygon();
        b.set_feature(11);
        b.begin_line();
        b.points(&[[120.0, 20.0], [180.0, 20.0], [180.0, 80.0]]);
        b.end_line();
        b.set_feature(12);
        b.point([150.0, 50.0]);
        b.finish()
    }

    /// The `d` of every path of class `name`, joined.
    fn path_data(out: &str, name: &str) -> String {
        let key = format!("class=\"{name}\"");
        out.lines().filter(|l| l.starts_with("<path") && l.contains(&key)).filter_map(|l| l.split(" d=\"").nth(1)).map(|d| d.trim_end_matches("\"/>")).collect()
    }

    #[test]
    fn a_line_in_two_tiles_guard_bands_is_written_once() {
        use crate::grid::TileFrame;
        use crate::prepare::Detail;
        let cam = camera();
        let mut svg = SvgWriter::new(cam.size_px, None);
        for cx in [50.0, 150.0] {
            let frame = TileFrame { center: [cx, 50.0], half_span: 50.0 };
            let mut b = MeshBuilder::new(frame, Detail { tile_px: 100.0, grid_px: 0.25, guard_px: 10.0 });
            b.set_class(1);
            b.begin_line();
            b.points(&[[20.0, 50.0], [180.0, 50.0]]);
            b.end_line();
            svg.draw(&b.finish(), &cam, &palette());
        }
        // Each tile its own half, where each would reach 10 into the other.
        assert_eq!(path_data(&svg.finish(), "l0-1"), "M20,50 100,50M100,50 180,50");
    }

    #[test]
    fn a_dashed_chain_part_way_through_its_pattern_has_its_own_offset() {
        let cam = camera();
        let mut b = MeshBuilder::for_view(&cam);
        b.set_class(1);
        b.begin_line();
        b.points(&[[10.0, 20.0], [60.0, 20.0]]);
        b.end_line();
        b.set_line_start(7.0);
        b.begin_line();
        b.points(&[[10.0, 40.0], [60.0, 40.0]]);
        b.end_line();
        let mut palette = palette();
        palette.set(1, Style { dash: crate::Dash::new(&[4.0, 2.0], 1.0), ..*palette.get(1) });
        let mut svg = SvgWriter::new(cam.size_px, None);
        svg.draw(&b.finish(), &cam, &palette);
        let out = svg.finish();
        assert!(out.contains(";stroke-dasharray:4,2;stroke-dashoffset:1}"), "{out}");
        // The first line shares the class's path; the second starts 7
        // along, so 1 + 7 into a pattern of 6.
        assert!(out.contains(r#"<path class="l0-1" d="M10,80 60,80"/>"#), "{out}");
        assert!(out.contains(r#"<path class="l0-1" style="stroke-dashoffset:2" d="M10,60 60,60"/>"#), "{out}");
    }

    #[test]
    fn a_shared_border_is_written_once_and_whole_through_a_state() {
        let cam = camera();
        let mut b = MeshBuilder::for_view(&cam);
        b.set_class(1);
        for (id, x) in [(1, 10.0), (2, 50.0)] {
            b.set_feature(id);
            b.begin_polygon();
            b.begin_ring();
            b.points(&[[x, 10.0], [x + 40.0, 10.0], [x + 40.0, 40.0], [x, 40.0], [x, 10.0]]);
            b.end_ring();
            b.end_polygon();
        }
        let mesh = b.finish();
        let mut svg = SvgWriter::new(cam.size_px, None);
        svg.draw(&mesh, &cam, &palette());
        // The first square's right side is left to the second.
        assert_eq!(path_data(&svg.finish(), "l0-1"), "M10,90 50,90M50,60 10,60 10,90M50,90 90,90 90,60 50,60 50,90Z");
        // Only the first holding the state: it writes the side itself.
        assert_eq!(path_data(&write_state(&mesh, &cam, 1, &[1, 0], 1), "l1-1"), "M10,90 50,90 50,60 10,60 10,90Z");
    }

    fn write_state(mesh: &Mesh, cam: &Camera, first: u32, flags: &[u8], mask: u8) -> String {
        let mut svg = SvgWriter::new(cam.size_px, None);
        svg.begin_layer();
        svg.draw_state(mesh, cam, &palette(), first, flags, mask);
        svg.finish()
    }

    #[test]
    fn a_state_writes_only_the_features_holding_it() {
        let cam = camera();
        let mesh = three_features(&cam);
        // The polygon and the point hold bit 0; the line holds only bit 1.
        let out = write_state(&mesh, &cam, 10, &[1, 2, 1], 1);
        assert!(out.contains(r#"<path class="f1-1" d="M10,10 90,10 90,90 10,90ZM60,60 60,40 40,40 40,60Z"/>"#), "{out}");
        // The polygon's outlines are its own; the line's are gone.
        assert!(out.contains(r#"<path class="l1-1" d="M10,90 90,90 90,10 10,10 10,90ZM40,60 40,40 60,40 60,60 40,60Z"/>"#), "{out}");
        assert!(!out.contains("M120,80"), "{out}");
        assert!(out.contains(r#"<circle class="p1-1""#), "{out}");
        // Only the line.
        let out = write_state(&mesh, &cam, 10, &[1, 2, 1], 2);
        assert!(!out.contains("f1-1") && !out.contains("<circle"), "{out}");
        assert!(out.contains(r#"<path class="l1-1" d="M120,80 180,80 180,20"/>"#), "{out}");
    }

    #[test]
    fn a_state_holding_nothing_writes_an_empty_layer() {
        let cam = camera();
        let mesh = three_features(&cam);
        let empty = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"200\" height=\"100\" viewBox=\"0 0 200 100\">\n</svg>\n";
        assert_eq!(write_state(&mesh, &cam, 10, &[0, 0, 0], 3), empty);
        assert_eq!(write_state(&mesh, &cam, 10, &[1, 1, 1], 2), empty);
        // A span that misses every feature, and one that wraps past the top
        // of the range to end just short of them.
        assert_eq!(write_state(&mesh, &cam, 13, &[1; 4], 1), empty);
        assert_eq!(write_state(&mesh, &cam, u32::MAX - 5, &[1; 16], 1), empty);
    }

    #[test]
    fn a_span_wrapping_past_the_top_holds_its_features() {
        let cam = camera();
        let mesh = three_features(&cam);
        let mut flags = [0u8; 20];
        flags[6 + 11] = 1; // id 11 from u32::MAX - 5
        let out = write_state(&mesh, &cam, u32::MAX - 5, &flags, 1);
        assert!(out.contains(r#"<path class="l1-1" d="M120,80 180,80 180,20"/>"#), "{out}");
        assert!(!out.contains("f1-1") && !out.contains("<circle"), "{out}");
    }

    #[test]
    fn every_flag_set_writes_what_draw_does() {
        let cam = camera();
        let mesh = three_features(&cam);
        let mut svg = SvgWriter::new(cam.size_px, None);
        svg.begin_layer();
        svg.draw(&mesh, &cam, &palette());
        assert_eq!(write_state(&mesh, &cam, 10, &[1, 1, 1], 1), svg.finish());
    }

    /// Symbols for markers: a 4 x 2 colour image, and a 2 x 4 mask anchored
    /// at its bottom middle.
    fn symbols() -> (Symbols, crate::SymbolId, crate::SymbolId) {
        let mut symbols = Symbols::new();
        let colour = symbols.add(Symbol::new(4, 2, [10, 20, 30, 255].repeat(8)).unwrap()).unwrap();
        let mut pin = Symbol::new(2, 4, [10, 20, 30, 128].repeat(8)).unwrap();
        pin.kind = Kind::Mask;
        pin.anchor = [0.5, 1.0];
        let pin = symbols.add(pin).unwrap();
        (symbols, colour, pin)
    }

    fn marker_palette(colour: crate::SymbolId, pin: crate::SymbolId) -> Palette {
        use crate::style::Marker;
        let mut p = Palette::new();
        let fill = [0.0, 0.0, 1.0, 1.0];
        p.set(1, Style { fill, point_size: 8.0, marker: Some(Marker { rotation: 30.0, ..Marker::new(colour) }), ..Style::default() });
        p.set(2, Style { fill, point_size: 8.0, marker: Some(Marker { rotation: 30.0, align: Align::Map, ..Marker::new(colour) }), ..Style::default() });
        p.set(3, Style { fill: [1.0, 0.0, 0.0, 0.5], point_size: 8.0, marker: Some(Marker::new(pin)), ..Style::default() });
        p
    }

    #[test]
    fn markers_are_uses_of_one_definition_each() {
        let (symbols, colour, pin) = symbols();
        let mut cam = camera();
        cam.bearing = std::f64::consts::FRAC_PI_2;
        let mut b = MeshBuilder::for_view(&cam);
        for class in 1..=3 {
            b.set_class(class);
            b.point([100.0, 50.0]);
            b.point([110.0, 50.0]);
        }
        let mesh = b.finish();
        let mut svg = SvgWriter::new(cam.size_px, None);
        svg.symbols(&symbols);
        svg.begin_layer();
        svg.draw(&mesh, &cam, &marker_palette(colour, pin));
        let out = svg.finish();
        // Each symbol once, the mask's with its mask.
        assert_eq!(out.matches("<image id=\"sym0\"").count(), 1, "{out}");
        assert_eq!(out.matches("<image id=\"sym1\"").count(), 1, "{out}");
        assert!(out.contains(r##"<mask id="mask1" style="mask-type:alpha"><use href="#sym1"/></mask>"##), "{out}");
        // Longest side 8 for an image of 4: scale 2, about the centre. On
        // the screen its rotation holds; on the map it turns back by the
        // bearing.
        assert!(out.contains(r##"<use href="#sym0" transform="translate(100,50) rotate(30) scale(2) translate(-2,-1)"/>"##), "{out}");
        assert!(out.contains(r##"<use href="#sym0" transform="translate(100,50) rotate(-60) scale(2) translate(-2,-1)"/>"##), "{out}");
        // The mask: the class's fill through it, anchored at its foot.
        assert!(out.contains(r#"<rect class="m1-3" width="2" height="4" mask="url(#mask1)" transform="translate(100,50) scale(2) translate(-1,-4)"/>"#), "{out}");
        assert!(out.contains(".m1-3{fill:#ff0000;opacity:0.5;stroke:none}"), "{out}");
        assert_eq!(out.matches("<use href=\"#sym0\"").count(), 4, "{out}");
        // A symbol not given draws nothing.
        let mut svg = SvgWriter::new(cam.size_px, None);
        svg.begin_layer();
        svg.draw(&mesh, &cam, &marker_palette(colour, pin));
        assert!(!svg.finish().contains("<use"));
    }

    #[test]
    fn a_point_in_two_guard_bands_is_written_once() {
        use crate::grid::TileFrame;
        use crate::prepare::Detail;
        let cam = camera();
        let tile = |cx: f64| {
            let frame = TileFrame { center: [cx, 50.0], half_span: 50.0 };
            let mut b = MeshBuilder::new(frame, Detail { tile_px: 100.0, grid_px: 0.25, guard_px: 10.0 });
            b.set_class(1);
            b.point([103.0, 50.0]);
            b.finish()
        };
        let tiles = [tile(50.0), tile(150.0)];
        assert!(tiles.iter().all(|t| t.points.len() == 1));
        let mut svg = SvgWriter::new(cam.size_px, None);
        svg.begin_layer();
        for t in &tiles {
            svg.draw(t, &cam, &palette());
        }
        assert_eq!(svg.finish().matches("<circle").count(), 1);
    }

    #[test]
    fn a_png_is_whole() {
        let pixels = [1, 2, 3, 4, 5, 6, 7, 8];
        let out = png(2, 1, &pixels, false);
        assert_eq!(&out[..8], b"\x89PNG\r\n\x1a\n");
        // IHDR: 2 x 1, 8-bit RGBA, and its CRC over type and data.
        assert_eq!(&out[8..29], &[0, 0, 0, 13, b'I', b'H', b'D', b'R', 0, 0, 0, 2, 0, 0, 0, 1, 8, 6, 0, 0, 0]);
        assert_eq!(&out[29..33], &crc32(&out[12..29]).to_be_bytes());
        // IDAT: one stored block of the filter byte and the row, then the
        // Adler-32 of the same nine bytes.
        let idat = &out[33..];
        assert_eq!(&idat[4..8], b"IDAT");
        assert_eq!(&idat[8..10], &[0x78, 0x01]);
        assert_eq!(&idat[10..15], &[1, 9, 0, 0xf6, 0xff]);
        assert_eq!(&idat[15..24], &[0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(&idat[24..28], &[0x00, 0x81, 0x00, 0x25]);
        assert!(out.ends_with(&[0, 0, 0, 0, b'I', b'E', b'N', b'D', 0xae, 0x42, 0x60, 0x82]));
        // A mask is white, its alpha kept.
        assert_eq!(&png(2, 1, &pixels, true)[48..57], &[0, 255, 255, 255, 4, 255, 255, 255, 8]);
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        assert_eq!([base64(b"M"), base64(b"Ma"), base64(b"Man")], ["TQ==", "TWE=", "TWFu"]);
    }

    #[test]
    fn numbers_are_short() {
        assert_eq!([num(1.0), num(1.256), num(-0.001), num(2.5)], ["1", "1.26", "0", "2.5"]);
        assert_eq!(paint([0.0, 0.5, 1.0, 1.0]), "#0080ff");
    }
}
