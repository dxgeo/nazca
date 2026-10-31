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
//!   their layer, as on screen.
//! - *Points* are `<circle>`s of the class's point size, in its fill colour.
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
//! Styles are written once each, as CSS classes. Rasters are not written: an
//! SVG image needs encoded image bytes, which is the `raster` backend's
//! business.

use crate::camera::Camera;
use crate::prepare::Mesh;
use crate::style::Palette;
use std::collections::BTreeMap;
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
    points: String,
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
            points: String::new(),
        }
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
        for (class, d) in std::mem::take(&mut self.lines) {
            let _ = writeln!(self.body, r#"<path class="l{layer}-{class}" d="{d}"/>"#);
        }
        self.body.push_str(&std::mem::take(&mut self.points));
    }

    /// Draws `mesh` where `camera` puts it, styled by `palette`, into the
    /// current layer's paths (see the module docs).
    pub fn draw(&mut self, mesh: &Mesh, camera: &Camera, palette: &Palette) {
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

        // Outline segments chained into polylines, one path per class.
        let lines = &mut self.lines;
        let segments = mesh.outlines.chunks_exact(2);
        let mut chain: Option<(u8, u32, u32)> = None; // class, first vertex, last vertex
        let close = |lines: &mut BTreeMap<u8, String>, chain: Option<(u8, u32, u32)>| {
            if let Some((class, first, last)) = chain
                && first == last
            {
                lines.get_mut(&class).unwrap().push('Z');
            }
        };
        for s in segments {
            let (a, b) = (s[0], s[1]);
            let class = class_of(a);
            let style = palette.get(class);
            if style.line_width <= 0.0 || style.line[3] <= 0.0 {
                continue;
            }
            lines.entry(class).or_default();
            match chain {
                Some((c, first, last)) if c == class && last == a => chain = Some((c, first, b)),
                _ => {
                    close(lines, chain);
                    let d = lines.get_mut(&class).unwrap();
                    let [x, y] = screen(a);
                    let _ = write!(d, "M{},{}", num(x), num(y));
                    chain = Some((class, a, b));
                }
            }
            let d = lines.get_mut(&class).unwrap();
            let [x, y] = screen(b);
            let _ = write!(d, " {},{}", num(x), num(y));
        }
        close(lines, chain);
        for &class in lines.keys() {
            let style = palette.get(class);
            self.styles.entry(format!("l{layer}-{class}")).or_insert_with(|| {
                format!("fill:none;stroke:{};stroke-width:{};stroke-linecap:round;stroke-linejoin:round", paint(style.line), num(style.line_width as f64))
            });
        }

        for &i in &mesh.points {
            let class = class_of(i);
            let style = palette.get(class);
            if style.point_size <= 0.0 || style.fill[3] <= 0.0 {
                continue;
            }
            let [x, y] = screen(i);
            let name = format!("p{layer}-{class}");
            self.styles.entry(name.clone()).or_insert_with(|| format!("fill:{};stroke:none", paint(style.fill)));
            let _ = writeln!(self.points, r#"<circle class="{name}" cx="{}" cy="{}" r="{}"/>"#, num(x), num(y), num(style.point_size as f64 / 2.0));
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
        if let Some(bg) = self.background {
            let _ = writeln!(out, r#"<rect width="{w}" height="{h}" style="fill:{}"/>"#, paint(bg));
        }
        out.push_str(&self.body);
        out.push_str("</svg>\n");
        out
    }
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
        p.set(1, Style { fill: [1.0, 0.0, 0.0, 1.0], line: [0.0, 0.0, 1.0, 0.5], line_width: 2.0, point_size: 6.0 });
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

    #[test]
    fn numbers_are_short() {
        assert_eq!([num(1.0), num(1.256), num(-0.001), num(2.5)], ["1", "1.26", "0", "2.5"]);
        assert_eq!(paint([0.0, 0.5, 1.0, 1.0]), "#0080ff");
    }
}
