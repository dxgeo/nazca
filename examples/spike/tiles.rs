//! The frame loop both spikes share: select the tiles the camera needs, plan
//! them against the cache, prepare and upload what is missing, and draw --
//! the counties and their markers as one layer, an optional hatch or
//! stipple over them, the chosen ones over that, an optional raster overlay
//! as the next -- and pick the county or marker
//! under a pixel. A filter shows
//! every other county, by drawing and picking the layer only through a
//! state; [`TiledMap::svg`] writes the same view through the same states.
//!
//! Preparing here is synchronous, so every tile a plan asks for is in the
//! cache before the frame draws and no stand-in is ever shown. A real
//! embedder fetches asynchronously and draws whatever the plan says now.

use crate::data::{self, Counties, Icons, MARKER_BASE, Markers, Outlines, Overlay, Patterns, QUARTERS};
use nazca::gl::{Context, GpuColormap, GpuMesh, GpuPalette, GpuRaster, GpuState, Pick, Picker, Renderer};
use nazca::grid::{Rect, TileFrame};
use nazca::svg::SvgWriter;
use nazca::{Camera, Dash, Detail, Mesh, MeshBuilder, Palette, Plan, Sink, Style, TileCache, TileGrid, TileId, TileKey, TileSelector};
use std::collections::HashMap;
use std::ops::Range;

/// The one tiled source the spikes draw, and its layer number for picking.
const COUNTIES: u32 = 0;

/// The feature-state bits the spikes use.
pub const SELECTED: u8 = 1;
pub const HOVERED: u8 = 2;
/// Held by the counties the filter shows, while it is on.
const SHOWN: u8 = 4;

fn selected_style() -> Palette {
    data::highlight([1.0, 0.55, 0.0, 0.45], [0.85, 0.3, 0.0, 1.0], 2.5, None)
}

/// A hovered county's outline, dashed when the counties' are.
fn hovered_style(outlines: Outlines) -> Palette {
    data::highlight([0.0; 4], [0.1, 0.35, 0.95, 1.0], 4.0, outlines.highlight_dash())
}

/// A tile on the GPU and the span of county ids in it.
struct Tile {
    mesh: GpuMesh,
    span: Range<usize>,
}

pub struct TiledMap {
    counties: Counties,
    grid: TileGrid,
    cache: TileCache<Tile>,
    /// The state bits of every county, by feature id.
    flags: Vec<u8>,
    /// The flags of the tiles that have held a chosen county.
    states: HashMap<TileId, GpuState>,
    selected: GpuPalette,
    hovered: GpuPalette,
    last_pick: Option<u32>,
    filtered: bool,
    needed: Vec<TileId>,
    plan: Plan,
    palette: GpuPalette,
    picker: Picker,
    overlay: Overlay,
    raster: Option<(GpuRaster, GpuColormap)>,
    icons: Icons,
    markers: Markers,
    outlines: Outlines,
    patterns: Patterns,
    hatching: GpuPalette,
}

/// What the last frame did.
pub struct Stats {
    pub drawn: usize,
    pub prepared: usize,
    pub level: u8,
    pub cached: usize,
    pub cached_bytes: usize,
}

impl TiledMap {
    /// The counties on 512-pixel Web Mercator tiles, cached up to `budget`
    /// bytes, with `overlay` over them. Sets `renderer`'s symbols to the
    /// markers'.
    pub fn new(gl: &Context, renderer: &mut Renderer, budget: usize, overlay: Overlay) -> TiledMap {
        let counties = Counties::load();
        let icons = data::icons();
        renderer.set_symbols(gl, &icons.symbols).unwrap();
        let mut map = TiledMap {
            flags: vec![0; counties.len()],
            counties,
            grid: TileGrid::web_mercator(512, 20).unwrap(),
            cache: TileCache::new(budget),
            states: HashMap::new(),
            selected: renderer.palette(gl, &selected_style()).unwrap(),
            hovered: renderer.palette(gl, &hovered_style(Outlines::Solid)).unwrap(),
            last_pick: None,
            filtered: false,
            needed: Vec::new(),
            plan: Plan::default(),
            palette: renderer.palette(gl, &data::palette(Markers::None, Outlines::Solid, &icons)).unwrap(),
            picker: Picker::new(gl).unwrap(),
            overlay: Overlay::None,
            raster: None,
            hatching: renderer.palette(gl, &data::hatching(Patterns::None, &icons)).unwrap(),
            icons,
            markers: Markers::None,
            outlines: Outlines::Solid,
            patterns: Patterns::None,
        };
        map.set_overlay(gl, renderer, overlay);
        map
    }

    pub fn bounds(&self) -> nazca::grid::Rect {
        self.counties.bounds
    }

    pub fn counties(&self) -> &Counties {
        &self.counties
    }

    pub fn overlay(&self) -> Overlay {
        self.overlay
    }

    pub fn set_overlay(&mut self, gl: &Context, renderer: &Renderer, overlay: Overlay) {
        if let Some((raster, colormap)) = self.raster.take() {
            raster.free(gl);
            colormap.free(gl);
        }
        self.overlay = overlay;
        if let Some((classes, values, mesh, colormap)) = data::overlay(overlay) {
            let raster = renderer.upload_raster(gl, &mesh, &data::band(overlay, &classes, &values)).unwrap();
            self.raster = Some((raster, renderer.colormap(gl, &colormap).unwrap()));
        }
    }

    /// Where the marker of the county called `name` stands.
    pub fn marker(&self, name: &str) -> Option<[f64; 2]> {
        self.counties.find(name).map(|i| self.counties.centre(i as usize))
    }

    pub fn markers(&self) -> Markers {
        self.markers
    }

    /// Restyles the markers: one palette upload, no tile touched. The
    /// palette is checked against the guard band the tiles are prepared
    /// with.
    pub fn set_markers(&mut self, gl: &Context, renderer: &Renderer, markers: Markers) {
        let palette = data::palette(markers, self.outlines, &self.icons);
        palette.check_reach(&self.icons.symbols, Detail::default().guard_px).unwrap();
        renderer.update_palette(gl, &mut self.palette, &palette);
        self.markers = markers;
    }

    pub fn outlines(&self) -> Outlines {
        self.outlines
    }

    /// Restyles the counties' outlines, and a hovered county's to match:
    /// palette uploads, no tile touched.
    pub fn set_outlines(&mut self, gl: &Context, renderer: &Renderer, outlines: Outlines) {
        self.outlines = outlines;
        renderer.update_palette(gl, &mut self.palette, &data::palette(self.markers, outlines, &self.icons));
        renderer.update_palette(gl, &mut self.hovered, &hovered_style(outlines));
    }

    pub fn patterns(&self) -> Patterns {
        self.patterns
    }

    /// Restyles the patterns' layer: a palette upload, no tile touched.
    pub fn set_patterns(&mut self, gl: &Context, renderer: &Renderer, patterns: Patterns) {
        self.patterns = patterns;
        renderer.update_palette(gl, &mut self.hatching, &data::hatching(patterns, &self.icons));
    }

    /// Draws one frame of `camera`'s view, preparing and uploading the tiles
    /// it needs that the cache lacks.
    pub fn draw(&mut self, gl: &Context, renderer: &Renderer, camera: &Camera) -> Stats {
        self.draw_with(gl, renderer, camera, &mut || {})
    }

    /// Draws as [`TiledMap::draw`] does, calling `between` after the
    /// counties and before the chosen ones over them: where an embedder's GL
    /// calls of its own would go.
    pub fn draw_with(&mut self, gl: &Context, renderer: &Renderer, camera: &Camera, between: &mut dyn FnMut()) -> Stats {
        renderer.begin_frame(gl, camera, data::BACKGROUND);
        self.cache.begin_frame();
        self.needed.clear();
        self.grid.select(camera, &mut self.needed);
        self.cache.plan(COUNTIES, &self.grid, &self.needed, &mut self.plan);
        let prepared = self.plan.fetch.len();
        for &id in &self.plan.fetch {
            let mesh = self.counties.prepare(&self.grid, id);
            // The counties' ids; their markers' hold no state.
            let counties = mesh.ids.iter().filter(|&&f| f < MARKER_BASE);
            let span = match (counties.clone().min(), counties.max()) {
                (Some(&lo), Some(&hi)) => lo as usize..hi as usize + 1,
                _ => 0..0,
            };
            let gpu = renderer.upload(gl, &mesh).unwrap();
            let bytes = gpu.bytes();
            for (key, evicted) in self.cache.insert(TileKey { source: COUNTIES, tile: id }, Tile { mesh: gpu, span }, bytes) {
                evicted.mesh.free(gl);
                if let Some(state) = self.states.remove(&key.tile) {
                    state.free(gl);
                }
            }
        }
        if prepared > 0 {
            // Everything asked for has arrived; plan again to draw it.
            self.cache.plan(COUNTIES, &self.grid, &self.needed, &mut self.plan);
        }
        renderer.begin_layer(gl);
        for &id in &self.plan.draw {
            let tile = self.cache.get(TileKey { source: COUNTIES, tile: id }).unwrap();
            let flags = &self.flags[tile.span.clone()];
            if !self.states.contains_key(&id) && flags.iter().any(|&f| f != 0) {
                self.states.insert(id, renderer.state(gl, tile.span.start as u32, flags).unwrap());
            }
            match (self.filtered, self.states.get(&id)) {
                (false, _) => renderer.draw(gl, &tile.mesh, camera, &self.palette),
                (true, Some(state)) => renderer.draw_state(gl, &tile.mesh, camera, &self.palette, state, SHOWN),
                (true, None) => {}
            }
        }
        if self.patterns != Patterns::None {
            renderer.begin_layer(gl);
            for &id in &self.plan.draw {
                let tile = self.cache.get(TileKey { source: COUNTIES, tile: id }).unwrap();
                match (self.filtered, self.states.get(&id)) {
                    (false, _) => renderer.draw(gl, &tile.mesh, camera, &self.hatching),
                    (true, Some(state)) => renderer.draw_state(gl, &tile.mesh, camera, &self.hatching, state, SHOWN),
                    (true, None) => {}
                }
            }
        }
        between();
        for (bit, palette) in [(SELECTED, &self.selected), (HOVERED, &self.hovered)] {
            renderer.begin_layer(gl);
            for &id in &self.plan.draw {
                if let Some(state) = self.states.get(&id) {
                    let tile = self.cache.get(TileKey { source: COUNTIES, tile: id }).unwrap();
                    renderer.draw_state(gl, &tile.mesh, camera, palette, state, bit);
                }
            }
        }
        if let Some((raster, colormap)) = &self.raster {
            renderer.begin_layer(gl);
            renderer.draw_raster(gl, raster, camera, colormap, 0.6);
        }
        Stats {
            drawn: self.plan.draw.len(),
            prepared,
            level: self.needed.first().map_or(0, |t| t.z),
            cached: self.cache.len(),
            cached_bytes: self.cache.used(),
        }
    }

    /// What lies under screen position `px`, over the tiles the last frame
    /// drew, a county or its marker, named, and the county's feature id.
    pub fn pick(&mut self, gl: &Context, renderer: &Renderer, camera: &Camera, px: [f64; 2]) -> Option<(String, u32)> {
        self.picker.begin(gl, camera, px, 6.0);
        self.picker.begin_layer(gl);
        for &id in &self.plan.draw {
            let tile = self.cache.get(TileKey { source: COUNTIES, tile: id }).unwrap();
            match (self.filtered, self.states.get(&id)) {
                (false, _) => self.picker.draw(gl, renderer, &tile.mesh, camera, &self.palette, COUNTIES),
                (true, Some(state)) => self.picker.draw_state(gl, renderer, &tile.mesh, camera, &self.palette, COUNTIES, state, SHOWN),
                (true, None) => {}
            }
        }
        let pick = self.picker.finish(gl)?;
        let county = pick.feature % MARKER_BASE;
        self.last_pick = Some(county);
        let name = self.counties.name(county)?;
        let what = if pick.feature >= MARKER_BASE { format!("the marker of {name} County") } else { format!("{name} County") };
        Some((what, county))
    }

    /// Flips state `bit` of `feature`, rewriting the flags of the tiles that
    /// hold it. Nothing is prepared or uploaded but those.
    pub fn toggle(&mut self, gl: &Context, renderer: &Renderer, feature: u32, bit: u8) {
        self.flags[feature as usize] ^= bit;
        for (&id, state) in &self.states {
            let span = self.cache.get(TileKey { source: COUNTIES, tile: id }).unwrap().span.clone();
            if span.contains(&(feature as usize)) {
                renderer.update_state(gl, state, &self.flags[span]).unwrap();
            }
        }
    }

    /// Shows every other county, or all of them again. Rewrites every tile's
    /// flags; prepares and uploads nothing.
    pub fn set_filter(&mut self, gl: &Context, renderer: &Renderer, on: bool) {
        self.filtered = on;
        for (i, f) in self.flags.iter_mut().enumerate() {
            *f = if on && i % 2 == 0 { *f | SHOWN } else { *f & !SHOWN };
        }
        for (&id, state) in &self.states {
            let span = self.cache.get(TileKey { source: COUNTIES, tile: id }).unwrap().span.clone();
            renderer.update_state(gl, state, &self.flags[span]).unwrap();
        }
    }

    pub fn filtered(&self) -> bool {
        self.filtered
    }

    /// `camera`'s view as an SVG, the layers as [`TiledMap::draw`] draws
    /// them but for the raster: the counties prepared again as one view,
    /// written through the same flags.
    pub fn svg(&self, camera: &Camera) -> String {
        let mut builder = MeshBuilder::for_view(camera);
        self.counties.prepare_into(&mut builder, camera.visible_bounds());
        let mesh = builder.finish();
        let mut svg = SvgWriter::new(camera.size_px, Some(data::BACKGROUND));
        svg.symbols(&self.icons.symbols);
        svg.begin_layer();
        let palette = data::palette(self.markers, self.outlines, &self.icons);
        let hatching = data::hatching(self.patterns, &self.icons);
        let looks = if self.patterns == Patterns::None { &[palette][..] } else { &[palette, hatching][..] };
        for (k, palette) in looks.iter().enumerate() {
            if k > 0 {
                svg.begin_layer();
            }
            if self.filtered {
                svg.draw_state(&mesh, camera, palette, 0, &self.flags, SHOWN);
            } else {
                svg.draw(&mesh, camera, palette);
            }
        }
        for (bit, palette) in [(SELECTED, selected_style()), (HOVERED, hovered_style(self.outlines))] {
            svg.begin_layer();
            svg.draw_state(&mesh, camera, &palette, 0, &self.flags, bit);
        }
        svg.finish()
    }

    /// Flips state `bit` of the county picked last, if any.
    pub fn toggle_last(&mut self, gl: &Context, renderer: &Renderer, bit: u8) {
        if let Some(feature) = self.last_pick {
            self.toggle(gl, renderer, feature, bit);
        }
    }
}

/// Checks the markers: a pin that reaches past half the guard band only
/// when it turns is refused, and with the colour markers drawn, a click on
/// a badge picks it, and a click on its transparent corner picks the county
/// under it. Leaves the markers as they were.
pub fn check_markers(map: &mut TiledMap, gl: &Context, renderer: &Renderer, camera: &Camera) -> Result<String, String> {
    let guard = Detail::default().guard_px;
    let refused = data::too_far(&map.icons).check_reach(&map.icons.symbols, guard).err().ok_or("a turning 16-pixel pin was not refused")?;
    let was = map.markers;
    map.set_markers(gl, renderer, Markers::Colour);
    map.draw(gl, renderer, camera);
    // Hampden's marker is a badge 22 pixels across, over land: a disc of
    // radius 10.3 in a square of 11 a side, so (8.8, 8.8) off its centre is
    // transparent. (Barnstable's, the first badge, stands in the bay.)
    let name = "Hampden";
    let [x, y] = camera.map_to_screen(map.marker(name).ok_or("no Hampden County")?);
    let centre = map.pick(gl, renderer, camera, [x, y]).map(|p| p.0);
    let corner = map.pick(gl, renderer, camera, [x + 8.8, y + 8.8]).map(|p| p.0);
    map.set_markers(gl, renderer, was);
    let badge = format!("the marker of {name} County");
    if centre.as_deref() != Some(badge.as_str()) {
        return Err(format!("the badge's centre picked {centre:?}"));
    }
    if corner.as_deref().is_none_or(|c| c.starts_with("the marker")) {
        return Err(format!("the badge's transparent corner picked {corner:?}"));
    }
    Ok(format!("marker check passed: refused \"{refused}\"; the badge's centre picked {badge}, its corner {}", corner.unwrap()))
}

/// Which primitive a feature of [`check_state`]'s grid is.
#[derive(Clone, Copy, Debug)]
enum Kind {
    Fill,
    Line,
    Point,
}

const GRID: u32 = 50;

/// A `GRID` x `GRID` mesh of one small feature a cell, with ids from `base`.
fn grid_mesh(kind: Kind, base: u32) -> Mesh {
    let mut m = Mesh::new(TileFrame { center: [0.0, 0.0], half_span: 1.0 });
    for n in 0..GRID * GRID {
        let [x, y] = cell_centre(n);
        let i = m.vertex_count() as u32;
        let r = 0.3 * 2.0 / GRID as f32;
        let corners: &[[f32; 2]] = match kind {
            Kind::Fill => &[[x - r, y - r], [x + r, y - r], [x + r, y + r], [x - r, y + r]],
            Kind::Line => &[[x - r, y], [x + r, y]],
            Kind::Point => &[[x, y]],
        };
        for c in corners {
            m.vertices.extend_from_slice(&[c[0], c[1], 1.0]);
            m.ids.push(base.wrapping_add(n));
        }
        match kind {
            Kind::Fill => m.fills.extend_from_slice(&[i, i + 1, i + 2, i, i + 2, i + 3]),
            Kind::Line => m.outlines.extend_from_slice(&[i, i + 1]),
            Kind::Point => m.points.push(i),
        }
    }
    m
}

fn cell_centre(n: u32) -> [f32; 2] {
    let c = |v: u32| -1.0 + 2.0 * (v as f32 + 0.5) / GRID as f32;
    [c(n % GRID), c(n / GRID)]
}

/// Measures the `GRID` x `GRID` points drawn as dots and as markers, at three
/// sizes, by `clock` in milliseconds: each look drawn ten times a frame, a
/// layer each, the frame waited for by reading a pixel back, and an empty
/// frame's time taken off, so what is left is the draws' own cost.
pub fn measure_markers(map: &TiledMap, gl: &Context, renderer: &Renderer, size: [u32; 2], clock: &dyn Fn() -> f64) -> Result<String, String> {
    const DRAWS: usize = 10;
    const FRAMES: usize = 30;
    let camera = Camera::new([0.0, 0.0], 2.0 / size[0].min(size[1]) as f64, size);
    let mesh = renderer.upload(gl, &grid_mesh(Kind::Point, 0))?;
    let time = |draw: &dyn Fn()| {
        let frame = || {
            renderer.begin_frame(gl, &camera, [1.0; 4]);
            draw();
            gl.read_pixels([1, 1]);
        };
        (0..3).for_each(|_| frame());
        let t = clock();
        (0..FRAMES).for_each(|_| frame());
        (clock() - t) / FRAMES as f64
    };
    let empty = time(&|| {});
    let mut report = vec![format!("an empty frame {empty:.3} ms")];
    for px in [8.0, 24.0, 64.0] {
        let mut row = Vec::new();
        for (name, palette) in data::grid_looks(&map.icons, px) {
            let gpu = renderer.palette(gl, &palette)?;
            let frame = time(&|| {
                for _ in 0..DRAWS {
                    renderer.begin_layer(gl);
                    renderer.draw(gl, &mesh, &camera, &gpu);
                }
            });
            row.push(format!("{name} {:.3}", (frame - empty) / DRAWS as f64));
            gpu.free(gl);
        }
        report.push(format!("{px} px: {}", row.join(", ")));
    }
    mesh.free(gl);
    Ok(format!("marker cost, ms a draw of {} points at {} x {}:\n  {}", GRID * GRID, size[0], size[1], report.join("\n  ")))
}

/// Checks what the counties are too few to reach, by drawing `GRID` x `GRID`
/// features of each primitive and reading the framebuffer: the flags' wrap
/// onto rows, the edges of a span, ids wrapping past the top of the range,
/// masks over several bits, and rewriting flags. Then picks every cell
/// through the state, over a grid of fills a layer below: a cell picks its
/// own feature where it holds the bit and the fill beneath where it doesn't.
pub fn check_state(gl: &Context, renderer: &Renderer, size: [u32; 2]) -> Result<String, String> {
    let count = (GRID * GRID) as usize;
    let camera = Camera::new([0.0, 0.0], 2.0 / size[0].min(size[1]) as f64, size);
    let mut palette = Palette::new();
    let mut style = Style::default();
    style.fill = [1.0, 0.0, 0.0, 1.0];
    style.line = [1.0, 0.0, 0.0, 1.0];
    style.line_width = 6.0;
    style.point_size = 8.0;
    palette.set(1, style);
    let palette = renderer.palette(gl, &palette)?;
    let pattern = |n: usize| u8::from(n.is_multiple_of(7)) | u8::from(n.is_multiple_of(11)) << 1;
    let flags: Vec<u8> = (0..count).map(pattern).collect();
    let inverted: Vec<u8> = flags.iter().map(|f| f ^ 3).collect();
    const BELOW: u32 = 1 << 20;
    let below = renderer.upload(gl, &grid_mesh(Kind::Fill, BELOW))?;
    let mut picker = Picker::new(gl)?;
    let mut checked = 0;
    let mut picked = 0;
    for kind in [Kind::Fill, Kind::Line, Kind::Point] {
        for base in [0, u32::MAX - 1000] {
            let mesh = renderer.upload(gl, &grid_mesh(kind, base))?;
            let whole = renderer.state(gl, base, &flags)?;
            let part = renderer.state(gl, base.wrapping_add(1500), &[1; 200])?;
            let mut verify = |name: &str, state: &GpuState, mask: u8, expect: &dyn Fn(usize) -> bool| -> Result<(), String> {
                renderer.begin_frame(gl, &camera, [0.0, 0.0, 0.0, 1.0]);
                renderer.begin_layer(gl);
                renderer.draw_state(gl, &mesh, &camera, &palette, state, mask);
                let [w, h] = size.map(|v| v as usize);
                let pixels = gl.read_pixels(size);
                for n in 0..count {
                    let [x, y] = cell_centre(n as u32);
                    let half = size[0].min(size[1]) as f32 / 2.0;
                    let px = (w as f32 / 2.0 + x * half) as usize;
                    let py = (h as f32 / 2.0 + y * half) as usize;
                    let red = pixels[(py * w + px) * 4];
                    if (red > 127) != expect(n) {
                        return Err(format!("{kind:?}, base {base}, {name}: feature {n} has red {red}, wanted drawn = {}", expect(n)));
                    }
                }
                checked += count;
                Ok(())
            };
            verify("bit 0", &whole, 1, &|n| pattern(n) & 1 != 0)?;
            verify("bit 1", &whole, 2, &|n| pattern(n) & 2 != 0)?;
            verify("both bits", &whole, 3, &|n| pattern(n) != 0)?;
            verify("no bits", &whole, 0, &|_| false)?;
            verify("span edges", &part, 1, &|n| (1500..1700).contains(&n))?;
            renderer.update_state(gl, &whole, &inverted)?;
            verify("rewritten", &whole, 1, &|n| pattern(n) & 1 == 0)?;
            let mut verify_pick = |name: &str, state: &GpuState, mask: u8, expect: &dyn Fn(usize) -> bool| -> Result<(), String> {
                let half = size[0].min(size[1]) as f64 / 2.0;
                for n in 0..count {
                    let [x, y] = cell_centre(n as u32);
                    let px = [size[0] as f64 / 2.0 + x as f64 * half, size[1] as f64 / 2.0 - y as f64 * half];
                    picker.begin(gl, &camera, px, 0.0);
                    picker.begin_layer(gl);
                    picker.draw(gl, renderer, &below, &camera, &palette, 0);
                    picker.begin_layer(gl);
                    picker.draw_state(gl, renderer, &mesh, &camera, &palette, 1, state, mask);
                    let got = picker.finish(gl);
                    let want = if expect(n) { Pick { layer: 1, feature: base.wrapping_add(n as u32) } } else { Pick { layer: 0, feature: BELOW + n as u32 } };
                    if got != Some(want) {
                        return Err(format!("{kind:?}, base {base}, pick {name}: feature {n} picked {got:?}, wanted {want:?}"));
                    }
                }
                picked += count;
                Ok(())
            };
            verify_pick("rewritten", &whole, 1, &|n| pattern(n) & 1 == 0)?;
            verify_pick("both bits", &whole, 3, &|n| pattern(n) != 3)?;
            verify_pick("span edges", &part, 1, &|n| (1500..1700).contains(&n))?;
            whole.free(gl);
            part.free(gl);
            mesh.free(gl);
        }
    }
    below.free(gl);
    picker.free(gl);
    palette.free(gl);
    Ok(format!("state check passed: {checked} features read back, {picked} picked"))
}

/// Checks dashes and shared borders on three rectangles drawn from one
/// tile, one wide over two narrow, so its bottom edge meets theirs at a
/// T-junction, by reading the framebuffer:
///
/// - a translucent outline is no darker on a shared border than on one of
///   its own, so each shared stretch is drawn once;
/// - a dashed outline covers as much of a shared border as of one of its
///   own, and not all of it, so the dash test runs and a shared border is
///   one pattern, not two out of phase;
/// - panned a whole number of pixels, the dashed outlines are the first
///   frame shifted, so dashes move with the map rather than crawl along it;
/// - each rectangle drawn alone through a state has every side, so a
///   suppressed copy still draws where its partner doesn't, and so does the
///   wide one when the narrow ones' class has no line;
/// - a dashed line cut into two tiles' pieces, the second given where it
///   starts, draws the pixels the whole line does, and without the start
///   doesn't.
///
/// Then measures what a picture can't settle: how far a line start
/// measured on a wiggly coastline misses nazca's own measure at a seam,
/// and that one measured by `MeshBuilder::line_length` in another tile at
/// the level doesn't; how far the phase of a line 1,000 and 5,000 km along
/// at the grid's deepest level is off, and how much it jitters between
/// vertices; and what matching shared borders costs to prepare, timed by
/// `clock` in milliseconds (`std::time::Instant` panics in a browser).
pub fn check_dashes(gl: &Context, renderer: &Renderer, counties: &Counties, size: [u32; 2], clock: &dyn Fn() -> f64) -> Result<String, String> {
    let [w, h] = size.map(|v| v as usize);
    let camera = Camera::new([200.0, 100.0], 1.0, size);
    let frame = TileFrame { center: [200.0, 100.0], half_span: 200.0 };
    let mut b = MeshBuilder::new(frame, Detail { tile_px: 400.0, grid_px: 0.25, guard_px: 0.0 });
    // The wide one in class 2, the narrow ones in class 1.
    for (id, [x0, y0, x1, y1]) in [[100.0, 100.0, 300.0, 160.0], [100.0, 40.0, 200.0, 100.0], [200.0, 40.0, 300.0, 100.0]].into_iter().enumerate() {
        b.set_class(if id == 0 { 2 } else { 1 });
        b.set_feature(id as u32);
        b.begin_polygon();
        b.begin_ring();
        b.points(&[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]);
        b.end_ring();
        b.end_polygon();
    }
    let mesh = renderer.upload(gl, &b.finish())?;
    // A map point's blue, the camera putting a unit on a pixel.
    let blue = |pixels: &[u8], [x, y]: [f64; 2]| {
        let px = (x - 200.0 + w as f64 / 2.0) as usize;
        let py = (y - 100.0 + h as f64 / 2.0) as usize;
        pixels[(py * w + px) * 4 + 2]
    };
    let style = |line: [f32; 4], width: f32, dash: Option<Dash>| {
        let mut s = Style::default();
        s.line = line;
        s.line_width = width;
        s.dash = dash;
        let mut p = Palette::new();
        p.set(1, s);
        p.set(2, s);
        renderer.palette(gl, &p)
    };
    let frame_with = |palette: &GpuPalette, state: Option<&GpuState>| {
        renderer.begin_frame(gl, &camera, [0.0, 0.0, 0.0, 1.0]);
        renderer.begin_layer(gl);
        match state {
            Some(state) => renderer.draw_state(gl, &mesh, &camera, palette, state, 1),
            None => renderer.draw(gl, &mesh, &camera, palette),
        }
        gl.read_pixels(size)
    };
    const SHARED: [[f64; 2]; 3] = [[150.0, 100.0], [250.0, 100.0], [200.0, 70.0]];
    const OWN: [[f64; 2]; 4] = [[150.0, 160.0], [100.0, 70.0], [300.0, 70.0], [250.0, 40.0]];
    let mut report = Vec::new();

    let translucent = style([0.0, 0.0, 1.0, 0.5], 6.0, None)?;
    let pixels = frame_with(&translucent, None);
    let own: Vec<u8> = OWN.iter().map(|&p| blue(&pixels, p)).collect();
    let shared: Vec<u8> = SHARED.iter().map(|&p| blue(&pixels, p)).collect();
    if own.iter().chain(&shared).any(|&v| v.abs_diff(own[0]) > 6) || own[0] < 100 {
        return Err(format!("a translucent outline's blue: {own:?} on borders of one, {shared:?} on shared ones"));
    }
    report.push(format!("translucent blue {} on every border", own[0]));
    translucent.free(gl);

    // Along the narrow ones' shared side, and the first's own left side.
    let dashed = style([0.0, 0.0, 1.0, 1.0], 3.0, Dash::new(&[8.0, 8.0], 0.0))?;
    let pixels = frame_with(&dashed, None);
    let covered = |x: f64| (45..95).filter(|&y| blue(&pixels, [x, y as f64]) > 127).count() as f64 / 50.0;
    let (shared, own) = (covered(200.0), covered(100.0));
    if !(0.4..0.85).contains(&own) || (shared - own).abs() > 0.15 {
        return Err(format!("dashes cover {own:.2} of a border of one and {shared:.2} of a shared one"));
    }
    report.push(format!("dashes cover {own:.2} of a border of one, {shared:.2} of a shared one"));

    // Panned a whole number of pixels, the dashes move with the map: the
    // frame is the first one shifted, not a pattern sliding along its line.
    let [dx, dy] = [37usize, 23usize];
    let panned = Camera::new([200.0 + dx as f64, 100.0 + dy as f64], 1.0, size);
    renderer.begin_frame(gl, &panned, [0.0, 0.0, 0.0, 1.0]);
    renderer.begin_layer(gl);
    renderer.draw(gl, &mesh, &panned, &dashed);
    let moved = gl.read_pixels(size);
    let (mut differ, mut most) = (0, 0u8);
    for y in 0..h - dy {
        for x in 0..w - dx {
            let (a, b) = (&pixels[((y + dy) * w + x + dx) * 4..][..4], &moved[(y * w + x) * 4..][..4]);
            let d = a.iter().zip(b).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
            most = most.max(d);
            differ += (d > 8) as usize;
        }
    }
    if differ > 0 {
        return Err(format!("panned {dx}, {dy} px, the dashed outlines differ from the first frame shifted at {differ} pixels"));
    }
    report.push(format!("panned {dx}, {dy} px, the dashed outlines are the first frame shifted, no channel off by more than {most}"));
    dashed.free(gl);

    let solid = style([0.0, 0.0, 1.0, 1.0], 4.0, None)?;
    let sides: [&[[f64; 2]]; 3] = [&[[150.0, 100.0], [250.0, 100.0], [150.0, 160.0]], &[[150.0, 100.0], [200.0, 70.0], [100.0, 70.0]], &[[250.0, 100.0], [200.0, 70.0], [300.0, 70.0]]];
    for (id, points) in sides.iter().enumerate() {
        let mut flags = [0u8; 3];
        flags[id] = 1;
        let state = renderer.state(gl, 0, &flags)?;
        let pixels = frame_with(&solid, Some(&state));
        state.free(gl);
        if let Some(p) = points.iter().find(|&&p| blue(&pixels, p) < 200) {
            return Err(format!("rectangle {id} alone through a state lacks its side at {p:?}"));
        }
    }
    report.push("each rectangle alone through a state has every side".into());
    solid.free(gl);
    let mut p = Palette::new();
    p.set(2, {
        let mut s = Style::default();
        s.line = [0.0, 0.0, 1.0, 1.0];
        s.line_width = 4.0;
        s
    });
    let wide_only = renderer.palette(gl, &p)?;
    let pixels = frame_with(&wide_only, None);
    if let Some(p) = [[150.0, 100.0], [250.0, 100.0]].into_iter().find(|&p| blue(&pixels, p) < 200) {
        return Err(format!("the wide rectangle's bottom is missing at {p:?} when the narrow ones have no line"));
    }
    report.push("a border left to a class with no line is drawn by the other".into());
    wide_only.free(gl);
    mesh.free(gl);

    // A dashed line along y = 130, whole in one tile, then in two tiles'
    // halves meeting at x = 200.
    let dashed = style([0.0, 0.0, 1.0, 1.0], 3.0, Dash::new(&[8.0, 8.0], 0.0))?;
    let line = |center: [f64; 2], half_span: f64, from: f64, to: f64, start: Option<f64>| {
        let mut b = MeshBuilder::new(TileFrame { center, half_span }, Detail { tile_px: 2.0 * half_span, grid_px: 0.25, guard_px: 0.0 });
        b.set_class(1);
        if let Some(start) = start {
            b.set_line_start(start);
        }
        b.begin_line();
        b.points(&[[from, 130.0], [to, 130.0]]);
        b.end_line();
        renderer.upload(gl, &b.finish())
    };
    let row = |meshes: &[GpuMesh]| -> Vec<u8> {
        renderer.begin_frame(gl, &camera, [0.0, 0.0, 0.0, 1.0]);
        renderer.begin_layer(gl);
        for m in meshes {
            renderer.draw(gl, m, &camera, &dashed);
        }
        let pixels = gl.read_pixels(size);
        (30..370).map(|x| blue(&pixels, [x as f64, 130.0])).collect()
    };
    let whole = [line([200.0, 100.0], 200.0, 20.0, 380.0, None)?];
    let started = [line([100.0, 100.0], 100.0, 20.0, 200.0, None)?, line([300.0, 100.0], 100.0, 200.0, 380.0, Some(180.0))?];
    let restarted = [line([100.0, 100.0], 100.0, 20.0, 200.0, None)?, line([300.0, 100.0], 100.0, 200.0, 380.0, None)?];
    let differ = |a: &[u8], b: &[u8]| a.iter().zip(b).filter(|(a, b)| a.abs_diff(**b) > 64).count();
    let reference = row(&whole);
    let (with, without) = (differ(&reference, &row(&started)), differ(&reference, &row(&restarted)));
    if with > 2 || without < 20 {
        return Err(format!("a line in two tiles' pieces differs from it whole at {with} pixels with its start, {without} without"));
    }
    report.push(format!("a dashed line in two tiles' pieces differs from it whole at {with} of 340 pixels with its start, {without} without"));
    for m in whole.into_iter().chain(started).chain(restarted) {
        m.free(gl);
    }
    dashed.free(gl);

    let grid = TileGrid::web_mercator(512, 20).unwrap();
    let tile_at = |z: u8, p: [f64; 2]| grid.tiles_in(z, Rect { min: p, max: p }).next().unwrap();

    // A line cut after `piece`, the second piece starting where the first
    // ends: by its own length in the source, and by its length as
    // `line_length` measures it in the tile where the first piece starts.
    // The seam's tile, fed the first piece, measures what the line handed
    // in whole would have there.
    let seams = |piece: &[[f64; 2]], levels: &[u8]| -> Result<(Vec<String>, Vec<String>), String> {
        let source: f64 = piece.windows(2).map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1])).sum();
        let (mut by_source, mut by_nazca) = (Vec::new(), Vec::new());
        for &z in levels {
            let mut b = MeshBuilder::for_tile(&grid, tile_at(z, piece[piece.len() - 1])).unwrap();
            b.begin_line();
            b.points(piece);
            b.end_line();
            let mesh = b.finish();
            let whole = *mesh.distances.last().unwrap() * mesh.frame.half_span;
            let measured = MeshBuilder::for_tile(&grid, tile_at(z, piece[0])).unwrap().line_length(piece);
            let res = grid.resolution(z).unwrap();
            if ((measured - whole) / res).abs() > 0.01 {
                return Err(format!("a start measured by line_length jumps {:.3} px at its seam at z{z}", (measured - whole) / res));
            }
            by_source.push(format!("z{z} {:.1}", (source - whole) / res));
            by_nazca.push(format!("{:.1}", (measured - whole) / res));
        }
        Ok((by_source, by_nazca))
    };
    let ring = counties.longest_ring("Barnstable").ok_or("no Barnstable")?;
    let (by_source, by_nazca) = seams(&ring[..=ring.len() / 2], &[6, 8, 10, 12, 14, 16])?;
    report.push(format!(
        "Barnstable's coast, {} vertices, cut at its middle: the seam jumps (px) {} measured on the source, {} by line_length",
        ring.len(),
        by_source.join(", "),
        by_nazca.join(", ")
    ));

    // The fixture's coasts are simplified; a line wiggling 5 m either side
    // every 10 m, for 20 km, is what a detailed source hands in.
    let origin = counties.centre(0);
    let wiggly: Vec<[f64; 2]> = (0..=2000).map(|i| [origin[0] + i as f64 * 10.0, origin[1] + if i % 2 == 0 { -5.0 } else { 5.0 }]).collect();
    let (by_source, by_nazca) = seams(&wiggly, &[8, 10, 12, 14, 16, 18])?;
    report.push(format!(
        "a 20 km line wiggling 5 m every 10 m: the seam after it jumps (px) {} measured on the source, {} by line_length",
        by_source.join(", "),
        by_nazca.join(", ")
    ));

    // A wiggling line on the grid's own points, so snapping moves nothing
    // and the distances nazca measures in `f64` are known exactly.
    let z = grid.max_zoom();
    let id = tile_at(z, counties.centre(0));
    let f = grid.frame(id).unwrap();
    let step = f.half_span / 1024.0; // a quarter pixel
    let res = grid.resolution(z).unwrap();
    for start in [1e6, 5e6] {
        let mut b = MeshBuilder::for_tile(&grid, id).unwrap();
        let points: Vec<[f64; 2]> = (0..150).map(|i| [f.center[0] + (i * 12 - 1000) as f64 * step, f.center[1] + (i % 2 * 4) as f64 * step]).collect();
        b.set_line_start(start);
        b.begin_line();
        b.points(&points);
        b.end_line();
        let mesh = b.finish();
        // One run, its base the first distance as an `f32`, as
        // `Renderer::upload` splits it.
        let base = mesh.distances[0] as f32;
        // At the level's own zoom a local unit is 256 pixels, a power of two
        // the base multiplies by exactly; past it, by a scale that rounds.
        for zoom in [z as f64, z as f64 + 0.4] {
            let res = res / (zoom - z as f64).exp2();
            let unit = (mesh.frame.half_span / res) as f32;
            let (mut exact, mut worst, mut jitter, mut last) = (start, 0.0f64, 0.0f64, None::<f64>);
            for (i, &d) in mesh.distances.iter().enumerate() {
                if i > 0 {
                    exact += (points[i][0] - points[i - 1][0]).hypot(points[i][1] - points[i - 1][1]);
                }
                // As the shader multiplies and adds, in `f32`.
                let offset = (d - base as f64) as f32;
                let err = ((base * unit) as f64 + (offset * unit) as f64) - exact / res;
                worst = worst.max(err.abs());
                if let Some(l) = last {
                    jitter = jitter.max((err - l).abs());
                }
                last = Some(err);
            }
            report.push(format!("a line {} km along at zoom {zoom}: phase off by up to {worst:.2} px, {jitter:.2} px between vertices", start / 1000.0));
        }
    }

    // Matching shared borders, over the counties' tiles at z10.
    let (mut feed, mut finish, mut segments, mut paired) = (0.0, 0.0, 0, 0);
    for id in grid.tiles_in(10, counties.bounds) {
        let mut b = MeshBuilder::for_tile(&grid, id).unwrap();
        let reach = grid.bounds(id).unwrap().expand(Detail::default().guard_px * grid.resolution(10).unwrap());
        let t = clock();
        counties.prepare_into(&mut b, reach);
        feed += clock() - t;
        let t = clock();
        let mesh = b.finish();
        finish += clock() - t;
        segments += mesh.outlines.len() / 2;
        paired += mesh.partners.iter().filter(|&&p| p != Mesh::NO_PARTNER).count();
    }
    report.push(format!(
        "z10: {segments} outline segments, {paired} paired; feeding {feed:.1} ms, matching (finish) {finish:.1} ms; partners {} KB of {} KB of segments",
        segments * 8 / 1024,
        segments * 40 / 1024
    ));
    Ok(format!("dash check passed:\n  {}", report.join("\n  ")))
}

/// Checks pattern fills on a square of one class over the whole view, by
/// reading the framebuffer, at z2, z14 and the grid's deepest level (each
/// 0.3 past the level), north up and turned 30 degrees, around the first
/// county:
///
/// - the square drawn from the level's tiles, from one mesh for the view,
///   and from the tiles six levels up standing in, is the same picture,
///   so the pattern has no seam at tile edges and keeps its precision in a
///   stand-in;
/// - a pattern of four coloured quarters shows, at every pixel clear of a
///   quarter's edge, the quarter the map point's own place in its repeat,
///   measured from the map's origin in `f64`, says: anchored to the map,
///   sized and turned as the camera says, with no swim far from the origin;
/// - an opaque white pattern, magnified and shrunk to each of the atlas's
///   mipmaps and past, is white at every pixel, so a repeat's edge shows no
///   line;
/// - panned a whole number of pixels, the quarters are the first frame
///   shifted;
/// - a click on a hatch's transparent gap picks the square.
pub fn check_patterns(map: &TiledMap, gl: &Context, renderer: &Renderer, size: [u32; 2]) -> Result<String, String> {
    let icons = &map.icons;
    let [w, h] = size.map(|v| v as usize);
    let grid = TileGrid::web_mercator(512, 20).unwrap();
    let centre = map.counties.centre(0);
    let style = |symbol, repeat: f32, fill: [f32; 4]| {
        let mut p = Palette::new();
        p.set(1, data::pattern(symbol, repeat, fill));
        renderer.palette(gl, &p)
    };
    const SQUARE: u32 = 7;
    let square = |b: &mut MeshBuilder, r: Rect| {
        b.set_class(1);
        b.set_feature(SQUARE);
        b.begin_polygon();
        b.begin_ring();
        b.points(&[r.min, [r.max[0], r.min[1]], r.max, [r.min[0], r.max[1]], r.min]);
        b.end_ring();
        b.end_polygon();
    };
    let frame = |camera: &Camera, meshes: &[GpuMesh], palette: &GpuPalette| {
        renderer.begin_frame(gl, camera, [0.0, 0.0, 0.0, 1.0]);
        renderer.begin_layer(gl);
        for m in meshes {
            renderer.draw(gl, m, camera, palette);
        }
        gl.read_pixels(size)
    };
    let mut report = Vec::new();
    for z in [2u8, 14, grid.max_zoom()] {
        for bearing in [0.0f64, 30.0] {
            let mut camera = Camera::new([centre[0] + 0.37, centre[1] - 0.61], grid.resolution(z).unwrap() / 0.3f64.exp2(), size);
            camera.bearing = bearing.to_radians();
            // Past the view on every side, so a stand-in's coarse snapping
            // moves the square's edges off the screen, and within the world.
            let view = camera.visible_bounds();
            let view = view.expand(0.25 * (view.max[0] - view.min[0]));
            let world = grid.bounds(TileId { z: 0, x: 0, y: 0 }).unwrap();
            let r = Rect { min: [view.min[0].max(world.min[0]), view.min[1].max(world.min[1])], max: [view.max[0].min(world.max[0]), view.max[1].min(world.max[1])] };
            // GL's pixel (col, row), rows bottom up, as a map point, and
            // whether it lies inside the square clear of its edge.
            let at = |col: usize, row: usize| camera.screen_to_map([col as f64 + 0.5, (h - row) as f64 - 0.5]);
            let margin = 2.0 * camera.units_per_px;
            let inside = |m: [f64; 2]| (0..2).all(|i| m[i] > r.min[i] + margin && m[i] < r.max[i] - margin);
            let label = format!("z{z}.3 at {bearing}°");

            let mut by_tiles = Vec::new();
            for id in grid.tiles_in(z, r) {
                let mut b = MeshBuilder::for_tile(&grid, id).unwrap();
                square(&mut b, r);
                by_tiles.push(renderer.upload(gl, &b.finish())?);
            }
            let mut b = MeshBuilder::for_view(&camera);
            square(&mut b, r);
            let whole = [renderer.upload(gl, &b.finish())?];
            let mut stand_ins = Vec::new();
            for id in grid.tiles_in(z.saturating_sub(6), r) {
                let mut b = MeshBuilder::for_tile(&grid, id).unwrap();
                square(&mut b, r);
                stand_ins.push(renderer.upload(gl, &b.finish())?);
            }

            let quarters = style(icons.quarters, 24.0, [1.0; 4])?;
            let first = frame(&camera, &whole, &quarters);
            let differ = |a: &[u8], b: &[u8]| {
                let off = |i: usize| a[i * 4..][..4].iter().zip(&b[i * 4..][..4]).any(|(a, b)| a.abs_diff(*b) > 8);
                (0..w * h).filter(|&i| inside(at(i % w, i / w)) && off(i)).count()
            };
            for (name, meshes) in [("its tiles", &by_tiles[..]), ("stand-ins six levels up", &stand_ins[..])] {
                let n = differ(&first, &frame(&camera, meshes, &quarters));
                if n > 0 {
                    return Err(format!("{label}: the quarters drawn from {name} differ from one mesh's at {n} pixels"));
                }
            }

            // The quarter each pixel shows, from its map point in `f64`.
            let cell = 24.0;
            let mut checked = 0;
            for row in (0..h).step_by(3) {
                for col in (0..w).step_by(3) {
                    let m = at(col, row);
                    if !inside(m) {
                        continue;
                    }
                    let fx = (m[0] / camera.units_per_px / cell).rem_euclid(1.0);
                    let fy = (-m[1] / camera.units_per_px / cell).rem_euclid(1.0);
                    // A pixel and a little clear of a quarter's edges.
                    let clear = |f: f64| [f, (f - 0.5).abs(), 1.0 - f].into_iter().all(|d| d * cell > 1.2);
                    if !clear(fx) || !clear(fy) {
                        continue;
                    }
                    let want = QUARTERS[(fx >= 0.5) as usize + 2 * (fy >= 0.5) as usize];
                    let got = &first[(row * w + col) * 4..][..3];
                    if got.iter().zip(want).any(|(g, w)| g.abs_diff(w) > 24) {
                        return Err(format!("{label}: pixel ({col}, {row}) shows {got:?}, wanted quarter {want:?}"));
                    }
                    checked += 1;
                }
            }

            // Panned a whole number of pixels, the frame is the first one
            // shifted, but where a pixel's centre lies on a repeat's edge: the
            // wrap is a hard edge there, which a rounding moves it across.
            let on_edge = |m: [f64; 2]| {
                (0..2).any(|i| {
                    let f = (m[i] / camera.units_per_px / cell).rem_euclid(1.0);
                    f.min(1.0 - f) * cell < 0.02
                })
            };
            let [dx, dy] = [37usize, 23usize];
            let mut panned = camera;
            panned.pan_px([dx as f64, dy as f64]);
            let moved = frame(&panned, &whole, &quarters);
            let (mut shifted, mut most) = (0, 0u8);
            for row in dy..h {
                for col in 0..w - dx {
                    if !inside(at(col, row)) || !inside(at(col + dx, row - dy)) || on_edge(at(col, row)) {
                        continue;
                    }
                    let (a, b) = (&first[(row * w + col) * 4..][..4], &moved[((row - dy) * w + col + dx) * 4..][..4]);
                    let d = a.iter().zip(b).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
                    most = most.max(d);
                    shifted += (d > 8) as usize;
                }
            }
            if shifted > 0 {
                return Err(format!("{label}: panned {dx}, {dy} px, the quarters differ from the first frame shifted at {shifted} pixels"));
            }
            quarters.free(gl);

            // Opaque white repeats, magnified, then shrunk to mipmaps 1 and
            // 2 and past them.
            let mut darkest = 255u8;
            for repeat in [40.0, 10.0, 5.0, 3.0] {
                let solid = style(icons.solid, repeat, [1.0; 4])?;
                let pixels = frame(&camera, &by_tiles, &solid);
                solid.free(gl);
                for row in 0..h {
                    for col in 0..w {
                        if inside(at(col, row)) {
                            let p = &pixels[(row * w + col) * 4..][..3];
                            if let Some(&v) = p.iter().find(|&&v| v < 250) {
                                return Err(format!("{label}: a white pattern repeating every {repeat} px has {v} at ({col}, {row})"));
                            }
                            darkest = darkest.min(*p.iter().min().unwrap());
                        }
                    }
                }
            }
            report.push(format!(
                "{label}: its tiles, one mesh and stand-ins the same picture; {checked} pixels checked, each the quarter the map says; panned {dx}, {dy} px, shifted, no channel off by more than {most}; white repeats at 40, 10, 5 and 3 px no darker than {darkest}"
            ));
            for m in by_tiles.into_iter().chain(whole).chain(stand_ins) {
                m.free(gl);
            }
        }
    }

    // A pick in a hatch's gap finds the square under it.
    let camera = Camera::new(centre, grid.resolution(14).unwrap(), size);
    let r = camera.visible_bounds();
    let mut b = MeshBuilder::for_view(&camera);
    square(&mut b, r);
    let mesh = renderer.upload(gl, &b.finish())?;
    let hatch = style(icons.hatch, 12.0, [1.0; 4])?;
    let pixels = frame(&camera, std::slice::from_ref(&mesh), &hatch);
    let gap = (0..w * h).map(|i| (i % w, i / w)).find(|&(col, row)| {
        let near = col.abs_diff(w / 2) < 40 && row.abs_diff(h / 2) < 40;
        near && pixels[(row * w + col) * 4] == 0
    });
    let (col, row) = gap.ok_or("no gap in the hatch near the centre")?;
    let mut picker = Picker::new(gl)?;
    picker.begin(gl, &camera, [col as f64 + 0.5, (h - row) as f64 - 0.5], 0.0);
    picker.begin_layer(gl);
    picker.draw(gl, renderer, &mesh, &camera, &hatch, 0);
    let picked = picker.finish(gl);
    picker.free(gl);
    hatch.free(gl);
    mesh.free(gl);
    if picked != Some(Pick { layer: 0, feature: SQUARE }) {
        return Err(format!("a pick in the hatch's gap at ({col}, {row}) found {picked:?}"));
    }
    report.push("a pick in a hatch's gap finds the square".into());
    Ok(format!("pattern check passed:\n  {}", report.join("\n  ")))
}
