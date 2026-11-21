//! The frame loop both spikes share: select the tiles the camera needs, plan
//! them against the cache, prepare and upload what is missing, and draw --
//! the counties and their markers as one layer, the chosen ones over them,
//! an optional raster overlay as the next -- and pick the county or marker
//! under a pixel. A filter shows
//! every other county, by drawing and picking the layer only through a
//! state; [`TiledMap::svg`] writes the same view through the same states.
//!
//! Preparing here is synchronous, so every tile a plan asks for is in the
//! cache before the frame draws and no stand-in is ever shown. A real
//! embedder fetches asynchronously and draws whatever the plan says now.

use crate::data::{self, Counties, Icons, MARKER_BASE, Markers, Overlay};
use nazca::gl::{Context, GpuColormap, GpuMesh, GpuPalette, GpuRaster, GpuState, Pick, Picker, Renderer};
use nazca::grid::TileFrame;
use nazca::svg::SvgWriter;
use nazca::{Camera, Detail, Mesh, MeshBuilder, Palette, Plan, Style, TileCache, TileGrid, TileId, TileKey, TileSelector};
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
    data::highlight([1.0, 0.55, 0.0, 0.45], [0.85, 0.3, 0.0, 1.0], 2.5)
}

fn hovered_style() -> Palette {
    data::highlight([0.0; 4], [0.1, 0.35, 0.95, 1.0], 4.0)
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
            hovered: renderer.palette(gl, &hovered_style()).unwrap(),
            last_pick: None,
            filtered: false,
            needed: Vec::new(),
            plan: Plan::default(),
            palette: renderer.palette(gl, &data::palette(Markers::None, &icons)).unwrap(),
            picker: Picker::new(gl).unwrap(),
            overlay: Overlay::None,
            raster: None,
            icons,
            markers: Markers::None,
        };
        map.set_overlay(gl, renderer, overlay);
        map
    }

    pub fn bounds(&self) -> nazca::grid::Rect {
        self.counties.bounds
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
        let palette = data::palette(markers, &self.icons);
        palette.check_reach(&self.icons.symbols, Detail::default().guard_px).unwrap();
        renderer.update_palette(gl, &self.palette, &palette);
        self.markers = markers;
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
        let palette = data::palette(self.markers, &self.icons);
        if self.filtered {
            svg.draw_state(&mesh, camera, &palette, 0, &self.flags, SHOWN);
        } else {
            svg.draw(&mesh, camera, &palette);
        }
        for (bit, palette) in [(SELECTED, selected_style()), (HOVERED, hovered_style())] {
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
    let mut m = Mesh {
        frame: TileFrame { center: [0.0, 0.0], half_span: 1.0 },
        vertices: Vec::new(),
        fills: Vec::new(),
        outlines: Vec::new(),
        points: Vec::new(),
        rings: Vec::new(),
        polygons: Vec::new(),
        ids: Vec::new(),
    };
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
