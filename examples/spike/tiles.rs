//! The frame loop both spikes share: select the tiles the camera needs, plan
//! them against the cache, prepare and upload what is missing, and draw --
//! the counties as one layer, the chosen ones over them, an optional raster
//! overlay as the next -- and pick the county under a pixel.
//!
//! Preparing here is synchronous, so every tile a plan asks for is in the
//! cache before the frame draws and no stand-in is ever shown. A real
//! embedder fetches asynchronously and draws whatever the plan says now.

use crate::data::{self, Counties, Overlay};
use nazca::gl::{GpuColormap, GpuMesh, GpuPalette, GpuRaster, GpuState, Picker, Renderer};
use nazca::grid::TileFrame;
use nazca::{Camera, Mesh, Palette, Plan, Style, TileCache, TileGrid, TileId, TileKey, TileSelector};
use std::collections::HashMap;
use std::ops::Range;

/// The one tiled source the spikes draw, and its layer number for picking.
const COUNTIES: u32 = 0;

/// The feature-state bits the spikes use.
pub const SELECTED: u8 = 1;
pub const HOVERED: u8 = 2;

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
    needed: Vec<TileId>,
    plan: Plan,
    palette: GpuPalette,
    picker: Picker,
    overlay: Overlay,
    raster: Option<(GpuRaster, GpuColormap)>,
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
    /// bytes, with `overlay` over them.
    pub fn new(gl: &glow::Context, renderer: &Renderer, budget: usize, overlay: Overlay) -> TiledMap {
        let counties = Counties::load();
        let mut map = TiledMap {
            flags: vec![0; counties.len()],
            counties,
            grid: TileGrid::web_mercator(512, 20).unwrap(),
            cache: TileCache::new(budget),
            states: HashMap::new(),
            selected: renderer.palette(gl, &data::highlight([1.0, 0.55, 0.0, 0.45], [0.85, 0.3, 0.0, 1.0], 2.5)).unwrap(),
            hovered: renderer.palette(gl, &data::highlight([0.0; 4], [0.1, 0.35, 0.95, 1.0], 4.0)).unwrap(),
            last_pick: None,
            needed: Vec::new(),
            plan: Plan::default(),
            palette: renderer.palette(gl, &data::palette()).unwrap(),
            picker: Picker::new(gl).unwrap(),
            overlay: Overlay::None,
            raster: None,
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

    pub fn set_overlay(&mut self, gl: &glow::Context, renderer: &Renderer, overlay: Overlay) {
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

    /// Draws one frame of `camera`'s view, preparing and uploading the tiles
    /// it needs that the cache lacks.
    pub fn draw(&mut self, gl: &glow::Context, renderer: &Renderer, camera: &Camera) -> Stats {
        renderer.begin_frame(gl, camera, data::BACKGROUND);
        self.cache.begin_frame();
        self.needed.clear();
        self.grid.select(camera, &mut self.needed);
        self.cache.plan(COUNTIES, &self.grid, &self.needed, &mut self.plan);
        let prepared = self.plan.fetch.len();
        for &id in &self.plan.fetch {
            let mesh = self.counties.prepare(&self.grid, id);
            let span = match (mesh.ids.iter().min(), mesh.ids.iter().max()) {
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
            renderer.draw(gl, &tile.mesh, camera, &self.palette);
            let flags = &self.flags[tile.span.clone()];
            if !self.states.contains_key(&id) && flags.iter().any(|&f| f != 0) {
                self.states.insert(id, renderer.state(gl, tile.span.start as u32, flags).unwrap());
            }
        }
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

    /// The county under screen position `px`, over the tiles the last frame
    /// drew, and its feature id.
    pub fn pick(&mut self, gl: &glow::Context, renderer: &Renderer, camera: &Camera, px: [f64; 2]) -> Option<(&'static str, u32)> {
        self.picker.begin(gl, camera, px, 6.0);
        self.picker.begin_layer(gl);
        for &id in &self.plan.draw {
            let tile = self.cache.get(TileKey { source: COUNTIES, tile: id }).unwrap();
            self.picker.draw(gl, renderer, &tile.mesh, camera, &self.palette, COUNTIES);
        }
        let pick = self.picker.finish(gl)?;
        self.last_pick = Some(pick.feature);
        Some((self.counties.name(pick.feature)?, pick.feature))
    }

    /// Flips state `bit` of `feature`, rewriting the flags of the tiles that
    /// hold it. Nothing is prepared or uploaded but those.
    pub fn toggle(&mut self, gl: &glow::Context, renderer: &Renderer, feature: u32, bit: u8) {
        self.flags[feature as usize] ^= bit;
        for (&id, state) in &self.states {
            let span = self.cache.get(TileKey { source: COUNTIES, tile: id }).unwrap().span.clone();
            if span.contains(&(feature as usize)) {
                renderer.update_state(gl, state, &self.flags[span]).unwrap();
            }
        }
    }

    /// Flips state `bit` of the county picked last, if any.
    pub fn toggle_last(&mut self, gl: &glow::Context, renderer: &Renderer, bit: u8) {
        if let Some(feature) = self.last_pick {
            self.toggle(gl, renderer, feature, bit);
        }
    }
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
/// masks over several bits, and rewriting flags.
pub fn check_state(gl: &glow::Context, renderer: &Renderer, size: [u32; 2]) -> Result<String, String> {
    use glow::HasContext;
    let count = (GRID * GRID) as usize;
    let camera = Camera::new([0.0, 0.0], 2.0 / size[0].min(size[1]) as f64, size);
    let mut palette = Palette::new();
    palette.set(1, Style { fill: [1.0, 0.0, 0.0, 1.0], line: [1.0, 0.0, 0.0, 1.0], line_width: 6.0, point_size: 8.0 });
    let palette = renderer.palette(gl, &palette)?;
    let pattern = |n: usize| u8::from(n.is_multiple_of(7)) | u8::from(n.is_multiple_of(11)) << 1;
    let flags: Vec<u8> = (0..count).map(pattern).collect();
    let inverted: Vec<u8> = flags.iter().map(|f| f ^ 3).collect();
    let mut checked = 0;
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
                let mut pixels = vec![0u8; w * h * 4];
                // SAFETY: reads the frame just drawn into a buffer sized for it.
                unsafe { gl.read_pixels(0, 0, w as i32, h as i32, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(Some(&mut pixels))) };
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
            whole.free(gl);
            part.free(gl);
            mesh.free(gl);
        }
    }
    palette.free(gl);
    Ok(format!("state check passed: {checked} features read back"))
}
