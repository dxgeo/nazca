//! The frame loop both spikes share: select the tiles the camera needs, plan
//! them against the cache, prepare and upload what is missing, and draw --
//! the counties as one layer, an optional raster overlay as the next -- and
//! pick the county under a pixel.
//!
//! Preparing here is synchronous, so every tile a plan asks for is in the
//! cache before the frame draws and no stand-in is ever shown. A real
//! embedder fetches asynchronously and draws whatever the plan says now.

use crate::data::{self, Counties, Overlay};
use nazca::gl::{GpuColormap, GpuMesh, GpuPalette, GpuRaster, Picker, Renderer};
use nazca::{Camera, Plan, TileCache, TileGrid, TileId, TileKey, TileSelector};

/// The one tiled source the spikes draw, and its layer number for picking.
const COUNTIES: u32 = 0;

pub struct TiledMap {
    counties: Counties,
    grid: TileGrid,
    cache: TileCache<GpuMesh>,
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
        let mut map = TiledMap {
            counties: Counties::load(),
            grid: TileGrid::web_mercator(512, 20).unwrap(),
            cache: TileCache::new(budget),
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
            let gpu = renderer.upload(gl, &self.counties.prepare(&self.grid, id)).unwrap();
            let bytes = gpu.bytes();
            for (_, evicted) in self.cache.insert(TileKey { source: COUNTIES, tile: id }, gpu, bytes) {
                evicted.free(gl);
            }
        }
        if prepared > 0 {
            // Everything asked for has arrived; plan again to draw it.
            self.cache.plan(COUNTIES, &self.grid, &self.needed, &mut self.plan);
        }
        renderer.begin_layer(gl);
        for &id in &self.plan.draw {
            let mesh = self.cache.get(TileKey { source: COUNTIES, tile: id }).unwrap();
            renderer.draw(gl, mesh, camera, &self.palette);
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
            let mesh = self.cache.get(TileKey { source: COUNTIES, tile: id }).unwrap();
            self.picker.draw(gl, renderer, mesh, camera, &self.palette, COUNTIES);
        }
        let pick = self.picker.finish(gl)?;
        Some((self.counties.name(pick.feature)?, pick.feature))
    }
}
