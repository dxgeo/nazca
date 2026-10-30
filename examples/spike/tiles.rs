//! The frame loop both spikes share: select the tiles the camera needs, plan
//! them against the cache, prepare and upload what is missing, and draw.
//!
//! Preparing here is synchronous, so every tile a plan asks for is in the
//! cache before the frame draws and no stand-in is ever shown. A real
//! embedder fetches asynchronously and draws whatever the plan says now.

use crate::data::{self, Counties};
use nazca::gl::{GpuMesh, Renderer};
use nazca::{Camera, Plan, TileCache, TileGrid, TileId, TileKey, TileSelector};

/// The one source the spikes draw.
const COUNTIES: u32 = 0;

pub struct TiledMap {
    counties: Counties,
    grid: TileGrid,
    cache: TileCache<GpuMesh>,
    needed: Vec<TileId>,
    plan: Plan,
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
    /// bytes.
    pub fn new(budget: usize) -> TiledMap {
        TiledMap {
            counties: Counties::load(),
            grid: TileGrid::web_mercator(512, 20).unwrap(),
            cache: TileCache::new(budget),
            needed: Vec::new(),
            plan: Plan::default(),
        }
    }

    pub fn bounds(&self) -> nazca::grid::Rect {
        self.counties.bounds
    }

    /// Draws one frame of `camera`'s view, preparing and uploading the tiles
    /// it needs that the cache lacks. The caller has cleared the viewport.
    pub fn draw(&mut self, gl: &glow::Context, renderer: &Renderer, camera: &Camera) -> Stats {
        self.cache.begin_frame();
        self.needed.clear();
        self.grid.select(camera, &mut self.needed);
        self.cache.plan(COUNTIES, &self.grid, &self.needed, &mut self.plan);
        let prepared = self.plan.fetch.len();
        for &id in &self.plan.fetch {
            let mesh = self.counties.prepare(&self.grid, id);
            let gpu = renderer.upload(gl, &mesh).unwrap();
            for (_, evicted) in self.cache.insert(TileKey { source: COUNTIES, tile: id }, gpu, mesh.byte_size()) {
                renderer.free(gl, evicted);
            }
        }
        if prepared > 0 {
            // Everything asked for has arrived; plan again to draw it.
            self.cache.plan(COUNTIES, &self.grid, &self.needed, &mut self.plan);
        }
        for &id in &self.plan.draw {
            let mesh = self.cache.get(TileKey { source: COUNTIES, tile: id }).unwrap();
            renderer.draw(gl, mesh, camera, &data::PALETTE, data::OUTLINE);
        }
        Stats {
            drawn: self.plan.draw.len(),
            prepared,
            level: self.needed.first().map_or(0, |t| t.z),
            cached: self.cache.len(),
            cached_bytes: self.cache.used(),
        }
    }
}
