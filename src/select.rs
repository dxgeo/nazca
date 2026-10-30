//! Tile selection: which tiles a camera's view needs.
//!
//! Behind a trait (`../plans/rendering.org` § TILE GRIDS) because selection
//! is the part a globe changes: a flat view needs one level of one grid,
//! while a globe culls by frustum and horizon and picks each tile's level by
//! its distance. v1 ships the flat selector, implemented on [`TileGrid`]
//! itself; the globe's plugs in later beside it.

use crate::camera::Camera;
use crate::grid::{TileGrid, TileId};

/// Chooses the tiles that draw a camera's view.
pub trait TileSelector {
    /// Appends to `out` the tiles needed to draw `camera`'s view, the most
    /// important first, so an embedder fetching in order fills the middle of
    /// the screen before its edges. `out` is not cleared, so one buffer can
    /// be reused frame after frame.
    fn select(&self, camera: &Camera, out: &mut Vec<TileId>);
}

/// The flat selector: the tiles of one level that the viewport overlaps.
///
/// The level is [`TileGrid::level_for_resolution`] at the camera's scale, so
/// no tile is shown magnified until the grid runs out of levels. A rotated
/// viewport is tested exactly, not by its map-aligned bounds, so a turned
/// map does not fetch the tiles in its bounding box's corners. Tiles come
/// nearest the centre of the view first.
impl TileSelector for TileGrid {
    fn select(&self, camera: &Camera, out: &mut Vec<TileId>) {
        let z = self.level_for_resolution(camera.units_per_px);
        let start = out.len();
        let [w, h] = [camera.size_px[0] as f64, camera.size_px[1] as f64];
        for id in self.tiles_in(z, camera.visible_bounds()) {
            if camera.bearing != 0.0 {
                // The tile overlaps the map-aligned bounds; it overlaps the
                // viewport too if it also does along the screen's own axes
                // (the separating-axis test for two rectangles).
                let b = self.bounds(id).unwrap();
                let corners = [b.min, b.max, [b.min[0], b.max[1]], [b.max[0], b.min[1]]].map(|p| camera.map_to_screen(p));
                let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
                for c in corners {
                    for i in 0..2 {
                        lo[i] = lo[i].min(c[i]);
                        hi[i] = hi[i].max(c[i]);
                    }
                }
                if hi[0] <= 0.0 || lo[0] >= w || hi[1] <= 0.0 || lo[1] >= h {
                    continue;
                }
            }
            out.push(id);
        }
        let center = camera.center;
        let distance = |id: &TileId| {
            let c = self.bounds(*id).unwrap().center();
            (c[0] - center[0]).powi(2) + (c[1] - center[1]).powi(2)
        };
        out[start..].sort_by(|a, b| distance(a).total_cmp(&distance(b)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::WEB_MERCATOR_HALF_EXTENT as H;

    fn selected(grid: &TileGrid, camera: &Camera) -> Vec<TileId> {
        let mut out = Vec::new();
        grid.select(camera, &mut out);
        out
    }

    #[test]
    fn a_view_of_exactly_four_tiles_needs_those_four() {
        let g = TileGrid::web_mercator(256, 22).unwrap();
        // Centred on the world's centre at level 3's resolution, 512 x 512
        // pixels: the four level-3 tiles meeting at (0, 0).
        let cam = Camera::new([0.0, 0.0], g.resolution(3).unwrap(), [512, 512]);
        let mut got = selected(&g, &cam);
        got.sort();
        let t = |x, y| TileId { z: 3, x, y };
        assert_eq!(got, vec![t(3, 3), t(3, 4), t(4, 3), t(4, 4)]);
    }

    #[test]
    fn the_middle_comes_first() {
        let g = TileGrid::web_mercator(256, 22).unwrap();
        let r = g.resolution(4).unwrap();
        // Centred on a tile's centre, three tiles across and down.
        let b = g.bounds(TileId { z: 4, x: 5, y: 7 }).unwrap();
        let cam = Camera::new(b.center(), r, [768, 768]);
        let got = selected(&g, &cam);
        assert_eq!(got.len(), 9);
        assert_eq!(got[0], TileId { z: 4, x: 5, y: 7 });
        // Corner tiles are farthest, so last.
        assert!(got[5..].iter().all(|t| t.x != 5 && t.y != 7), "{got:?}");
    }

    #[test]
    fn a_rotated_view_skips_its_bounding_boxs_corners() {
        let g = TileGrid::web_mercator(256, 22).unwrap();
        let r = g.resolution(6).unwrap();
        let b = g.bounds(TileId { z: 6, x: 20, y: 20 }).unwrap();
        // A long thin view turned 45 degrees: its bounding box is a large
        // square, most of which the view does not cover.
        let mut cam = Camera::new(b.center(), r, [4096, 64]);
        cam.bearing = std::f64::consts::FRAC_PI_4;
        let in_bounds = g.tiles_in(6, cam.visible_bounds()).count();
        let got = selected(&g, &cam);
        assert!(got.len() * 3 < in_bounds, "{} of {in_bounds}", got.len());
        // Every tile the view's centre line crosses is still there.
        for i in 0..=40 {
            let p = cam.screen_to_map([i as f64 * 4096.0 / 40.0, 32.0]);
            let t = g.tiles_in(6, crate::grid::Rect { min: p, max: p }).next().unwrap();
            assert!(got.contains(&t), "{t:?}");
        }
    }

    #[test]
    fn levels_follow_the_scale_and_stop_at_the_grid() {
        let g = TileGrid::web_mercator(256, 5).unwrap();
        let z = |upp| selected(&g, &Camera::new([0.0, 0.0], upp, [300, 300]))[0].z;
        assert_eq!(z(g.resolution(2).unwrap() * 1.5), 2);
        assert_eq!(z(g.resolution(2).unwrap() * 0.9), 3);
        // Past the finest level: the finest, magnified.
        assert_eq!(z(1e-3), 5);
        // Zoomed out past the whole world: level 0, the one tile.
        assert_eq!(selected(&g, &Camera::new([0.0, 0.0], H, [300, 300])), vec![TileId { z: 0, x: 0, y: 0 }]);
    }

    #[test]
    fn the_buffer_is_appended_to() {
        let g = TileGrid::web_mercator(256, 22).unwrap();
        let cam = Camera::new([0.0, 0.0], g.resolution(3).unwrap(), [512, 512]);
        let mut out = vec![TileId { z: 9, x: 9, y: 9 }];
        g.select(&cam, &mut out);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0], TileId { z: 9, x: 9, y: 9 });
    }
}
