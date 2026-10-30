//! The camera: what part of the map plane a viewport shows, and the matrix
//! that draws a tile there.
//!
//! Pure math, with no event handling: the embedder calls [`Camera::pan_px`],
//! [`Camera::zoom_about`] and [`Camera::rotate_about`] from its own input
//! events, and [`Camera::fit_bounds`] to frame a dataset
//! (`../plans/rendering.org` § CAMERA).
//!
//! Positions on screen are in the viewport's pixels -- device pixels, if the
//! embedder sized the viewport in them -- with `[0, 0]` at the top-left
//! corner and Y running down, as pointer events report them. Map positions
//! are in map units with Y up (`crate::grid`).
//!
//! The matrix is 4x4 although v1 only ever draws a flat orthographic view:
//! a perspective camera for the globe (§ GLOBE) then changes what fills the
//! matrix, not the backends or the shaders that take it.

use crate::grid::{Rect, TileFrame};
use std::f64::consts::PI;

/// A flat view of the map plane. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// The map point at the centre of the viewport.
    pub center: [f64; 2],
    /// How many map units one pixel spans.
    pub units_per_px: f64,
    /// The map direction that points up on screen, in radians clockwise from
    /// map north (+Y), as MapLibre's bearing is: at `PI / 2` east is up.
    /// Kept in `(-PI, PI]` by the methods that change it.
    pub bearing: f64,
    /// The viewport's size in pixels.
    pub size_px: [u32; 2],
}

impl Camera {
    /// A camera at `center`, `units_per_px`, north up.
    pub fn new(center: [f64; 2], units_per_px: f64, size_px: [u32; 2]) -> Camera {
        Camera { center, units_per_px, bearing: 0.0, size_px }
    }

    /// The viewport's new size. The map point at its centre and the scale stay.
    pub fn resize(&mut self, size_px: [u32; 2]) {
        self.size_px = size_px;
    }

    /// Moves the map by `delta` pixels on screen, as a drag does: the map
    /// point that was under the pointer stays under it.
    pub fn pan_px(&mut self, delta: [f64; 2]) {
        let d = self.unrotate([delta[0] * self.units_per_px, -delta[1] * self.units_per_px]);
        self.center = [self.center[0] - d[0], self.center[1] - d[1]];
    }

    /// Zooms in by `factor` (above 1 zooms in, below 1 out) keeping the map
    /// point under `px` where it is. A factor that is not a positive finite
    /// number does nothing.
    pub fn zoom_about(&mut self, px: [f64; 2], factor: f64) {
        if !(factor.is_finite() && factor > 0.0) {
            return;
        }
        let anchor = self.screen_to_map(px);
        self.units_per_px /= factor;
        self.center = [
            anchor[0] + (self.center[0] - anchor[0]) / factor,
            anchor[1] + (self.center[1] - anchor[1]) / factor,
        ];
    }

    /// Turns the map by `radians` (positive is clockwise on screen, the way
    /// the bearing grows) about the map point under `px`, which stays where
    /// it is.
    pub fn rotate_about(&mut self, px: [f64; 2], radians: f64) {
        if !radians.is_finite() {
            return;
        }
        let anchor = self.screen_to_map(px);
        self.bearing = normalize(self.bearing + radians);
        let moved = self.screen_to_map(px);
        self.center = [self.center[0] + anchor[0] - moved[0], self.center[1] + anchor[1] - moved[1]];
    }

    /// Frames `bounds` at the current bearing, leaving `padding_px` clear on
    /// every side. A rectangle with no width and height is centred at the
    /// current scale. `false`, and nothing changed, for a rectangle that is
    /// not finite or is inverted.
    pub fn fit_bounds(&mut self, bounds: Rect, padding_px: f64) -> bool {
        let finite = bounds.min.iter().chain(&bounds.max).all(|v| v.is_finite());
        if !finite || bounds.min[0] > bounds.max[0] || bounds.min[1] > bounds.max[1] {
            return false;
        }
        self.center = [(bounds.min[0] + bounds.max[0]) / 2.0, (bounds.min[1] + bounds.max[1]) / 2.0];
        // The rectangle's extent across and down the rotated screen.
        let (w, h) = (bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1]);
        let (s, c) = (self.bearing.sin().abs(), self.bearing.cos().abs());
        let (across, down) = (w * c + h * s, w * s + h * c);
        // Padding that leaves no room is ignored rather than inverting the view.
        let room = |n: u32| {
            let n = n as f64;
            if n - 2.0 * padding_px > 0.0 { n - 2.0 * padding_px } else { n.max(1.0) }
        };
        let upp = (across / room(self.size_px[0])).max(down / room(self.size_px[1]));
        if upp > 0.0 {
            self.units_per_px = upp;
        }
        true
    }

    /// The map point under screen position `px`.
    pub fn screen_to_map(&self, px: [f64; 2]) -> [f64; 2] {
        let v = [
            (px[0] - self.size_px[0] as f64 / 2.0) * self.units_per_px,
            (self.size_px[1] as f64 / 2.0 - px[1]) * self.units_per_px,
        ];
        let d = self.unrotate(v);
        [self.center[0] + d[0], self.center[1] + d[1]]
    }

    /// Where map point `xy` falls on screen.
    pub fn map_to_screen(&self, xy: [f64; 2]) -> [f64; 2] {
        let v = self.rotate([xy[0] - self.center[0], xy[1] - self.center[1]]);
        [
            self.size_px[0] as f64 / 2.0 + v[0] / self.units_per_px,
            self.size_px[1] as f64 / 2.0 - v[1] / self.units_per_px,
        ]
    }

    /// The map points at the viewport's four corners: top-left, top-right,
    /// bottom-right, bottom-left.
    pub fn corners(&self) -> [[f64; 2]; 4] {
        let [w, h] = [self.size_px[0] as f64, self.size_px[1] as f64];
        [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]].map(|px| self.screen_to_map(px))
    }

    /// The smallest map-aligned rectangle holding the whole viewport. When
    /// the map is rotated it is larger than what is visible.
    pub fn visible_bounds(&self) -> Rect {
        let c = self.corners();
        let xs = c.map(|p| p[0]);
        let ys = c.map(|p| p[1]);
        Rect {
            min: [xs.iter().copied().fold(f64::INFINITY, f64::min), ys.iter().copied().fold(f64::INFINITY, f64::min)],
            max: [xs.iter().copied().fold(f64::NEG_INFINITY, f64::max), ys.iter().copied().fold(f64::NEG_INFINITY, f64::max)],
        }
    }

    /// The matrix taking `frame`'s tile-local coordinates to clip space,
    /// column-major, for a shader's `mat4`.
    ///
    /// Composed in `f64` and only then cast: the tile's offset from the
    /// camera is a difference of two large map coordinates, which must
    /// cancel before precision is lost (`crate::grid::TileFrame`).
    pub fn matrix(&self, frame: &TileFrame) -> [f32; 16] {
        let kx = 2.0 / (self.units_per_px * self.size_px[0] as f64);
        let ky = 2.0 / (self.units_per_px * self.size_px[1] as f64);
        let (s, c) = self.bearing.sin_cos();
        let hs = frame.half_span;
        let t = self.rotate([frame.center[0] - self.center[0], frame.center[1] - self.center[1]]);
        let m = [
            kx * hs * c, ky * hs * s, 0.0, 0.0,
            -kx * hs * s, ky * hs * c, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            kx * t[0], ky * t[1], 0.0, 1.0,
        ];
        m.map(|v| v as f32)
    }

    /// A map-unit vector to the screen's axes (Y still up).
    fn rotate(&self, d: [f64; 2]) -> [f64; 2] {
        let (s, c) = self.bearing.sin_cos();
        [c * d[0] - s * d[1], s * d[0] + c * d[1]]
    }

    /// The inverse of [`Camera::rotate`].
    fn unrotate(&self, v: [f64; 2]) -> [f64; 2] {
        let (s, c) = self.bearing.sin_cos();
        [c * v[0] + s * v[1], -s * v[0] + c * v[1]]
    }
}

/// An angle in `(-PI, PI]`.
fn normalize(a: f64) -> f64 {
    let a = a.rem_euclid(2.0 * PI);
    if a > PI { a - 2.0 * PI } else { a }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f64; 2], b: [f64; 2], tol: f64) -> bool {
        (a[0] - b[0]).abs() <= tol && (a[1] - b[1]).abs() <= tol
    }

    fn camera() -> Camera {
        Camera { center: [1000.0, 2000.0], units_per_px: 2.0, bearing: 0.3, size_px: [800, 600] }
    }

    /// Clip space for a map point through `matrix`, via a frame centred on it.
    fn clip_of(cam: &Camera, frame: &TileFrame, xy: [f64; 2]) -> [f64; 2] {
        let m = cam.matrix(frame).map(|v| v as f64);
        let l = frame.to_local_f64(xy);
        [m[0] * l[0] + m[4] * l[1] + m[12], m[1] * l[0] + m[5] * l[1] + m[13]]
    }

    #[test]
    fn north_up_screen_and_map_agree() {
        let cam = Camera::new([10.0, 20.0], 0.5, [200, 100]);
        assert_eq!(cam.screen_to_map([100.0, 50.0]), [10.0, 20.0]);
        // Right is +X, down is -Y.
        assert_eq!(cam.screen_to_map([200.0, 100.0]), [60.0, -5.0]);
        assert_eq!(cam.map_to_screen([60.0, -5.0]), [200.0, 100.0]);
        assert_eq!(cam.visible_bounds(), Rect { min: [-40.0, -5.0], max: [60.0, 45.0] });
    }

    #[test]
    fn a_bearing_turns_that_direction_up() {
        let mut cam = Camera::new([0.0, 0.0], 1.0, [100, 100]);
        cam.bearing = PI / 2.0;
        // East is up: the map point 10 units east sits 10 pixels above centre.
        assert!(close(cam.map_to_screen([10.0, 0.0]), [50.0, 40.0], 1e-9));
        // And north is to the left.
        assert!(close(cam.map_to_screen([0.0, 10.0]), [40.0, 50.0], 1e-9));
    }

    #[test]
    fn screen_and_map_round_trip_at_any_bearing() {
        for b in [-3.0, -1.0, 0.0, 0.3, 2.0, PI] {
            let cam = Camera { bearing: b, ..camera() };
            for px in [[0.0, 0.0], [800.0, 600.0], [123.4, 567.8]] {
                assert!(close(cam.map_to_screen(cam.screen_to_map(px)), px, 1e-9), "{b} {px:?}");
            }
        }
    }

    #[test]
    fn panning_keeps_the_dragged_point_under_the_pointer() {
        let mut cam = camera();
        let p = [300.0, 200.0];
        let under = cam.screen_to_map(p);
        cam.pan_px([25.0, -40.0]);
        assert!(close(cam.map_to_screen(under), [325.0, 160.0], 1e-9));
    }

    #[test]
    fn zooming_and_rotating_keep_the_anchor_fixed() {
        let mut cam = camera();
        let p = [700.0, 50.0];
        let under = cam.screen_to_map(p);
        cam.zoom_about(p, 2.5);
        assert!((cam.units_per_px - 0.8).abs() < 1e-12);
        assert!(close(cam.screen_to_map(p), under, 1e-9));
        cam.rotate_about(p, 1.0);
        assert!((cam.bearing - 1.3).abs() < 1e-12);
        assert!(close(cam.screen_to_map(p), under, 1e-9));
        // A factor that cannot be a zoom changes nothing.
        let before = cam;
        cam.zoom_about(p, 0.0);
        cam.zoom_about(p, f64::NAN);
        cam.rotate_about(p, f64::INFINITY);
        assert_eq!(cam, before);
    }

    #[test]
    fn the_bearing_stays_in_range() {
        let mut cam = camera();
        cam.rotate_about([0.0, 0.0], 7.0 * PI);
        assert!(cam.bearing > -PI && cam.bearing <= PI);
        assert!((normalize(-PI) - PI).abs() < 1e-12);
    }

    #[test]
    fn fitting_bounds_shows_all_of_them() {
        let r = Rect { min: [100.0, 100.0], max: [500.0, 200.0] };
        for b in [0.0, 0.5, PI / 2.0, -2.0] {
            let mut cam = Camera { bearing: b, ..camera() };
            assert!(cam.fit_bounds(r, 20.0));
            for corner in [r.min, r.max, [r.min[0], r.max[1]], [r.max[0], r.min[1]]] {
                let s = cam.map_to_screen(corner);
                assert!(s[0] >= 20.0 - 1e-6 && s[0] <= 780.0 + 1e-6 && s[1] >= 20.0 - 1e-6 && s[1] <= 580.0 + 1e-6, "{b} {s:?}");
            }
        }
        // North up, the wide rectangle fills the padded width exactly.
        let mut cam = camera();
        cam.bearing = 0.0;
        cam.fit_bounds(r, 20.0);
        assert!((cam.units_per_px - 400.0 / 760.0).abs() < 1e-12);
        // A point keeps the scale; a NaN or inverted rectangle changes nothing.
        let mut cam = camera();
        assert!(cam.fit_bounds(Rect { min: [5.0, 6.0], max: [5.0, 6.0] }, 0.0));
        assert_eq!((cam.center, cam.units_per_px), ([5.0, 6.0], 2.0));
        assert!(!cam.fit_bounds(Rect { min: [f64::NAN, 0.0], max: [1.0, 1.0] }, 0.0));
        assert!(!cam.fit_bounds(Rect { min: [2.0, 0.0], max: [1.0, 1.0] }, 0.0));
        assert_eq!(cam.center, [5.0, 6.0]);
    }

    #[test]
    fn the_matrix_puts_map_points_where_the_screen_says() {
        let cam = camera();
        let frame = TileFrame { center: [1100.0, 1900.0], half_span: 64.0 };
        for px in [[0.0, 0.0], [800.0, 600.0], [400.0, 300.0], [10.0, 590.0]] {
            let clip = clip_of(&cam, &frame, cam.screen_to_map(px));
            let want = [2.0 * px[0] / 800.0 - 1.0, 1.0 - 2.0 * px[1] / 600.0];
            assert!(close(clip, want, 1e-5), "{px:?}: {clip:?} vs {want:?}");
        }
    }

    #[test]
    fn the_matrix_keeps_street_level_precision_far_from_the_origin() {
        // Web Mercator's far east at a tenth of a metre per pixel: raw f32
        // map coordinates would step in metres here.
        let cam = Camera { center: [20_000_000.0, -3_000_000.0], units_per_px: 0.1, bearing: 0.7, size_px: [1024, 768] };
        let frame = TileFrame { center: [20_000_010.0, -3_000_005.0], half_span: 19.0 };
        let p = [20_000_012.345, -3_000_004.321];
        let clip = clip_of(&cam, &frame, p);
        let s = cam.map_to_screen(p);
        let want = [2.0 * s[0] / 1024.0 - 1.0, 1.0 - 2.0 * s[1] / 768.0];
        // Within a hundredth of a pixel.
        assert!(close(clip, want, 2.0 * 0.01 / 768.0), "{clip:?} vs {want:?}");
    }
}
