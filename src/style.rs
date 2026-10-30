//! How prepared geometry and rasters look: a palette of styles indexed by
//! style class, and colormaps for raster bands (`../plans/rendering.org` §
//! STYLING, § RASTER).
//!
//! Prepared tiles carry only a style class per vertex, never a colour, so
//! restyling rewrites a palette or a colormap and re-prepares nothing. Both
//! are small: a backend keeps them as lookup textures.
//!
//! Colours are RGBA in 0..1 with straight (not premultiplied) alpha, as
//! written into an ordinary framebuffer; opacity is the alpha.

/// How many style classes a palette holds: a class is a `u8`.
pub const CLASSES: usize = 256;

/// The look of one style class.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Style {
    /// Polygon interiors.
    pub fill: [f32; 4],
    /// Polygon outlines and lines.
    pub line: [f32; 4],
    /// Line width in pixels; 0 draws no line.
    pub line_width: f32,
    /// A point's diameter in pixels, drawn as a dot in the fill colour; 0
    /// draws no point.
    pub point_size: f32,
}

/// A style per class. Classes never set are [`Style::default`]: nothing
/// drawn, though a fill still picks.
#[derive(Debug, Clone, PartialEq)]
pub struct Palette {
    styles: Vec<Style>,
}

impl Default for Palette {
    fn default() -> Palette {
        Palette { styles: vec![Style::default(); CLASSES] }
    }
}

impl Palette {
    pub fn new() -> Palette {
        Palette::default()
    }

    pub fn set(&mut self, class: u8, style: Style) {
        self.styles[class as usize] = style;
    }

    pub fn get(&self, class: u8) -> &Style {
        &self.styles[class as usize]
    }

    /// All [`CLASSES`] styles, in class order.
    pub fn styles(&self) -> &[Style] {
        &self.styles
    }
}

/// A value that follows the zoom: MapLibre's `["interpolate", ["linear" |
/// ["exponential", base]], ["zoom"], z0, v0, z1, v1, ...]`, over `stops` of
/// `(zoom, value)` in ascending zoom. `base` 1 is linear; above 1 the value
/// changes faster at the high end of each stop. Before the first stop and
/// after the last the end value holds. Zoom comes from
/// [`crate::TileGrid::fractional_level`].
///
/// Styles stay plain values (`../plans/rendering.org` § STYLING): an
/// embedder evaluates its curves at the camera's zoom when the zoom changes
/// and rewrites the palette, one small upload.
pub fn interpolate(stops: &[(f64, f32)], base: f64, zoom: f64) -> f32 {
    let Some(&(z_first, v_first)) = stops.first() else { return 0.0 };
    if zoom <= z_first {
        return v_first;
    }
    for w in stops.windows(2) {
        let ((z0, v0), (z1, v1)) = (w[0], w[1]);
        if zoom <= z1 {
            let span = z1 - z0;
            let t = if span <= 0.0 {
                1.0
            } else if (base - 1.0).abs() < 1e-12 {
                (zoom - z0) / span
            } else {
                (base.powf(zoom - z0) - 1.0) / (base.powf(span) - 1.0)
            };
            return v0 + (v1 - v0) * t as f32;
        }
    }
    stops[stops.len() - 1].1
}

/// The lookup that colours a raster band: 256 RGBA8 entries, indexed by the
/// value for a categorical band, or spread evenly over `domain` for a
/// continuous one.
#[derive(Debug, Clone, PartialEq)]
pub struct Colormap {
    pub entries: [[u8; 4]; 256],
    /// The continuous values the first and last entries stand for. Values
    /// outside it take the end colours.
    pub domain: [f32; 2],
}

impl Colormap {
    /// A categorical colormap: each listed value its colour, every other
    /// value transparent (which is how a nodata class is left out).
    pub fn categorical(classes: &[(u8, [u8; 4])]) -> Colormap {
        let mut entries = [[0; 4]; 256];
        for &(value, colour) in classes {
            entries[value as usize] = colour;
        }
        Colormap { entries, domain: [0.0, 255.0] }
    }

    /// A continuous ramp through `stops`, `(value, colour)` in ascending
    /// value order, interpolated linearly per channel. `None` for no stops,
    /// a value that is not finite, or values out of order.
    pub fn ramp(stops: &[(f32, [u8; 4])]) -> Option<Colormap> {
        let (first, last) = (stops.first()?, stops.last()?);
        if stops.iter().any(|s| !s.0.is_finite()) || stops.windows(2).any(|w| w[0].0 > w[1].0) {
            return None;
        }
        let domain = [first.0, last.0];
        let mut entries = [[0; 4]; 256];
        for (k, entry) in entries.iter_mut().enumerate() {
            let v = domain[0] + (domain[1] - domain[0]) * k as f32 / 255.0;
            // The first stop at or past v, and the one before it.
            let i = stops.iter().position(|s| s.0 >= v).unwrap_or(stops.len() - 1);
            *entry = if i == 0 {
                stops[0].1
            } else {
                let (a, b) = (stops[i - 1], stops[i]);
                let t = if b.0 > a.0 { (v - a.0) / (b.0 - a.0) } else { 1.0 };
                std::array::from_fn(|c| (a.1[c] as f32 + (b.1[c] as f32 - a.1[c] as f32) * t).round() as u8)
            };
        }
        Some(Colormap { entries, domain })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_palette_starts_invisible() {
        let mut p = Palette::new();
        assert_eq!(p.styles().len(), CLASSES);
        assert_eq!(*p.get(200), Style::default());
        let s = Style { fill: [1.0, 0.0, 0.0, 1.0], line_width: 2.0, ..Style::default() };
        p.set(200, s);
        assert_eq!(*p.get(200), s);
    }

    #[test]
    fn zoom_curves_interpolate_like_maplibre() {
        // DuckMap's city circles: radius 1 at z3 to 5 at z10.
        let stops = [(3.0, 1.0), (10.0, 5.0)];
        assert_eq!(interpolate(&stops, 1.0, 0.0), 1.0);
        assert_eq!(interpolate(&stops, 1.0, 6.5), 3.0);
        assert_eq!(interpolate(&stops, 1.0, 14.0), 5.0);
        // Exponential base 2 between z0 and z2: a third of the way by z1.
        let e = interpolate(&[(0.0, 0.0), (2.0, 3.0)], 2.0, 1.0);
        assert!((e - 1.0).abs() < 1e-6, "{e}");
        assert_eq!(interpolate(&[], 1.0, 5.0), 0.0);
    }

    #[test]
    fn categorical_entries_are_the_listed_values() {
        let c = Colormap::categorical(&[(3, [1, 2, 3, 255]), (11, [9, 9, 9, 255])]);
        assert_eq!(c.entries[3], [1, 2, 3, 255]);
        assert_eq!(c.entries[11], [9, 9, 9, 255]);
        assert_eq!(c.entries[0], [0, 0, 0, 0]);
    }

    #[test]
    fn a_ramp_interpolates_between_stops() {
        let c = Colormap::ramp(&[(0.0, [0, 0, 0, 255]), (10.0, [200, 0, 0, 255]), (20.0, [200, 100, 0, 255])]).unwrap();
        assert_eq!(c.domain, [0.0, 20.0]);
        assert_eq!(c.entries[0], [0, 0, 0, 255]);
        assert_eq!(c.entries[255], [200, 100, 0, 255]);
        // Entry 51 stands for 4.0: 40% of the way to the second stop.
        assert_eq!(c.entries[51], [80, 0, 0, 255]);
        // One stop is a single colour.
        let one = Colormap::ramp(&[(5.0, [1, 2, 3, 4])]).unwrap();
        assert!(one.entries.iter().all(|&e| e == [1, 2, 3, 4]));
        assert_eq!(Colormap::ramp(&[]), None);
        assert_eq!(Colormap::ramp(&[(2.0, [0; 4]), (1.0, [0; 4])]), None);
        assert_eq!(Colormap::ramp(&[(f32::NAN, [0; 4])]), None);
    }
}
