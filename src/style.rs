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
//!
//! A class can also draw with symbols (`../plans/custom-symbols.org`): a
//! [`Marker`] in place of a point's dot, a [`Dash`] pattern on its lines and
//! outlines, a [`Pattern`] in place of its flat fill. They name a symbol by
//! [`SymbolId`], so restyling a class's markers uploads no image.
//!
//! [`Style`] and these types are `#[non_exhaustive]`, so a field added later
//! breaks no one: start from [`Style::default`] (which draws nothing),
//! [`Marker::new`], [`Dash::new`] or [`Pattern::new`] and assign the fields
//! wanted.

use crate::symbol::{SymbolId, Symbols};

/// How many style classes a palette holds: a class is a `u8`.
pub const CLASSES: usize = 256;

/// The most on and off lengths a [`Dash`] holds.
pub const DASH_LENGTHS: usize = 8;

/// The look of one style class.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Style {
    /// Polygon interiors, and a mask symbol's tint.
    pub fill: [f32; 4],
    /// Polygon outlines and lines.
    pub line: [f32; 4],
    /// Line width in pixels; 0 draws no line.
    pub line_width: f32,
    /// A point's diameter in pixels, drawn as a dot in the fill colour, or
    /// its marker's longest side; 0 draws no point.
    pub point_size: f32,
    /// A symbol drawn at each point in place of the dot.
    pub marker: Option<Marker>,
    /// Dashes on lines and polygon outlines.
    pub dash: Option<Dash>,
    /// A repeated symbol in place of the flat fill.
    pub pattern: Option<Pattern>,
}

/// What a marker's rotation is against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Align {
    /// The screen: the marker keeps its rotation as the camera turns.
    #[default]
    Screen,
    /// The map: the marker turns as the camera turns, as a north arrow does.
    Map,
}

/// A symbol drawn at a point, its longest side the class's `point_size`. A
/// mask marker takes the class's `fill` colour; a colour marker's alpha is
/// multiplied by the fill's, so a class needs an opaque fill, not
/// [`Style::default`]'s transparent one, to draw it as it is. Check a
/// palette's markers against the tiles' guard band with
/// [`Palette::check_reach`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Marker {
    pub symbol: SymbolId,
    /// Degrees clockwise, as a camera's bearing grows.
    pub rotation: f32,
    pub align: Align,
}

impl Marker {
    /// `symbol` upright on the screen.
    pub fn new(symbol: SymbolId) -> Marker {
        Marker { symbol, rotation: 0.0, align: Align::Screen }
    }
}

/// On and off lengths in pixels, repeated along a line from `phase` pixels
/// into the pattern. An on length of 0 is a dot. An odd number of lengths
/// repeats twice to make an even one, as in SVG.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dash {
    lengths: [f32; DASH_LENGTHS],
    count: u8,
    pub phase: f32,
}

impl Dash {
    /// `None` for no lengths or more than [`DASH_LENGTHS`], a length that is
    /// negative or not finite, lengths that sum to 0, or a phase that is not
    /// finite.
    pub fn new(lengths: &[f32], phase: f32) -> Option<Dash> {
        let ok = |l: &f32| l.is_finite() && *l >= 0.0;
        if lengths.is_empty() || lengths.len() > DASH_LENGTHS || !lengths.iter().all(ok) || lengths.iter().sum::<f32>() <= 0.0 || !phase.is_finite() {
            return None;
        }
        let mut all = [0.0; DASH_LENGTHS];
        all[..lengths.len()].copy_from_slice(lengths);
        Some(Dash { lengths: all, count: lengths.len() as u8, phase })
    }

    /// The lengths, on first.
    pub fn lengths(&self) -> &[f32] {
        &self.lengths[..self.count as usize]
    }
}

/// A symbol repeated over a fill, anchored to the map, one repeat's longest
/// side `repeat` pixels at every zoom. A mask pattern takes the class's
/// `fill` colour.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pattern {
    pub symbol: SymbolId,
    pub repeat: f32,
}

impl Pattern {
    pub fn new(symbol: SymbolId, repeat: f32) -> Pattern {
        Pattern { symbol, repeat }
    }
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

    /// The classes that draw a symbol larger than [`crate::Symbol::sharp_px`],
    /// which blurs: as a marker of their `point_size` or a pattern of its
    /// `repeat`. An embedder adds such a symbol again at a larger size. A
    /// symbol not in `symbols` draws nothing and is not named.
    pub fn blurred(&self, symbols: &Symbols) -> Vec<u8> {
        let past = |id: SymbolId, size: f32| symbols.get(id).is_some_and(|s| size > s.sharp_px());
        let classes = self.styles.iter().enumerate();
        classes
            .filter(|(_, s)| s.marker.is_some_and(|m| past(m.symbol, s.point_size)) || s.pattern.is_some_and(|p| past(p.symbol, p.repeat)))
            .map(|(class, _)| class as u8)
            .collect()
    }

    /// Whether every marker is whole across tile edges, for tiles prepared
    /// with a guard band of `guard_px` ([`crate::Detail::guard_px`]). A
    /// marker drawn by one tile is cut at its edge and the neighbour draws
    /// the rest from its own copy of the point, so its reach from its point
    /// must lie within the guard band. A level's tiles are shown down to
    /// half their size before the next coarser level takes over, while a
    /// marker keeps its size, so the reach is checked against half the guard
    /// band.
    ///
    /// The reach is measured from the anchor. A marker against the map keeps
    /// its rotation against the tile's edges, and is measured along them at
    /// that rotation; one against the screen turns against them as the
    /// camera turns, and is measured to its farthest corner. The error names
    /// the first class that reaches too far, its size, and the `guard_px` it
    /// needs. A symbol not in `symbols` draws nothing and is not checked.
    pub fn check_reach(&self, symbols: &Symbols, guard_px: f64) -> Result<(), String> {
        for (class, s) in self.styles.iter().enumerate() {
            let Some(m) = s.marker else { continue };
            let Some(symbol) = symbols.get(m.symbol) else { continue };
            let reach = marker_reach(symbol, m, s.point_size);
            if reach > guard_px / 2.0 {
                let need = (2.0 * reach).ceil();
                return Err(format!(
                    "class {class} draws a {}-pixel marker reaching {reach:.1} pixels from its point, past half the guard band of {guard_px} pixels: it needs a guard_px of {need}",
                    s.point_size
                ));
            }
        }
        Ok(())
    }
}

/// How far a marker of `size` reaches from its point, in pixels (see
/// [`Palette::check_reach`]).
fn marker_reach(symbol: &crate::Symbol, m: Marker, size: f32) -> f64 {
    if size <= 0.0 {
        return 0.0;
    }
    let (w, h) = (symbol.width() as f64, symbol.height() as f64);
    let scale = size as f64 / w.max(h);
    let [ax, ay] = symbol.anchor.map(|a| a as f64);
    let xs = [-ax * w * scale, (1.0 - ax) * w * scale];
    let ys = [-ay * h * scale, (1.0 - ay) * h * scale];
    let corners = xs.iter().flat_map(|&x| ys.iter().map(move |&y| (x, y)));
    match m.align {
        Align::Screen => corners.map(|(x, y)| x.hypot(y)).fold(0.0, f64::max),
        Align::Map => {
            let (sin, cos) = (m.rotation as f64).to_radians().sin_cos();
            corners.map(|(x, y)| (x * cos - y * sin).abs().max((x * sin + y * cos).abs())).fold(0.0, f64::max)
        }
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
    fn dashes_hold_what_they_were_given() {
        let d = Dash::new(&[4.0, 2.0, 0.0, 2.0], 1.5).unwrap();
        assert_eq!(d.lengths(), &[4.0, 2.0, 0.0, 2.0]);
        assert_eq!(d.phase, 1.5);
        assert!(Dash::new(&[1.0; DASH_LENGTHS], 0.0).is_some());
        assert_eq!(Dash::new(&[1.0; DASH_LENGTHS + 1], 0.0), None);
        assert_eq!(Dash::new(&[], 0.0), None);
        assert_eq!(Dash::new(&[2.0, -1.0], 0.0), None);
        assert_eq!(Dash::new(&[0.0, 0.0], 0.0), None);
        assert_eq!(Dash::new(&[f32::INFINITY], 0.0), None);
        assert_eq!(Dash::new(&[1.0], f32::NAN), None);
    }

    #[test]
    fn a_palette_names_classes_drawing_a_symbol_past_its_pixels() {
        let mut symbols = Symbols::new();
        let mut pin = crate::Symbol::new(32, 48, vec![0; 32 * 48 * 4]).unwrap();
        pin.scale = 2.0;
        let pin = symbols.add(pin).unwrap();
        let mut p = Palette::new();
        let marker = Style { marker: Some(Marker::new(pin)), point_size: 24.0, ..Style::default() };
        p.set(1, marker);
        p.set(2, Style { point_size: 25.0, ..marker });
        p.set(3, Style { pattern: Some(Pattern::new(pin, 30.0)), ..Style::default() });
        assert_eq!(p.blurred(&symbols), vec![2, 3]);
        symbols.remove(pin);
        assert_eq!(p.blurred(&symbols), Vec::<u8>::new());
    }

    #[test]
    fn markers_reach_from_their_anchor_at_their_worst_rotation() {
        let mut symbols = Symbols::new();
        let square = symbols.add(crate::Symbol::new(10, 10, vec![0; 400]).unwrap()).unwrap();
        let mut pin = crate::Symbol::new(10, 20, vec![0; 800]).unwrap();
        pin.anchor = [0.5, 1.0];
        let pin = symbols.add(pin).unwrap();
        let reach = |id, align, rotation, size| {
            let marker = Marker { align, rotation, ..Marker::new(id) };
            marker_reach(symbols.get(id).unwrap(), marker, size)
        };
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        // A centred square: half its side against the map, half its
        // diagonal against the screen, as the map turns under it.
        assert!(close(reach(square, Align::Map, 0.0, 20.0), 10.0));
        assert!(close(reach(square, Align::Map, 45.0, 20.0), 200f64.sqrt()));
        assert!(close(reach(square, Align::Screen, 0.0, 20.0), 200f64.sqrt()));
        // A pin anchored at its tip, 20 tall: its whole height above the
        // point, and its top corners farther still once it can turn.
        assert!(close(reach(pin, Align::Map, 0.0, 20.0), 20.0));
        assert!(close(reach(pin, Align::Screen, 0.0, 20.0), 425f64.sqrt()));
        assert_eq!(reach(pin, Align::Map, 0.0, 0.0), 0.0);
    }

    #[test]
    fn a_palette_is_checked_against_half_the_guard_band() {
        let mut symbols = Symbols::new();
        let mut pin = crate::Symbol::new(10, 20, vec![0; 800]).unwrap();
        pin.anchor = [0.5, 1.0];
        let pin = symbols.add(pin).unwrap();
        let mut p = Palette::new();
        let marker = Marker { align: Align::Map, ..Marker::new(pin) };
        p.set(4, Style { marker: Some(marker), point_size: 16.0, ..Style::default() });
        assert_eq!(p.check_reach(&symbols, 32.0), Ok(()));
        p.set(5, Style { marker: Some(marker), point_size: 17.0, ..Style::default() });
        let err = p.check_reach(&symbols, 32.0).unwrap_err();
        assert!(err.starts_with("class 5 draws a 17-pixel marker"), "{err}");
        assert!(err.ends_with("guard_px of 34"), "{err}");
        assert_eq!(p.check_reach(&symbols, 34.0), Ok(()));
        // A dot is no marker, and a missing symbol draws nothing.
        symbols.remove(pin);
        assert_eq!(p.check_reach(&symbols, 0.0), Ok(()));
    }

    #[test]
    fn zoom_curves_interpolate_like_maplibre() {
        // City circles: radius 1 at z3 to 5 at z10.
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
