//! The geometry sink: how coordinates reach nazca from any source.
//!
//! A reader (nazca's own [`crate::wkb`], or a caller's adapter over
//! `geo-types`, GeoArrow, `geosetta`'s IR, ...) drives a [`Sink`] with a
//! stream of events instead of building a geometry tree, so nothing about a
//! geometry is held longer than it takes to pass through. This is the same
//! "name the capability, not the provider" shape as the retired
//! `CrsRegistry` trait (`../architecture.org`).

/// Receives one geometry at a time as a stream of events.
///
/// The event grammar, for one geometry:
///
/// ```text
/// geometry := point | line | polygon | (any of these, repeated)
/// point    := point(xy)
/// line     := begin_line points* end_line
/// polygon  := begin_polygon (begin_ring points* end_ring)* end_polygon
/// ```
///
/// - A run of coordinates may arrive split over several `points` calls; the
///   split falls wherever is convenient for the reader, never at a meaningful
///   boundary. A sink must treat consecutive calls as one run.
/// - A polygon's first ring is its exterior; any later rings are holes.
/// - Multi-geometries and collections arrive as their members in order, with
///   no event of their own: a `MultiPolygon` is just several polygons.
/// - Coordinates are X/Y in the source's own CRS. Nazca does not reproject
///   (`../project.org` § CHARTER).
///
/// The begin/end methods default to doing nothing, so a sink that only needs
/// coordinates (a bounds fold, a vertex count) implements two methods.
///
/// If a reader fails partway through a geometry, the sink has already seen
/// that geometry's earlier events; the caller should discard what the sink
/// built for it.
pub trait Sink {
    /// A single point (a `Point`, or one member of a `MultiPoint`).
    fn point(&mut self, xy: [f64; 2]);

    /// Some of the current line's or ring's coordinates, in order.
    fn points(&mut self, xy: &[[f64; 2]]);

    fn begin_line(&mut self) {}
    fn end_line(&mut self) {}
    fn begin_polygon(&mut self) {}
    fn end_polygon(&mut self) {}
    fn begin_ring(&mut self) {}
    fn end_ring(&mut self) {}
}
