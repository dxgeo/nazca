//! Nazca -- a rendering library for geospatial data: interactive maps and
//! static layouts from one pipeline, natively and in WebAssembly. See
//! `../project.org` for scope and `../plans/rendering.org` for the design.
//!
//! Geometry reaches nazca through a [`Sink`]: either nazca's own streaming
//! [`wkb`] reader drives one, or a caller's adapter over any other source
//! does. A [`TileGrid`] cuts the map's plane into the tiles it is prepared
//! and drawn in; a [`Camera`] says which part of the plane is on screen, a
//! [`TileSelector`] which tiles that needs, and a [`TileCache`] keeps the
//! prepared ones between frames. A [`Palette`] styles them by class, and a
//! [`Colormap`] colours [`raster`] bands. The `gl` backend draws them
//! interactively and [`svg`] writes layouts; the `mvt` and `mlt` features
//! decode basemap vector tiles into a [`Sink`].
//!
//! Everything but the `gl` backend is free of `unsafe`. `glow`'s calls are
//! all `unsafe fn`, so `gl` alone is allowed it; `forbid` cannot be lifted for
//! one module, hence `deny`.
#![deny(unsafe_code)]

pub mod cache;
pub mod camera;
pub mod earcut;
#[cfg(feature = "gl")]
pub mod gl;
pub mod grid;
#[cfg(feature = "mlt")]
pub mod mlt;
#[cfg(feature = "mvt")]
pub mod mvt;
pub mod prepare;
pub mod raster;
pub mod select;
pub mod sink;
pub mod style;
pub mod svg;
pub mod wkb;

pub use cache::{Plan, TileCache, TileKey};
pub use camera::Camera;
pub use grid::{TileGrid, TileId};
pub use prepare::{Detail, Mesh, MeshBuilder};
pub use select::TileSelector;
pub use sink::Sink;
pub use style::{Colormap, Palette, Style};
