//! Nazca -- a rendering library for geospatial data: interactive maps and
//! static layouts from one pipeline, natively and in WebAssembly. See
//! `../project.org` for scope and `../plans/rendering.org` for the design.
//!
//! Geometry reaches nazca through a [`Sink`]: either nazca's own streaming
//! [`wkb`] reader drives one, or a caller's adapter over any other source
//! does. A [`TileGrid`] cuts the map's plane into the tiles it is prepared
//! and drawn in.
//!
//! Everything but the `gl` backend is free of `unsafe`. `glow`'s calls are
//! all `unsafe fn`, so `gl` alone is allowed it; `forbid` cannot be lifted for
//! one module, hence `deny`.
#![deny(unsafe_code)]

pub mod earcut;
#[cfg(feature = "gl")]
pub mod gl;
pub mod grid;
pub mod prepare;
pub mod sink;
pub mod wkb;

pub use grid::{TileGrid, TileId};
pub use prepare::{Detail, Mesh, MeshBuilder};
pub use sink::Sink;
