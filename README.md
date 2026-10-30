# Nazca

A rendering library for geospatial data: interactive maps and static layouts
from one pipeline, natively and in WebAssembly. Built to be lightweight, fast
and memory-efficient.

Nazca was planned as a reprojection crate until 2026-10-22. Reprojection, and
vector/raster math generally, moved to
[Geoscribe](https://github.com/dxgeo/geoscribe), which holds the CRS database
that math needs. Its plans went with it (`geoscribe/plans/reprojection.org`,
`geoscribe/plans/transform-bounds.org`).

Like its siblings [Geosetta](https://github.com/dxgeo/geosetta) and Geoscribe,
Nazca depends on neither of them, and neither depends on it.

**Status: early.** A streaming WKB reader, the geometry `Sink` trait and tile
grids exist, and a spike draws polygons with OpenGL natively and WebGL2 in a
browser (`examples/spike`, `spike-web/`). Scope is in [`project.org`](project.org); the pipeline
design is in [`plans/rendering.org`](plans/rendering.org).

## License

MIT — see [LICENSE](LICENSE). `src/earcut.rs` is a port of
[mapbox/earcut](https://github.com/mapbox/earcut), under its ISC licence — see
[NOTICE](NOTICE).
