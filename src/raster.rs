//! Raster tiles: a band of values, and the mesh it is drawn over
//! (`../plans/rendering.org` § RASTER).
//!
//! A band is stored raw -- one byte a pixel for a categorical band, an `f32`
//! for a continuous one -- and coloured at draw time through a
//! [`crate::style::Colormap`], so a colormap change is instant and a `u8`
//! band costs a quarter of RGBA. Pre-rendered RGBA tiles, such as a
//! third-party basemap's, are kept as they are.
//!
//! *Mesh warp.* An image in one CRS shown in another needs warping, which is
//! CRS math and so the caller's, in time through `geoscribe`, which plans it
//! as of 0.5.1 (`../project.org` § CHARTER). The caller projects a coarse control grid -- say 16 x 16 points
//! -- and [`RasterMesh::warped`] stretches the image over it; the GPU
//! interpolates between control points, a fraction of a pixel off for most
//! projections at tile scale. An image already in the map's CRS is
//! [`RasterMesh::rect`].

use crate::grid::{Rect, TileFrame};

/// One raster band's pixels, row by row from the top.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Band<'a> {
    /// Class values, one byte a pixel, coloured by colormap entry.
    Categorical(&'a [u8]),
    /// Measurements, coloured along the colormap's domain. NaN, and
    /// `nodata` if given, draw nothing.
    Continuous { values: &'a [f32], nodata: Option<f32> },
    /// Already coloured: four bytes a pixel, straight alpha.
    Rgba(&'a [u8]),
}

/// A band and its size in pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Raster<'a> {
    width: u32,
    height: u32,
    band: Band<'a>,
}

/// Why a [`Raster`] or a [`RasterMesh`] could not be built.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The band holds a different number of pixels than its size says.
    /// Carries the length expected and the length given.
    Length { expected: usize, got: usize },
    /// No pixels.
    Empty,
    /// A control grid under 2 x 2, or with a different number of points
    /// than its size says.
    Grid,
    /// A coordinate that is not finite.
    NotFinite,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Length { expected, got } => write!(f, "a raster band of {got} values where its size needs {expected}"),
            Error::Empty => write!(f, "a raster with no pixels"),
            Error::Grid => write!(f, "a raster control grid needs at least 2 x 2 points, and exactly as many as its size"),
            Error::NotFinite => write!(f, "a raster control point that is not finite"),
        }
    }
}

impl std::error::Error for Error {}

impl<'a> Raster<'a> {
    pub fn new(width: u32, height: u32, band: Band<'a>) -> Result<Raster<'a>, Error> {
        if width == 0 || height == 0 {
            return Err(Error::Empty);
        }
        let pixels = width as usize * height as usize;
        let (expected, got) = match band {
            Band::Categorical(v) => (pixels, v.len()),
            Band::Continuous { values, .. } => (pixels, values.len()),
            Band::Rgba(v) => (pixels * 4, v.len()),
        };
        if expected != got {
            return Err(Error::Length { expected, got });
        }
        Ok(Raster { width, height, band })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn band(&self) -> Band<'a> {
        self.band
    }
}

/// The triangles a raster is drawn over, in one tile's local coordinates,
/// with each vertex's place in the image.
#[derive(Debug, Clone, PartialEq)]
pub struct RasterMesh {
    pub frame: TileFrame,
    /// Four `f32` a vertex: tile-local X and Y, then the image position U
    /// (0 at the left edge, 1 at the right) and V (0 at the top, 1 at the
    /// bottom).
    pub vertices: Vec<f32>,
    /// Three vertex indices a triangle.
    pub triangles: Vec<u32>,
}

impl RasterMesh {
    /// An image covering `bounds` in map units, unwarped: two triangles.
    pub fn rect(frame: TileFrame, bounds: Rect) -> RasterMesh {
        let points = [[bounds.min[0], bounds.max[1]], bounds.max, bounds.min, [bounds.max[0], bounds.min[1]]];
        RasterMesh::grid(frame, 2, 2, &points)
    }

    /// An image stretched over a control grid of `cols` x `rows` points, row
    /// by row from the top. Point `(i, j)` is the map position of the image
    /// at `i / (cols - 1)` of its width and `j / (rows - 1)` of its height,
    /// measured from the top-left corner of its top-left pixel: the first
    /// point is that corner and the last the bottom-right corner of the
    /// bottom-right pixel.
    pub fn warped(frame: TileFrame, cols: u32, rows: u32, points: &[[f64; 2]]) -> Result<RasterMesh, Error> {
        if cols < 2 || rows < 2 || points.len() != cols as usize * rows as usize {
            return Err(Error::Grid);
        }
        if points.iter().any(|p| !(p[0].is_finite() && p[1].is_finite())) {
            return Err(Error::NotFinite);
        }
        Ok(RasterMesh::grid(frame, cols, rows, points))
    }

    fn grid(frame: TileFrame, cols: u32, rows: u32, points: &[[f64; 2]]) -> RasterMesh {
        let mut vertices = Vec::with_capacity(points.len() * 4);
        for (k, &p) in points.iter().enumerate() {
            let (i, j) = (k as u32 % cols, k as u32 / cols);
            let l = frame.to_local(p);
            vertices.extend_from_slice(&[l[0], l[1], i as f32 / (cols - 1) as f32, j as f32 / (rows - 1) as f32]);
        }
        let mut triangles = Vec::with_capacity(((cols - 1) * (rows - 1) * 6) as usize);
        for j in 0..rows - 1 {
            for i in 0..cols - 1 {
                let a = j * cols + i;
                let (b, c, d) = (a + 1, a + cols, a + cols + 1);
                triangles.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        RasterMesh { frame, vertices, triangles }
    }

    /// The bytes its buffers hold.
    pub fn byte_size(&self) -> usize {
        4 * (self.vertices.len() + self.triangles.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: TileFrame = TileFrame { center: [0.0, 0.0], half_span: 10.0 };

    #[test]
    fn a_band_must_fill_its_size() {
        assert!(Raster::new(3, 2, Band::Categorical(&[0; 6])).is_ok());
        assert!(Raster::new(3, 2, Band::Rgba(&[0; 24])).is_ok());
        assert_eq!(
            Raster::new(3, 2, Band::Continuous { values: &[0.0; 5], nodata: None }),
            Err(Error::Length { expected: 6, got: 5 })
        );
        assert_eq!(Raster::new(0, 2, Band::Categorical(&[])), Err(Error::Empty));
    }

    #[test]
    fn a_rect_maps_the_image_corners_to_the_bounds() {
        let m = RasterMesh::rect(FRAME, Rect { min: [-10.0, -5.0], max: [10.0, 5.0] });
        let v: Vec<&[f32]> = m.vertices.chunks(4).collect();
        assert_eq!(v[0], [-1.0, 0.5, 0.0, 0.0]); // top-left
        assert_eq!(v[3], [1.0, -0.5, 1.0, 1.0]); // bottom-right
        assert_eq!(m.triangles.len(), 6);
    }

    #[test]
    fn a_warped_grid_has_two_triangles_a_cell() {
        let pts: Vec<[f64; 2]> = (0..12).map(|k| [(k % 4) as f64, -((k / 4) as f64)]).collect();
        let m = RasterMesh::warped(FRAME, 4, 3, &pts).unwrap();
        assert_eq!(m.vertices.len(), 12 * 4);
        assert_eq!(m.triangles.len(), 3 * 2 * 6);
        // The last point is the image's bottom-right corner.
        assert_eq!(&m.vertices[44..], [0.3, -0.2, 1.0, 1.0]);
        assert!(m.triangles.iter().all(|&i| i < 12));
        assert_eq!(RasterMesh::warped(FRAME, 1, 3, &pts[..3]), Err(Error::Grid));
        assert_eq!(RasterMesh::warped(FRAME, 4, 3, &pts[..11]), Err(Error::Grid));
        let mut bad = pts.clone();
        bad[5] = [f64::NAN, 0.0];
        assert_eq!(RasterMesh::warped(FRAME, 4, 3, &bad), Err(Error::NotFinite));
    }
}
