//! The GPU backend, over `glow`: WebGL2 in a browser, OpenGL 3.3 core
//! natively. Enabled by the `gl` feature (`../plans/rendering.org` §
//! BACKENDS).
//!
//! - *Fills* are the mesh's triangles, coloured by a [`Palette`] held as a
//!   lookup texture, so restyling uploads one small texture.
//! - *Lines* are uploaded as centerline segments and expanded in the vertex
//!   shader: each segment is one instanced quad around it, and the fragment
//!   shader keeps what lies within half the width of the segment, with a
//!   one-pixel antialiased edge. That draws round caps and round joins with
//!   no neighbour information, and widths stay right at every zoom. Points
//!   are the same quads around a segment of no length: dots.
//! - *Rasters* are drawn over their [`RasterMesh`], the band coloured in the
//!   shader through a [`Colormap`] texture.
//! - *Picking* draws feature ids instead of colours into a one-pixel integer
//!   framebuffer and reads it back ([`Picker`]).
//!
//! *Each pixel belongs to one tile.* Tiles overlap -- each keeps a guard
//! band past its edge, and an ancestor standing in for a missing tile
//! covers its loaded siblings too -- and anything translucent drawn twice
//! would blend twice. So each tile first *claims* the pixels of its own area
//! not yet claimed (stencil 0 -> 2), draws only there (stencil == 2), and
//! then *settles* them (-> 1). Drawn finest first, as
//! [`crate::cache::Plan::draw`] lists them, a loaded tile keeps its area and
//! a stand-in fills only the rest. A tile's claimed area is its square grown
//! by one pixel, so the rounding of two neighbours' matrices cannot open a
//! seam between them; the guard band holds the same geometry there.
//!
//! This is the one module allowed `unsafe`: every `glow` call is an
//! `unsafe fn`, because GL itself checks nothing. The crate denies `unsafe`
//! everywhere else.
#![allow(unsafe_code)]

use crate::camera::Camera;
use crate::grid::TileFrame;
use crate::prepare::Mesh;
use crate::raster::{Band, Raster, RasterMesh};
use crate::style::{CLASSES, Colormap, Palette};
use glow::HasContext;
use std::collections::HashMap;

const IDENTITY: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

/// Bytes a vertex: X, Y and the style class as `f32`, the feature id as `u32`.
const VERTEX_BYTES: usize = 16;
/// Bytes a segment instance: both ends' X and Y, the class, the feature id.
const SEGMENT_BYTES: usize = 24;

const FILL_VS: &str = r#"
layout(location = 0) in vec2 a_pos;
layout(location = 1) in float a_class;
layout(location = 2) in uint a_id;
uniform mat4 u_matrix;
uniform mat4 u_post;
flat out int v_class;
flat out uint v_id;
void main() {
    gl_Position = u_post * (u_matrix * vec4(a_pos, 0.0, 1.0));
    v_class = int(a_class + 0.5);
    v_id = a_id;
}
"#;

const FILL_FS: &str = r#"
flat in int v_class;
flat in uint v_id;
#ifdef PICK
uniform uint u_layer;
out uvec4 o_pick;
void main() {
    o_pick = uvec4(u_layer, v_id, 1u, 0u);
}
#else
uniform highp sampler2D u_palette;
out vec4 o_color;
void main() {
    o_color = texelFetch(u_palette, ivec2(v_class, 0), 0);
}
#endif
"#;

// One instance a segment; the four vertices of its quad come from
// gl_VertexID, drawn as a triangle strip. The expansion happens in the
// full viewport's pixels, before `u_post`, so a pick sees the same shapes.
const LINE_VS: &str = r#"
layout(location = 0) in vec2 a_p0;
layout(location = 1) in vec2 a_p1;
layout(location = 2) in float a_class;
layout(location = 3) in uint a_id;
uniform mat4 u_matrix;
uniform mat4 u_post;
uniform vec2 u_viewport;
uniform highp sampler2D u_palette;
uniform int u_points;
uniform float u_min_half;
flat out vec2 v_a;
flat out vec2 v_b;
flat out float v_half;
flat out vec4 v_color;
flat out uint v_id;
out vec2 v_px;
vec2 to_px(vec2 p) {
    vec4 c = u_matrix * vec4(p, 0.0, 1.0);
    return (c.xy / c.w * 0.5 + 0.5) * u_viewport;
}
void main() {
    int cls = int(a_class + 0.5);
    vec4 sizes = texelFetch(u_palette, ivec2(cls, 2), 0);
    float width = u_points != 0 ? sizes.y : sizes.x;
    v_color = texelFetch(u_palette, ivec2(cls, u_points != 0 ? 0 : 1), 0);
    v_half = width > 0.0 ? max(width * 0.5, u_min_half) : 0.0;
    vec2 a = to_px(a_p0);
    vec2 b = to_px(a_p1);
    vec2 d = b - a;
    float len = length(d);
    vec2 dir = len > 1e-6 ? d / len : vec2(1.0, 0.0);
    vec2 n = vec2(-dir.y, dir.x);
    // A pixel past the half-width leaves room for the antialiased edge;
    // no width, no quad.
    float r = v_half > 0.0 ? v_half + 1.0 : 0.0;
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
    vec2 p = (corner.x < 0.0 ? a : b) + (dir * corner.x + n * corner.y) * r;
    v_a = a;
    v_b = b;
    v_px = p;
    v_id = a_id;
    gl_Position = u_post * vec4(p / u_viewport * 2.0 - 1.0, 0.0, 1.0);
}
"#;

const LINE_FS: &str = r#"
flat in vec2 v_a;
flat in vec2 v_b;
flat in float v_half;
flat in vec4 v_color;
flat in uint v_id;
in vec2 v_px;
float distance_to_segment() {
    vec2 ab = v_b - v_a;
    vec2 ap = v_px - v_a;
    float l = dot(ab, ab);
    float t = l > 0.0 ? clamp(dot(ap, ab) / l, 0.0, 1.0) : 0.0;
    return length(ap - ab * t);
}
#ifdef PICK
uniform uint u_layer;
out uvec4 o_pick;
void main() {
    if (distance_to_segment() > v_half) discard;
    o_pick = uvec4(u_layer, v_id, 1u, 0u);
}
#else
out vec4 o_color;
void main() {
    float cover = clamp(v_half + 0.5 - distance_to_segment(), 0.0, 1.0);
    if (cover <= 0.0) discard;
    o_color = vec4(v_color.rgb, v_color.a * cover);
}
#endif
"#;

const RASTER_VS: &str = r#"
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
uniform mat4 u_matrix;
uniform mat4 u_post;
out vec2 v_uv;
void main() {
    gl_Position = u_post * (u_matrix * vec4(a_pos, 0.0, 1.0));
    v_uv = a_uv;
}
"#;

// u_kind: 0 RGBA as is, 1 categorical (entry by value), 2 continuous (ramp
// over u_domain).
const RASTER_FS: &str = r#"
in vec2 v_uv;
uniform highp sampler2D u_band;
uniform highp sampler2D u_ramp;
uniform int u_kind;
uniform vec2 u_domain;
uniform float u_opacity;
uniform int u_has_nodata;
uniform float u_nodata;
out vec4 o_color;
void main() {
    vec4 c;
    if (u_kind == 0) {
        c = texture(u_band, v_uv);
    } else if (u_kind == 1) {
        int k = int(texture(u_band, v_uv).r * 255.0 + 0.5);
        c = texelFetch(u_ramp, ivec2(k, 0), 0);
    } else {
        float v = texture(u_band, v_uv).r;
        if (isnan(v) || (u_has_nodata != 0 && v == u_nodata)) discard;
        float span = u_domain.y - u_domain.x;
        float t = span > 0.0 ? clamp((v - u_domain.x) / span, 0.0, 1.0) : 0.0;
        c = texture(u_ramp, vec2((t * 255.0 + 0.5) / 256.0, 0.5));
    }
    o_color = vec4(c.rgb, c.a * u_opacity);
}
"#;

// Writes only stencil: no colour output at all, so it suits the colour
// framebuffer and the picker's integer one alike.
const STENCIL_VS: &str = r#"
layout(location = 0) in vec2 a_pos;
uniform mat4 u_matrix;
uniform mat4 u_post;
uniform float u_grow;
void main() {
    gl_Position = u_post * (u_matrix * vec4(a_pos * u_grow, 0.0, 1.0));
}
"#;

const STENCIL_FS: &str = "void main() {}\n";

/// A linked program and its uniforms' locations.
struct Program {
    program: glow::Program,
    uniforms: HashMap<&'static str, glow::UniformLocation>,
}

impl Program {
    fn u(&self, name: &str) -> Option<&glow::UniformLocation> {
        self.uniforms.get(name)
    }
}

/// Whether a draw goes to the screen or to the picker.
#[derive(Clone, Copy)]
enum Mode {
    Screen,
    Pick { post: [f32; 16], layer: u32, min_half: f32 },
}

impl Mode {
    fn post(&self) -> [f32; 16] {
        match self {
            Mode::Screen => IDENTITY,
            Mode::Pick { post, .. } => *post,
        }
    }
}

/// The compiled shaders and the shared unit square, built once per GL
/// context.
pub struct Renderer {
    fill: Program,
    fill_pick: Program,
    line: Program,
    line_pick: Program,
    raster: Program,
    stencil: Program,
    square_vao: glow::VertexArray,
    square_vbo: glow::Buffer,
    float_linear: bool,
}

/// A [`Mesh`] uploaded to the GPU. Nazca keeps no CPU copy once it is here
/// (`../plans/rendering.org` § TILE CACHE); [`GpuMesh::free`] releases it.
pub struct GpuMesh {
    frame: TileFrame,
    fill_vao: glow::VertexArray,
    line_vao: glow::VertexArray,
    point_vao: glow::VertexArray,
    buffers: [glow::Buffer; 3],
    fill_count: i32,
    line_count: i32,
    point_count: i32,
    bytes: usize,
}

/// A raster band and its mesh, uploaded. [`GpuRaster::free`] releases it.
pub struct GpuRaster {
    frame: TileFrame,
    vao: glow::VertexArray,
    buffers: [glow::Buffer; 2],
    texture: glow::Texture,
    index_count: i32,
    kind: i32,
    nodata: Option<f32>,
    bytes: usize,
}

/// A [`Palette`] as a lookup texture: 256 x 3 RGBA32F, a column per class,
/// rows fill colour, line colour, then line width and point size.
pub struct GpuPalette {
    texture: glow::Texture,
}

/// A [`Colormap`] as a 256 x 1 RGBA8 lookup texture.
pub struct GpuColormap {
    texture: glow::Texture,
    domain: [f32; 2],
}

/// What a pick found: the layer number the embedder drew it under, and the
/// feature id prepare was given ([`crate::MeshBuilder::set_feature`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pick {
    pub layer: u32,
    pub feature: u32,
}

impl GpuMesh {
    /// The GPU memory it holds, the size to give [`crate::TileCache::insert`].
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn free(self, gl: &glow::Context) {
        // SAFETY: deletes objects this mesh created and alone refers to.
        unsafe {
            for vao in [self.fill_vao, self.line_vao, self.point_vao] {
                gl.delete_vertex_array(vao);
            }
            for b in self.buffers {
                gl.delete_buffer(b);
            }
        }
    }
}

impl GpuRaster {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn free(self, gl: &glow::Context) {
        // SAFETY: deletes objects this raster created and alone refers to.
        unsafe {
            gl.delete_vertex_array(self.vao);
            for b in self.buffers {
                gl.delete_buffer(b);
            }
            gl.delete_texture(self.texture);
        }
    }
}

impl GpuPalette {
    pub fn free(self, gl: &glow::Context) {
        // SAFETY: deletes the palette's own texture.
        unsafe { gl.delete_texture(self.texture) }
    }
}

impl GpuColormap {
    pub fn free(self, gl: &glow::Context) {
        // SAFETY: deletes the colormap's own texture.
        unsafe { gl.delete_texture(self.texture) }
    }
}

impl Renderer {
    pub fn new(gl: &glow::Context) -> Result<Renderer, String> {
        let embedded = gl.version().is_embedded;
        let header = if embedded { "#version 300 es\nprecision highp float;\nprecision highp int;\n" } else { "#version 330 core\n" };
        let pick = format!("{header}#define PICK\n");
        let fill_u = ["u_matrix", "u_post", "u_palette", "u_layer"];
        let line_u = ["u_matrix", "u_post", "u_viewport", "u_palette", "u_points", "u_min_half", "u_layer"];
        let raster_u = ["u_matrix", "u_post", "u_band", "u_ramp", "u_kind", "u_domain", "u_opacity", "u_has_nodata", "u_nodata"];
        let renderer = Renderer {
            fill: program(gl, header, FILL_VS, FILL_FS, &fill_u)?,
            fill_pick: program(gl, &pick, FILL_VS, FILL_FS, &fill_u)?,
            line: program(gl, header, LINE_VS, LINE_FS, &line_u)?,
            line_pick: program(gl, &pick, LINE_VS, LINE_FS, &line_u)?,
            raster: program(gl, header, RASTER_VS, RASTER_FS, &raster_u)?,
            stencil: program(gl, header, STENCIL_VS, STENCIL_FS, &["u_matrix", "u_post", "u_grow"])?,
            // SAFETY: creates and fills a buffer and array of its own; the
            // slice outlives the copy.
            square_vao: unsafe { gl.create_vertex_array()? },
            square_vbo: unsafe { gl.create_buffer()? },
            // Desktop GL filters float textures linearly; WebGL2 only with
            // the extension (which glow enables when it is offered).
            float_linear: !embedded || gl.supported_extensions().contains("OES_texture_float_linear"),
        };
        let square: [f32; 8] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        // SAFETY: binds the square's own objects and copies from a local.
        unsafe {
            gl.bind_vertex_array(Some(renderer.square_vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(renderer.square_vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &f32_bytes(&square), glow::STATIC_DRAW);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 8, 0);
            gl.bind_vertex_array(None);
        }
        Ok(renderer)
    }

    /// Uploads `mesh`: one buffer of vertices for the fill triangles' index
    /// buffer to point into, and one of segment instances, the outline
    /// segments then the points.
    pub fn upload(&self, gl: &glow::Context, mesh: &Mesh) -> Result<GpuMesh, String> {
        let mut vertices = Vec::with_capacity(mesh.vertex_count() * VERTEX_BYTES);
        for (v, id) in mesh.vertices.chunks_exact(3).zip(&mesh.ids) {
            vertices.extend_from_slice(&f32_bytes(v));
            vertices.extend_from_slice(&id.to_ne_bytes());
        }
        let line_count = mesh.outlines.len() / 2;
        let point_count = mesh.points.len();
        let mut segments = Vec::with_capacity((line_count + point_count) * SEGMENT_BYTES);
        let mut segment = |a: u32, b: u32| {
            let (a, b) = (a as usize, b as usize);
            let v = &mesh.vertices;
            segments.extend_from_slice(&f32_bytes(&[v[a * 3], v[a * 3 + 1], v[b * 3], v[b * 3 + 1], v[a * 3 + 2]]));
            segments.extend_from_slice(&mesh.ids[a].to_ne_bytes());
        };
        for s in mesh.outlines.chunks_exact(2) {
            segment(s[0], s[1]);
        }
        for &p in &mesh.points {
            segment(p, p);
        }
        let indices: Vec<u8> = mesh.fills.iter().flat_map(|i| i.to_ne_bytes()).collect();
        // SAFETY: every object is created and bound here; `buffer_data`
        // copies from slices that live for the whole call; the attribute
        // layouts match the bytes just written.
        unsafe {
            let vbo = gl.create_buffer()?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &vertices, glow::STATIC_DRAW);
            let fill_vao = gl.create_vertex_array()?;
            gl.bind_vertex_array(Some(fill_vao));
            let ebo = gl.create_buffer()?;
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
            gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, &indices, glow::STATIC_DRAW);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, VERTEX_BYTES as i32, 0);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 1, glow::FLOAT, false, VERTEX_BYTES as i32, 8);
            gl.enable_vertex_attrib_array(2);
            gl.vertex_attrib_pointer_i32(2, 1, glow::UNSIGNED_INT, VERTEX_BYTES as i32, 12);

            let sbo = gl.create_buffer()?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(sbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &segments, glow::STATIC_DRAW);
            // Two arrays over the one buffer, since WebGL2 cannot start an
            // instanced draw at an instance other than the first.
            let segment_vao = |first: usize| -> Result<glow::VertexArray, String> {
                let vao = gl.create_vertex_array()?;
                gl.bind_vertex_array(Some(vao));
                let base = (first * SEGMENT_BYTES) as i32;
                let stride = SEGMENT_BYTES as i32;
                for (loc, size, offset) in [(0, 2, 0), (1, 2, 8), (2, 1, 16)] {
                    gl.enable_vertex_attrib_array(loc);
                    gl.vertex_attrib_pointer_f32(loc, size, glow::FLOAT, false, stride, base + offset);
                    gl.vertex_attrib_divisor(loc, 1);
                }
                gl.enable_vertex_attrib_array(3);
                gl.vertex_attrib_pointer_i32(3, 1, glow::UNSIGNED_INT, stride, base + 20);
                gl.vertex_attrib_divisor(3, 1);
                Ok(vao)
            };
            let line_vao = segment_vao(0)?;
            let point_vao = segment_vao(line_count)?;
            gl.bind_vertex_array(None);
            Ok(GpuMesh {
                frame: mesh.frame,
                fill_vao,
                line_vao,
                point_vao,
                buffers: [vbo, ebo, sbo],
                fill_count: mesh.fills.len() as i32,
                line_count: line_count as i32,
                point_count: point_count as i32,
                bytes: vertices.len() + indices.len() + segments.len(),
            })
        }
    }

    /// Uploads a raster band and the mesh it is drawn over. A categorical
    /// band samples its nearest pixel, since classes cannot be blended; a
    /// continuous one blends its neighbours where the GPU can filter float
    /// textures and the band has no `nodata` value to blend into them.
    pub fn upload_raster(&self, gl: &glow::Context, mesh: &RasterMesh, raster: &Raster) -> Result<GpuRaster, String> {
        let (w, h) = (raster.width() as i32, raster.height() as i32);
        let (internal, format, ty, data, kind, nodata, linear) = match raster.band() {
            Band::Categorical(v) => (glow::R8, glow::RED, glow::UNSIGNED_BYTE, v.to_vec(), 1, None, false),
            Band::Continuous { values, nodata } => {
                (glow::R32F, glow::RED, glow::FLOAT, f32_bytes(values), 2, nodata, self.float_linear && nodata.is_none())
            }
            Band::Rgba(v) => (glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE, v.to_vec(), 0, None, true),
        };
        let vertices = f32_bytes(&mesh.vertices);
        let indices: Vec<u8> = mesh.triangles.iter().flat_map(|i| i.to_ne_bytes()).collect();
        // SAFETY: as for `upload`; the texture data is exactly `w * h`
        // pixels of `format`/`ty` (checked by `Raster::new`), read with an
        // unpack alignment of 1 so rows need no padding.
        unsafe {
            let texture = texture(gl, internal, w, h, format, ty, &data, linear)?;
            let vao = gl.create_vertex_array()?;
            gl.bind_vertex_array(Some(vao));
            let vbo = gl.create_buffer()?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &vertices, glow::STATIC_DRAW);
            let ebo = gl.create_buffer()?;
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
            gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, &indices, glow::STATIC_DRAW);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 16, 0);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 16, 8);
            gl.bind_vertex_array(None);
            Ok(GpuRaster {
                frame: mesh.frame,
                vao,
                buffers: [vbo, ebo],
                texture,
                index_count: mesh.triangles.len() as i32,
                kind,
                nodata,
                bytes: vertices.len() + indices.len() + data.len(),
            })
        }
    }

    pub fn palette(&self, gl: &glow::Context, palette: &Palette) -> Result<GpuPalette, String> {
        // SAFETY: a texture of its own, filled from a local of 256 x 3 texels.
        let texture = unsafe { texture(gl, glow::RGBA32F, CLASSES as i32, 3, glow::RGBA, glow::FLOAT, &palette_texels(palette), false)? };
        Ok(GpuPalette { texture })
    }

    /// Restyles: rewrites `gpu` from `palette`. No tile is touched.
    pub fn update_palette(&self, gl: &glow::Context, gpu: &GpuPalette, palette: &Palette) {
        let texels = palette_texels(palette);
        // SAFETY: rewrites the whole of the palette's own texture, at the
        // size it was made with, from a local.
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(gpu.texture));
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, CLASSES as i32, 3, glow::RGBA, glow::FLOAT, glow::PixelUnpackData::Slice(Some(&texels)));
        }
    }

    pub fn colormap(&self, gl: &glow::Context, colormap: &Colormap) -> Result<GpuColormap, String> {
        let texels: Vec<u8> = colormap.entries.iter().flatten().copied().collect();
        // SAFETY: a texture of its own, filled from a local of 256 texels.
        // Linear, so a continuous ramp is smooth between entries; a
        // categorical lookup reads exact entries by `texelFetch`.
        let texture = unsafe { texture(gl, glow::RGBA8, 256, 1, glow::RGBA, glow::UNSIGNED_BYTE, &texels, true)? };
        Ok(GpuColormap { texture, domain: colormap.domain })
    }

    /// Starts a frame on the bound framebuffer: sets the viewport to the
    /// camera's, clears to `background`, and sets the blending and stencil
    /// state every draw relies on. Needs a stencil buffer; without one,
    /// overlapping tiles draw over each other.
    pub fn begin_frame(&self, gl: &glow::Context, camera: &Camera, background: [f32; 4]) {
        let [w, h] = camera.size_px;
        let [r, g, b, a] = background;
        // SAFETY: global state and a clear, on the caller's context.
        unsafe {
            gl.viewport(0, 0, w as i32, h as i32);
            gl.clear_color(r, g, b, a);
            gl.stencil_mask(0xff);
            gl.clear_stencil(0);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::STENCIL_BUFFER_BIT);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.enable(glow::BLEND);
            gl.blend_func_separate(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA, glow::ONE, glow::ONE_MINUS_SRC_ALPHA);
            gl.enable(glow::STENCIL_TEST);
        }
    }

    /// Starts a layer: every pixel is unclaimed again, so this layer's tiles
    /// draw over the last layer's.
    pub fn begin_layer(&self, gl: &glow::Context) {
        // SAFETY: a stencil clear on the bound framebuffer.
        unsafe {
            gl.stencil_mask(0xff);
            gl.clear(glow::STENCIL_BUFFER_BIT);
        }
    }

    /// Draws `mesh` where `camera` puts it, styled by `palette`: fills, then
    /// lines, then points, all within the pixels the tile claims.
    pub fn draw(&self, gl: &glow::Context, mesh: &GpuMesh, camera: &Camera, palette: &GpuPalette) {
        self.draw_mesh(gl, mesh, camera, palette, Mode::Screen);
    }

    /// Draws a raster tile through `colormap` at `opacity`, within the
    /// pixels its mesh claims.
    pub fn draw_raster(&self, gl: &glow::Context, raster: &GpuRaster, camera: &Camera, colormap: &GpuColormap, opacity: f32) {
        let matrix = camera.matrix(&raster.frame);
        let footprint = |gl: &glow::Context| {
            // SAFETY: draws the raster's own triangles from its own array.
            unsafe { gl.draw_elements(glow::TRIANGLES, raster.index_count, glow::UNSIGNED_INT, 0) }
        };
        // SAFETY: binds the raster's own array and textures to this
        // renderer's program, and draws within its own index buffer.
        unsafe {
            gl.bind_vertex_array(Some(raster.vao));
            self.claim(gl, &matrix, &IDENTITY, 1.0, footprint);
            let p = &self.raster;
            gl.use_program(Some(p.program));
            gl.uniform_matrix_4_f32_slice(p.u("u_matrix"), false, &matrix);
            gl.uniform_matrix_4_f32_slice(p.u("u_post"), false, &IDENTITY);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(raster.texture));
            gl.uniform_1_i32(p.u("u_band"), 0);
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(colormap.texture));
            gl.uniform_1_i32(p.u("u_ramp"), 1);
            gl.active_texture(glow::TEXTURE0);
            gl.uniform_1_i32(p.u("u_kind"), raster.kind);
            gl.uniform_2_f32(p.u("u_domain"), colormap.domain[0], colormap.domain[1]);
            gl.uniform_1_f32(p.u("u_opacity"), opacity);
            gl.uniform_1_i32(p.u("u_has_nodata"), raster.nodata.is_some() as i32);
            gl.uniform_1_f32(p.u("u_nodata"), raster.nodata.unwrap_or(0.0));
            gl.stencil_func(glow::EQUAL, 2, 0xff);
            gl.stencil_op(glow::KEEP, glow::KEEP, glow::KEEP);
            footprint(gl);
            gl.bind_vertex_array(Some(raster.vao));
            self.settle(gl, &matrix, &IDENTITY, 1.0, footprint);
            gl.bind_vertex_array(None);
        }
    }

    fn draw_mesh(&self, gl: &glow::Context, mesh: &GpuMesh, camera: &Camera, palette: &GpuPalette, mode: Mode) {
        let matrix = camera.matrix(&mesh.frame);
        let post = mode.post();
        // One pixel of the tile's own square, in its local units: the claim
        // grows by it so neighbours' claims overlap rather than leave a seam.
        let grow = 1.0 + (camera.units_per_px / mesh.frame.half_span) as f32;
        let square = |gl: &glow::Context| {
            // SAFETY: draws the four vertices of the renderer's own square.
            unsafe { gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4) }
        };
        let (fill, line) = match mode {
            Mode::Screen => (&self.fill, &self.line),
            Mode::Pick { .. } => (&self.fill_pick, &self.line_pick),
        };
        // SAFETY: binds this renderer's programs and the mesh's own arrays,
        // and draws index and instance ranges that lie inside its buffers.
        unsafe {
            gl.bind_vertex_array(Some(self.square_vao));
            self.claim(gl, &matrix, &post, grow, square);
            gl.stencil_func(glow::EQUAL, 2, 0xff);
            gl.stencil_op(glow::KEEP, glow::KEEP, glow::KEEP);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(palette.texture));

            gl.use_program(Some(fill.program));
            gl.uniform_matrix_4_f32_slice(fill.u("u_matrix"), false, &matrix);
            gl.uniform_matrix_4_f32_slice(fill.u("u_post"), false, &post);
            gl.uniform_1_i32(fill.u("u_palette"), 0);
            if let Mode::Pick { layer, .. } = mode {
                gl.uniform_1_u32(fill.u("u_layer"), layer);
            }
            gl.bind_vertex_array(Some(mesh.fill_vao));
            gl.draw_elements(glow::TRIANGLES, mesh.fill_count, glow::UNSIGNED_INT, 0);

            gl.use_program(Some(line.program));
            gl.uniform_matrix_4_f32_slice(line.u("u_matrix"), false, &matrix);
            gl.uniform_matrix_4_f32_slice(line.u("u_post"), false, &post);
            gl.uniform_2_f32(line.u("u_viewport"), camera.size_px[0] as f32, camera.size_px[1] as f32);
            gl.uniform_1_i32(line.u("u_palette"), 0);
            let min_half = match mode {
                Mode::Screen => 0.0,
                Mode::Pick { layer, min_half, .. } => {
                    gl.uniform_1_u32(line.u("u_layer"), layer);
                    min_half
                }
            };
            gl.uniform_1_f32(line.u("u_min_half"), min_half);
            for (vao, count, points) in [(mesh.line_vao, mesh.line_count, 0), (mesh.point_vao, mesh.point_count, 1)] {
                if count > 0 {
                    gl.uniform_1_i32(line.u("u_points"), points);
                    gl.bind_vertex_array(Some(vao));
                    gl.draw_arrays_instanced(glow::TRIANGLE_STRIP, 0, 4, count);
                }
            }

            gl.bind_vertex_array(Some(self.square_vao));
            self.settle(gl, &matrix, &post, grow, square);
            gl.bind_vertex_array(None);
        }
    }

    /// Marks the unclaimed pixels `footprint` covers as this tile's
    /// (stencil 0 -> 2), writing no colour. The footprint's array is bound.
    ///
    /// The test passes where the low bit is clear -- 0, or 2, which no pixel
    /// holds between tiles -- and writes the reference, 2.
    unsafe fn claim(&self, gl: &glow::Context, matrix: &[f32; 16], post: &[f32; 16], grow: f32, footprint: impl Fn(&glow::Context)) {
        // SAFETY: state changes and a draw with this renderer's program, on
        // the array the caller bound.
        unsafe {
            self.footprint_pass(gl, matrix, post, grow);
            gl.stencil_func(glow::EQUAL, 2, 0x01);
            gl.stencil_op(glow::KEEP, glow::KEEP, glow::REPLACE);
            footprint(gl);
            gl.color_mask(true, true, true, true);
        }
    }

    /// Marks this tile's claimed pixels as settled (-> 1), so no later tile
    /// draws there. Inside the footprint every pixel is 2 or already 1, so
    /// writing 1 throughout is exact.
    unsafe fn settle(&self, gl: &glow::Context, matrix: &[f32; 16], post: &[f32; 16], grow: f32, footprint: impl Fn(&glow::Context)) {
        // SAFETY: as for `claim`.
        unsafe {
            self.footprint_pass(gl, matrix, post, grow);
            gl.stencil_func(glow::ALWAYS, 1, 0xff);
            gl.stencil_op(glow::KEEP, glow::KEEP, glow::REPLACE);
            footprint(gl);
            gl.color_mask(true, true, true, true);
        }
    }

    unsafe fn footprint_pass(&self, gl: &glow::Context, matrix: &[f32; 16], post: &[f32; 16], grow: f32) {
        let p = &self.stencil;
        // SAFETY: binds this renderer's own program and sets its uniforms.
        unsafe {
            gl.color_mask(false, false, false, false);
            gl.stencil_mask(0xff);
            gl.use_program(Some(p.program));
            gl.uniform_matrix_4_f32_slice(p.u("u_matrix"), false, matrix);
            gl.uniform_matrix_4_f32_slice(p.u("u_post"), false, post);
            gl.uniform_1_f32(p.u("u_grow"), grow);
        }
    }

    /// Frees the shaders and the shared square.
    pub fn free(self, gl: &glow::Context) {
        // SAFETY: deletes objects this renderer created and alone refers to.
        unsafe {
            for p in [self.fill, self.fill_pick, self.line, self.line_pick, self.raster, self.stencil] {
                gl.delete_program(p.program);
            }
            gl.delete_vertex_array(self.square_vao);
            gl.delete_buffer(self.square_vbo);
        }
    }
}

/// ID-buffer picking (`../plans/rendering.org` § HIT-TESTING): the tiles
/// are drawn again into a one-pixel integer framebuffer, each fragment
/// writing its layer and feature id, and the pixel is read back.
///
/// The camera's view is mapped so that the one pixel asked about fills the
/// framebuffer, which keeps the picker's memory at sixteen bytes whatever
/// the viewport. The line expansion happens in the full viewport's pixels
/// first, so what picks is exactly what draws; draw order and stencil claims
/// are the same as on screen, so the feature on top wins.
///
/// Rasters are not picked, and do not hide the features beneath them from a
/// pick. Reading back stalls the GPU pipeline, which is free for a click but
/// wants throttling for hover.
pub struct Picker {
    framebuffer: glow::Framebuffer,
    renderbuffers: [glow::Renderbuffer; 2],
    post: [f32; 16],
    min_half: f32,
}

impl Picker {
    pub fn new(gl: &glow::Context) -> Result<Picker, String> {
        // SAFETY: creates a framebuffer and renderbuffers of its own and
        // checks the result is complete before it is used.
        unsafe {
            let framebuffer = gl.create_framebuffer()?;
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
            let color = gl.create_renderbuffer()?;
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(color));
            gl.renderbuffer_storage(glow::RENDERBUFFER, glow::RGBA32UI, 1, 1);
            gl.framebuffer_renderbuffer(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::RENDERBUFFER, Some(color));
            let stencil = gl.create_renderbuffer()?;
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(stencil));
            gl.renderbuffer_storage(glow::RENDERBUFFER, glow::DEPTH24_STENCIL8, 1, 1);
            gl.framebuffer_renderbuffer(glow::FRAMEBUFFER, glow::DEPTH_STENCIL_ATTACHMENT, glow::RENDERBUFFER, Some(stencil));
            let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.bind_renderbuffer(glow::RENDERBUFFER, None);
            if status != glow::FRAMEBUFFER_COMPLETE {
                return Err(format!("pick framebuffer incomplete: 0x{status:x}"));
            }
            Ok(Picker { framebuffer, renderbuffers: [color, stencil], post: IDENTITY, min_half: 0.0 })
        }
    }

    /// Starts a pick of the pixel at screen position `px` in `camera`'s view.
    /// Lines and points narrower than `tolerance_px` pick as if that wide,
    /// so a hairline can still be clicked. Binds the pick framebuffer until
    /// [`Picker::finish`].
    pub fn begin(&mut self, gl: &glow::Context, camera: &Camera, px: [f64; 2], tolerance_px: f32) {
        let [w, h] = [camera.size_px[0] as f64, camera.size_px[1] as f64];
        // The pixel's centre in clip space, then a map from clip space that
        // puts it at the centre and one pixel across the whole target.
        let cx = 2.0 * (px[0].floor() + 0.5) / w - 1.0;
        let cy = 1.0 - 2.0 * (px[1].floor() + 0.5) / h;
        self.post = [w as f32, 0.0, 0.0, 0.0, 0.0, h as f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, (-cx * w) as f32, (-cy * h) as f32, 0.0, 1.0];
        self.min_half = tolerance_px / 2.0;
        // SAFETY: binds and clears the picker's own framebuffer.
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.framebuffer));
            gl.viewport(0, 0, 1, 1);
            gl.disable(glow::BLEND);
            gl.enable(glow::STENCIL_TEST);
            gl.stencil_mask(0xff);
            gl.clear_buffer_u32_slice(glow::COLOR, 0, &[0, 0, 0, 0]);
            gl.clear_buffer_depth_stencil(glow::DEPTH_STENCIL, 0, 1.0, 0);
        }
    }

    /// Starts a layer, as [`Renderer::begin_layer`] does on screen.
    pub fn begin_layer(&self, gl: &glow::Context) {
        // SAFETY: a stencil clear on the picker's framebuffer.
        unsafe { gl.clear_buffer_depth_stencil(glow::DEPTH_STENCIL, 0, 1.0, 0) }
    }

    /// Draws `mesh` for picking, as `layer`. Call in the order the screen
    /// draws.
    pub fn draw(&self, gl: &glow::Context, renderer: &Renderer, mesh: &GpuMesh, camera: &Camera, palette: &GpuPalette, layer: u32) {
        renderer.draw_mesh(gl, mesh, camera, palette, Mode::Pick { post: self.post, layer, min_half: self.min_half });
    }

    /// Reads the pixel back and rebinds the default framebuffer. `None` if
    /// nothing was drawn there. The next frame's [`Renderer::begin_frame`]
    /// restores the viewport and blending.
    pub fn finish(&self, gl: &glow::Context) -> Option<Pick> {
        let mut texel = [0u8; 16];
        // SAFETY: reads one RGBA32UI texel of the picker's own framebuffer
        // into a buffer of exactly that size.
        unsafe {
            gl.read_pixels(0, 0, 1, 1, glow::RGBA_INTEGER, glow::UNSIGNED_INT, glow::PixelPackData::Slice(Some(&mut texel)));
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        let word = |i: usize| u32::from_ne_bytes(texel[i * 4..i * 4 + 4].try_into().unwrap());
        (word(2) == 1).then(|| Pick { layer: word(0), feature: word(1) })
    }

    pub fn free(self, gl: &glow::Context) {
        // SAFETY: deletes the picker's own objects.
        unsafe {
            gl.delete_framebuffer(self.framebuffer);
            for r in self.renderbuffers {
                gl.delete_renderbuffer(r);
            }
        }
    }
}

fn program(gl: &glow::Context, header: &str, vs: &str, fs: &str, uniforms: &[&'static str]) -> Result<Program, String> {
    // SAFETY: plain GL object creation and queries on handles created here.
    unsafe {
        let program = gl.create_program()?;
        let mut shaders = Vec::new();
        for (kind, body) in [(glow::VERTEX_SHADER, vs), (glow::FRAGMENT_SHADER, fs)] {
            let shader = gl.create_shader(kind)?;
            gl.shader_source(shader, &format!("{header}{body}"));
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                return Err(gl.get_shader_info_log(shader));
            }
            gl.attach_shader(program, shader);
            shaders.push(shader);
        }
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            return Err(gl.get_program_info_log(program));
        }
        for shader in shaders {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }
        // A uniform the compiler optimised away has no location; setting it
        // through `None` is a no-op.
        let uniforms = uniforms.iter().filter_map(|&n| gl.get_uniform_location(program, n).map(|l| (n, l))).collect();
        Ok(Program { program, uniforms })
    }
}

/// A 2D texture of `w` x `h` texels, clamped at its edges.
///
/// # Safety
/// `data` must hold exactly `w * h` texels of `format` and `ty`.
#[allow(clippy::too_many_arguments)]
unsafe fn texture(gl: &glow::Context, internal: u32, w: i32, h: i32, format: u32, ty: u32, data: &[u8], linear: bool) -> Result<glow::Texture, String> {
    let filter = if linear { glow::LINEAR } else { glow::NEAREST } as i32;
    // SAFETY: a texture of its own, filled per the caller's guarantee.
    unsafe {
        let texture = gl.create_texture()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
        gl.tex_image_2d(glow::TEXTURE_2D, 0, internal as i32, w, h, 0, format, ty, glow::PixelUnpackData::Slice(Some(data)));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
        Ok(texture)
    }
}

/// The palette's texels: a row of fill colours, a row of line colours, a
/// row of (line width, point size, 0, 0).
fn palette_texels(palette: &Palette) -> Vec<u8> {
    let s = palette.styles();
    let rows = s.iter().map(|s| s.fill).chain(s.iter().map(|s| s.line)).chain(s.iter().map(|s| [s.line_width, s.point_size, 0.0, 0.0]));
    rows.flat_map(|t| f32_bytes(&t)).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_ne_bytes()).collect()
}
