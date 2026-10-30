//! The GPU backend, over `glow`: WebGL2 in a browser, OpenGL 3.3 core
//! natively. Enabled by the `gl` feature.
//!
//! *Spike quality*: it uploads a [`Mesh`] and draws its fills in the palette
//! colour of each vertex's style class and its outlines as 1-pixel GL lines.
//! Shader-expanded lines, raster, picking and the tile cache come later
//! (`../plans/rendering.org` § V1 SLICE).
//!
//! This is the one module allowed `unsafe`: every `glow` call is an
//! `unsafe fn`, because GL itself checks nothing. The crate denies `unsafe`
//! everywhere else.
#![allow(unsafe_code)]

use crate::grid::TileFrame;
use crate::prepare::Mesh;
use glow::HasContext;

/// How many style classes the palette holds.
pub const PALETTE_SIZE: usize = 16;

const VERTEX: &str = r#"
in vec2 a_pos;
in float a_class;
uniform mat3 u_transform;
flat out int v_class;
void main() {
    vec3 p = u_transform * vec3(a_pos, 1.0);
    gl_Position = vec4(p.xy, 0.0, 1.0);
    v_class = int(a_class + 0.5);
}
"#;

const FRAGMENT: &str = r#"
flat in int v_class;
uniform vec4 u_palette[16];
uniform vec4 u_line;
uniform int u_lines;
out vec4 color;
void main() {
    color = u_lines != 0 ? u_line : u_palette[v_class & 15];
}
"#;

/// What part of the map the viewport shows: the map point at its centre, and
/// how many map units one pixel spans. *Spike*: the camera (step 4) replaces
/// this.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct View {
    pub center: [f64; 2],
    pub units_per_px: f64,
    pub size_px: [u32; 2],
}

impl View {
    /// The matrix taking `frame`'s tile-local coordinates to clip space,
    /// column-major. Composed in `f64` and only then cast, so the large map
    /// offsets cancel before any precision is lost.
    pub fn transform(&self, frame: &TileFrame) -> [f32; 9] {
        let half_w = self.units_per_px * self.size_px[0] as f64 / 2.0;
        let half_h = self.units_per_px * self.size_px[1] as f64 / 2.0;
        let sx = frame.half_span / half_w;
        let sy = frame.half_span / half_h;
        let tx = (frame.center[0] - self.center[0]) / half_w;
        let ty = (frame.center[1] - self.center[1]) / half_h;
        [sx as f32, 0.0, 0.0, 0.0, sy as f32, 0.0, tx as f32, ty as f32, 1.0]
    }
}

/// The compiled shaders, built once per GL context.
pub struct Renderer {
    program: glow::Program,
    u_transform: Option<glow::UniformLocation>,
    u_palette: Option<glow::UniformLocation>,
    u_line: Option<glow::UniformLocation>,
    u_lines: Option<glow::UniformLocation>,
}

/// A [`Mesh`] uploaded to the GPU. Nazca keeps no CPU copy once it is here
/// (`../plans/rendering.org` § TILE CACHE).
pub struct GpuMesh {
    frame: TileFrame,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    ebo: glow::Buffer,
    fill_count: i32,
    line_count: i32,
}

impl Renderer {
    pub fn new(gl: &glow::Context) -> Result<Renderer, String> {
        let header = if gl.version().is_embedded {
            "#version 300 es\nprecision highp float;\nprecision highp int;\n"
        } else {
            "#version 330 core\n"
        };
        // SAFETY: plain GL object creation and queries on handles created
        // here, on the context the caller passed.
        unsafe {
            let program = gl.create_program()?;
            let mut shaders = Vec::new();
            for (kind, body) in [(glow::VERTEX_SHADER, VERTEX), (glow::FRAGMENT_SHADER, FRAGMENT)] {
                let shader = gl.create_shader(kind)?;
                gl.shader_source(shader, &format!("{header}{body}"));
                gl.compile_shader(shader);
                if !gl.get_shader_compile_status(shader) {
                    return Err(gl.get_shader_info_log(shader));
                }
                gl.attach_shader(program, shader);
                shaders.push(shader);
            }
            gl.bind_attrib_location(program, 0, "a_pos");
            gl.bind_attrib_location(program, 1, "a_class");
            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                return Err(gl.get_program_info_log(program));
            }
            for shader in shaders {
                gl.detach_shader(program, shader);
                gl.delete_shader(shader);
            }
            Ok(Renderer {
                program,
                u_transform: gl.get_uniform_location(program, "u_transform"),
                u_palette: gl.get_uniform_location(program, "u_palette"),
                u_line: gl.get_uniform_location(program, "u_line"),
                u_lines: gl.get_uniform_location(program, "u_lines"),
            })
        }
    }

    /// Uploads `mesh`. The vertex buffer is shared by the fills and the
    /// outlines; one index buffer holds the fill triangles, then the outline
    /// segments.
    pub fn upload(&self, gl: &glow::Context, mesh: &Mesh) -> Result<GpuMesh, String> {
        let vertices: Vec<u8> = mesh.vertices.iter().flat_map(|v| v.to_le_bytes()).collect();
        let indices: Vec<u8> = mesh.fills.iter().chain(&mesh.outlines).flat_map(|i| i.to_le_bytes()).collect();
        // SAFETY: the buffers are created and bound here, and `buffer_data`
        // copies from slices that live for the whole call. The attribute
        // layout matches `Mesh::vertices`: three f32 per vertex.
        unsafe {
            let vao = gl.create_vertex_array()?;
            gl.bind_vertex_array(Some(vao));
            let vbo = gl.create_buffer()?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &vertices, glow::STATIC_DRAW);
            let ebo = gl.create_buffer()?;
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
            gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, &indices, glow::STATIC_DRAW);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 12, 0);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 1, glow::FLOAT, false, 12, 8);
            gl.bind_vertex_array(None);
            Ok(GpuMesh {
                frame: mesh.frame,
                vao,
                vbo,
                ebo,
                fill_count: mesh.fills.len() as i32,
                line_count: mesh.outlines.len() as i32,
            })
        }
    }

    /// Draws `mesh`: fills in `palette[class]`, then outlines in `line`.
    /// Colours are linear RGBA in 0..1.
    pub fn draw(&self, gl: &glow::Context, mesh: &GpuMesh, view: &View, palette: &[[f32; 4]; PALETTE_SIZE], line: [f32; 4]) {
        let palette: Vec<f32> = palette.iter().flatten().copied().collect();
        // SAFETY: binds this renderer's program and the mesh's own VAO, and
        // draws index ranges that lie inside the uploaded index buffer.
        unsafe {
            gl.use_program(Some(self.program));
            gl.uniform_matrix_3_f32_slice(self.u_transform.as_ref(), false, &view.transform(&mesh.frame));
            gl.uniform_4_f32_slice(self.u_palette.as_ref(), &palette);
            gl.uniform_4_f32_slice(self.u_line.as_ref(), &line);
            gl.bind_vertex_array(Some(mesh.vao));
            gl.uniform_1_i32(self.u_lines.as_ref(), 0);
            gl.draw_elements(glow::TRIANGLES, mesh.fill_count, glow::UNSIGNED_INT, 0);
            gl.uniform_1_i32(self.u_lines.as_ref(), 1);
            gl.draw_elements(glow::LINES, mesh.line_count, glow::UNSIGNED_INT, mesh.fill_count * 4);
            gl.bind_vertex_array(None);
        }
    }

    /// Frees `mesh`'s GPU buffers.
    pub fn free(&self, gl: &glow::Context, mesh: GpuMesh) {
        // SAFETY: deletes handles this renderer created, which `mesh` owned
        // and nothing else refers to.
        unsafe {
            gl.delete_vertex_array(mesh.vao);
            gl.delete_buffer(mesh.vbo);
            gl.delete_buffer(mesh.ebo);
        }
    }
}
