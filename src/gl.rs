//! The GPU backend, over `glow`: WebGL2 in a browser, OpenGL 3.3 core
//! natively. Enabled by the `gl` feature (`../plans/rendering.org` §
//! BACKENDS).
//!
//! The embedder makes the GL context and wraps it in a [`Context`]
//! ([`Context::from_webgl2`] in a browser, [`Context::from_loader`]
//! natively), which every call here takes. `glow` is not part of the API.
//! An embedder with GL calls of its own keeps its own binding over the same
//! context; no handle passes between the two. Draws set the bindings they
//! use (program, arrays, textures, stencil function), but the frame's state
//! is set once, by [`Renderer::begin_frame`]: calls of the embedder's own
//! between it and the frame's last draw restore what they change of the
//! viewport, blending, the stencil, depth and scissor tests, face culling
//! and the bound framebuffer.
//!
//! - *Fills* are the mesh's triangles, coloured by a [`Palette`] held as a
//!   lookup texture, so restyling uploads one small texture. A class with a
//!   [`crate::Pattern`] samples its symbol instead, repeated from the map's
//!   origin. The repeat is found from the fragment's pixel on screen, not
//!   the tile's coordinates, so tiles meet without a seam and a stand-in
//!   ancestor, whose coordinates are coarse at a deep zoom, draws it as
//!   finely; where the screen's centre lies in a class's repeat is reduced
//!   in `f64` for each draw, from the camera. The mipmap is picked from the
//!   unwrapped coordinate, and a sample is kept to the middle of the
//!   symbol's edge texels, so neither the wrap nor the atlas's padding
//!   leaves a line at a repeat's edge. Picking ignores the pattern: a fill
//!   picks wherever it covers.
//! - *Lines* are uploaded as centerline segments and expanded in the vertex
//!   shader: each segment is one instanced quad around it, and the fragment
//!   shader keeps what lies within half the width of the segment, with a
//!   one-pixel antialiased edge. That draws round caps and round joins with
//!   no neighbour information, and widths stay right at every zoom. A class
//!   with a [`crate::Dash`] keeps only what lies within half the width of an
//!   on length of its pattern, measured in pixels from the segment's
//!   distance along its line, so dash ends are round too and an on length of
//!   0 is a dot. A run of segments chained end to start shares a base
//!   distance and each carries its offset from it, so a run far along its
//!   line keeps one phase: `f32` loses the base's pixels, up to about one
//!   1,000 km along at the deepest level, but alike for every segment, a
//!   jump at the run's ends rather than a jitter at each vertex. A segment
//!   whose partner draws the same stretch of a shared border
//!   ([`Mesh::partners`]) collapses wherever the partner is drawn: its class
//!   has a line, and in a state draw it holds the state.
//! - *Points* are an instanced quad each, drawn by one program: a round dot
//!   in the fill colour, or, where the class has a [`crate::Marker`], its
//!   symbol sized by the point size, placed by its anchor and turned against
//!   the screen or the map. A layer mixing the two is one draw. A marker
//!   picks where its image is at least half opaque, so a click on a pin's
//!   transparent corner picks what lies under it.
//! - *Symbols* are one atlas texture and a table of where each id lies in
//!   it, held by the renderer and set by [`Renderer::set_symbols`]. A
//!   palette names a symbol by id, so a repack touches no palette. The atlas
//!   is mipmapped to level 2, which its 4 pixels of padding keep from
//!   averaging two symbols together ([`crate::symbol`]).
//! - *Rasters* are drawn over their [`RasterMesh`], the band coloured in the
//!   shader through a [`Colormap`] texture.
//! - *Picking* draws feature ids instead of colours into a one-pixel integer
//!   framebuffer and reads it back ([`Picker`]).
//! - *Feature state* draws a mesh again showing only the features whose
//!   flags a small texture marks ([`Renderer::draw_state`]), so choosing a
//!   feature rewrites one texture and touches no geometry. A pick can follow
//!   the same state ([`Picker::draw_state`]), so a layer drawn only through
//!   a state picks only what it shows.
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
use crate::style::{Align, CLASSES, Colormap, DASH_LENGTHS, Palette, Pattern, Style, pattern_phase};
use crate::symbol::{Kind, PADDING, SymbolId, Symbols};
use glow::HasContext;
use std::collections::HashMap;
#[cfg(any(not(target_family = "wasm"), target_os = "emscripten"))]
use std::ffi::{CStr, c_void};

const IDENTITY: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

/// Bytes a vertex: X, Y and the style class as `f32`, the feature id as `u32`.
const VERTEX_BYTES: usize = 16;
/// Bytes a segment instance: both ends' X and Y, the class, the feature id,
/// the distance along its line at its start as its run's base and its
/// offset from it, and its partner's class (-1 for none) and feature id.
const SEGMENT_BYTES: usize = 40;
/// Features a row of a state texture; `STATE_VS` wraps on the same number.
/// OpenGL 3.3 guarantees textures this wide.
const STATE_ROW: usize = 1024;
/// Rows of a palette texture (`palette_texels`).
const PALETTE_ROWS: i32 = 7;
/// Symbol ids a row of the symbol table; each takes two texels.
const SYMBOL_ROW: usize = 256;

// Prepended to a vertex shader for a state draw. Ids wrap as `uint`s, so a
// span may run past the top of the range; any id outside it holds no state.
const STATE_VS: &str = r#"
uniform highp usampler2D u_state;
uniform uint u_state_first;
uniform uint u_state_count;
uniform uint u_state_mask;
bool has_state(uint id) {
    uint i = id - u_state_first;
    if (i >= u_state_count) return false;
    return (texelFetch(u_state, ivec2(i & 1023u, i >> 10u), 0).r & u_state_mask) != 0u;
}
"#;

const FILL_VS: &str = r#"
layout(location = 0) in vec2 a_pos;
layout(location = 1) in float a_class;
layout(location = 2) in uint a_id;
uniform mat4 u_matrix;
uniform mat4 u_post;
flat out int v_class;
flat out uint v_id;
void main() {
#ifdef STATE
    if (!has_state(a_id)) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
#endif
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
uniform highp sampler2D u_table;
uniform highp sampler2D u_atlas;
uniform vec2 u_viewport;
uniform vec2 u_turn;
uniform vec4 u_phase[128];
out vec4 o_color;
void main() {
    vec4 fill = texelFetch(u_palette, ivec2(v_class, 0), 0);
    vec4 pattern = texelFetch(u_palette, ivec2(v_class, 6), 0);
    int id = int(pattern.x + 0.5) - 1;
    if (id < 0) {
        o_color = fill;
        return;
    }
    // The symbol's rectangle in the atlas and its kind, as for a marker.
    vec4 rect = vec4(0.0);
    float mask = 0.0;
    if (id / 256 < textureSize(u_table, 0).y) {
        ivec2 at = ivec2(2 * (id % 256), id / 256);
        rect = texelFetch(u_table, at, 0);
        mask = texelFetch(u_table, at + ivec2(1, 0), 0).z;
    }
    if (rect.z <= 0.0 || !(pattern.y > 0.0)) discard;
    vec2 cell = rect.zw * (pattern.y / max(rect.z, rect.w));
    vec4 pair = u_phase[v_class / 2];
    vec2 phase = v_class % 2 == 0 ? pair.xy : pair.zw;
    // The fragment's pixels from the viewport's centre, Y up, turned onto
    // the map's axes, from the centre's place in its repeat: the pattern
    // depends on no tile's coordinates, so tiles meet without a seam.
    vec2 v = gl_FragCoord.xy - 0.5 * u_viewport;
    vec2 q = phase + vec2(u_turn.x * v.x + u_turn.y * v.y, u_turn.x * v.y - u_turn.y * v.x);
    // In repeats, Y down as the image's rows run. The gradient is this
    // unwrapped coordinate's, worked out rather than taken by dFdx, which
    // the branches above leave undefined; the wrapped one jumps at each
    // repeat's edge and would pick the smallest mipmap along it.
    vec2 w = vec2(q.x, -q.y) / cell;
    vec2 size = vec2(textureSize(u_atlas, 0));
    vec2 dx = vec2(u_turn.x, u_turn.y) / cell * rect.zw / size;
    vec2 dy = vec2(u_turn.y, -u_turn.x) / cell * rect.zw / size;
    // Clamped to the middle of the edge texels of the coarser of the two
    // levels the gradient blends, so a repeat's edge takes nothing from the
    // padding around it.
    float lod = clamp(log2(max(rect.z, rect.w) / pattern.y), 0.0, 2.0);
    vec2 inset = min(vec2(0.5 * exp2(ceil(lod))), 0.5 * rect.zw);
    vec2 uv = clamp(rect.xy + fract(w) * rect.zw, rect.xy + inset, rect.xy + rect.zw - inset);
    vec4 t = textureGrad(u_atlas, uv / size, dx, dy);
    vec3 rgb = mask > 0.5 ? fill.rgb : (t.a > 0.0 ? t.rgb / t.a : vec3(0.0));
    float a = t.a * fill.a;
    if (a <= 0.0) discard;
    o_color = vec4(rgb, a);
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
layout(location = 4) in float a_base;
layout(location = 5) in float a_offset;
layout(location = 6) in float a_partner_class;
layout(location = 7) in uint a_partner_id;
uniform mat4 u_matrix;
uniform mat4 u_post;
uniform vec2 u_viewport;
uniform highp sampler2D u_palette;
uniform float u_min_half;
uniform float u_unit_px;
flat out vec2 v_a;
flat out vec2 v_b;
flat out float v_half;
flat out vec4 v_color;
flat out uint v_id;
flat out vec4 v_dash0;
flat out vec4 v_dash1;
flat out float v_dash_count;
flat out float v_period;
flat out float v_start;
out vec2 v_px;
vec2 to_px(vec2 p) {
    vec4 c = u_matrix * vec4(p, 0.0, 1.0);
    return (c.xy / c.w * 0.5 + 0.5) * u_viewport;
}
bool draws_line(int cls) {
    return texelFetch(u_palette, ivec2(cls, 2), 0).x > 0.0 && texelFetch(u_palette, ivec2(cls, 1), 0).a > 0.0;
}
void main() {
#ifdef STATE
    if (!has_state(a_id)) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
#endif
    // The later copy of a shared border draws it, where it is drawn at all.
    if (a_partner_class >= 0.0) {
        bool partner = draws_line(int(a_partner_class + 0.5));
#ifdef STATE
        partner = partner && has_state(a_partner_id);
#endif
        if (partner) {
            gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
            return;
        }
    }
    int cls = int(a_class + 0.5);
    float width = texelFetch(u_palette, ivec2(cls, 2), 0).x;
    v_color = texelFetch(u_palette, ivec2(cls, 1), 0);
    v_half = width > 0.0 ? max(width * 0.5, u_min_half) : 0.0;
    // The dash pattern, an odd count of lengths repeated twice, and where
    // in it the segment starts: the run's base reduced by the pattern
    // before the offset is added, so the base's rounding, the same for the
    // whole run, is all a segment far along its line loses.
    v_dash0 = texelFetch(u_palette, ivec2(cls, 4), 0);
    v_dash1 = texelFetch(u_palette, ivec2(cls, 5), 0);
    vec4 phase_count = texelFetch(u_palette, ivec2(cls, 6), 0);
    v_dash_count = phase_count.w;
    float sum = dot(v_dash0, vec4(1.0)) + dot(v_dash1, vec4(1.0));
    v_period = mod(phase_count.w, 2.0) > 0.5 ? 2.0 * sum : sum;
    v_start = v_period > 0.0 ? mod(mod(a_base * u_unit_px, v_period) + a_offset * u_unit_px + phase_count.z, v_period) : 0.0;
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
flat in vec4 v_dash0;
flat in vec4 v_dash1;
flat in float v_dash_count;
flat in float v_period;
flat in float v_start;
in vec2 v_px;
float distance_to_segment() {
    vec2 ab = v_b - v_a;
    vec2 ap = v_px - v_a;
    float l = dot(ab, ab);
    float t = l > 0.0 ? clamp(dot(ap, ab) / l, 0.0, 1.0) : 0.0;
    return length(ap - ab * t);
}
// How far the fragment lies from the nearest on length of the dash
// pattern, laid along the segment's line from where the segment starts in
// it: 0 along it inside a dash, so a dash is a capsule and its ends round.
float distance_to_dash() {
    vec2 ab = v_b - v_a;
    float len = length(ab);
    vec2 dir = len > 0.0 ? ab / len : vec2(1.0, 0.0);
    vec2 ap = v_px - v_a;
    float u = mod(v_start + dot(ap, dir), v_period);
    float across = abs(dot(ap, vec2(-dir.y, dir.x)));
    float lengths[8] = float[8](v_dash0.x, v_dash0.y, v_dash0.z, v_dash0.w, v_dash1.x, v_dash1.y, v_dash1.z, v_dash1.w);
    int n = int(v_dash_count + 0.5);
    int total = n % 2 == 1 ? 2 * n : n;
    float at = 0.0;
    float gap = 1e30;
    for (int i = 0; i < 16; i++) {
        if (i >= total) break;
        float l = lengths[i % n];
        if (i % 2 == 0) {
            for (int k = -1; k <= 1; k++) {
                float lo = at + float(k) * v_period;
                gap = min(gap, max(max(lo - u, u - lo - l), 0.0));
            }
        }
        at += l;
    }
    return length(vec2(gap, across));
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
    float d = distance_to_segment();
    if (v_period > 0.0) d = max(d, distance_to_dash());
    float cover = clamp(v_half + 0.5 - d, 0.0, 1.0);
    if (cover <= 0.0) discard;
    o_color = vec4(v_color.rgb, v_color.a * cover);
}
#endif
"#;

// One instance a point, as for lines: a dot where its class has no marker,
// the marker's symbol where it has one, so a layer mixing the two is one
// draw. A marker's quad is its symbol's box, its longest side the point
// size, placed by its anchor and turned by its rotation, plus a pixel on
// every side for the atlas's transparent padding to soften the edge.
const MARKER_VS: &str = r#"
layout(location = 0) in vec2 a_pos;
layout(location = 2) in float a_class;
layout(location = 3) in uint a_id;
uniform mat4 u_matrix;
uniform mat4 u_post;
uniform vec2 u_viewport;
uniform highp sampler2D u_palette;
uniform highp sampler2D u_table;
uniform float u_min_half;
flat out vec2 v_center;
flat out float v_half;
flat out vec4 v_color;
flat out uint v_id;
flat out int v_kind;
flat out vec4 v_rect;
out vec2 v_px;
out vec2 v_uv;
vec2 to_px(vec2 p) {
    vec4 c = u_matrix * vec4(p, 0.0, 1.0);
    return (c.xy / c.w * 0.5 + 0.5) * u_viewport;
}
void main() {
#ifdef STATE
    if (!has_state(a_id)) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
#endif
    int cls = int(a_class + 0.5);
    float size = texelFetch(u_palette, ivec2(cls, 2), 0).y;
    vec4 marker = texelFetch(u_palette, ivec2(cls, 3), 0);
    v_color = texelFetch(u_palette, ivec2(cls, 0), 0);
    v_id = a_id;
    vec2 c = to_px(a_pos);
    v_center = c;
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    // The symbol's rectangle in the atlas, and its anchor and kind; a
    // symbol the table doesn't hold has no width.
    int id = int(marker.x + 0.5) - 1;
    vec4 rect = vec4(0.0);
    vec4 shape = vec4(0.0);
    if (id >= 0 && id / 256 < textureSize(u_table, 0).y) {
        ivec2 at = ivec2(2 * (id % 256), id / 256);
        rect = texelFetch(u_table, at, 0);
        shape = texelFetch(u_table, at + ivec2(1, 0), 0);
    }
    vec2 p;
    if (rect.z > 0.0 && size > 0.0) {
        vec2 box = rect.zw * (size / max(rect.z, rect.w));
        vec2 grow = 1.0 / box;
        // Fractions of the image, Y down, and their offset from the anchor
        // in pixels, Y up.
        vec2 u = mix(-grow, 1.0 + grow, corner);
        vec2 off = vec2(u.x - shape.x, shape.y - u.y) * box;
        float turn = radians(marker.y);
        if (marker.z > 0.5) {
            // Against the map: map north's direction on screen, clockwise
            // from up.
            vec2 north = (u_matrix * vec4(0.0, 1.0, 0.0, 0.0)).xy * u_viewport;
            turn += atan(north.x, north.y);
        }
        float cs = cos(turn);
        float sn = sin(turn);
        p = c + vec2(off.x * cs + off.y * sn, off.y * cs - off.x * sn);
        v_uv = rect.xy + u * rect.zw;
        v_rect = rect;
        v_kind = int(shape.z + 0.5);
        v_half = 0.0;
    } else {
        v_half = id < 0 && size > 0.0 ? max(size * 0.5, u_min_half) : 0.0;
        float r = v_half > 0.0 ? v_half + 1.0 : 0.0;
        p = c + (corner * 2.0 - 1.0) * r;
        v_uv = vec2(0.0);
        v_rect = vec4(0.0);
        v_kind = -1;
    }
    v_px = p;
    gl_Position = u_post * vec4(p / u_viewport * 2.0 - 1.0, 0.0, 1.0);
}
"#;

// v_kind: -1 a dot, 0 a colour symbol, 1 a mask. The atlas is premultiplied
// (`crate::symbol`), so a colour sample is divided back to straight alpha.
// The lookup is clamped a texel past the symbol's rectangle, inside its
// padding, and its gradient taken before the clamp and the branch.
const MARKER_FS: &str = r#"
flat in vec2 v_center;
flat in float v_half;
flat in vec4 v_color;
flat in uint v_id;
flat in int v_kind;
flat in vec4 v_rect;
in vec2 v_px;
in vec2 v_uv;
uniform highp sampler2D u_atlas;
vec4 sample_atlas(vec2 dx, vec2 dy) {
    vec2 uv = clamp(v_uv, v_rect.xy - 1.0, v_rect.xy + v_rect.zw + 1.0);
    return textureGrad(u_atlas, uv / vec2(textureSize(u_atlas, 0)), dx, dy);
}
#ifdef PICK
uniform uint u_layer;
out uvec4 o_pick;
void main() {
    vec2 size = vec2(textureSize(u_atlas, 0));
    vec2 dx = dFdx(v_uv / size);
    vec2 dy = dFdy(v_uv / size);
    if (v_kind < 0) {
        if (length(v_px - v_center) > v_half) discard;
    } else if (sample_atlas(dx, dy).a < 0.5) {
        discard;
    }
    o_pick = uvec4(u_layer, v_id, 1u, 0u);
}
#else
out vec4 o_color;
void main() {
    vec2 size = vec2(textureSize(u_atlas, 0));
    vec2 dx = dFdx(v_uv / size);
    vec2 dy = dFdy(v_uv / size);
    vec4 c;
    if (v_kind < 0) {
        float cover = clamp(v_half + 0.5 - length(v_px - v_center), 0.0, 1.0);
        c = vec4(v_color.rgb, v_color.a * cover);
    } else {
        vec4 t = sample_atlas(dx, dy);
        vec3 rgb = v_kind == 1 ? v_color.rgb : (t.a > 0.0 ? t.rgb / t.a : vec3(0.0));
        c = vec4(rgb, t.a * v_color.a);
    }
    if (c.a <= 0.0) discard;
    o_color = c;
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

/// Whether a draw goes to the screen, to the screen for the features holding
/// a state, or to the picker, for every feature or those holding a state.
#[derive(Clone, Copy)]
enum Mode {
    Screen,
    State(StateTest),
    Pick { post: [f32; 16], layer: u32, min_half: f32, state: Option<StateTest> },
}

/// A [`GpuState`] and the mask a draw tests its flags against.
#[derive(Clone, Copy)]
struct StateTest {
    texture: glow::Texture,
    first: u32,
    count: u32,
    mask: u32,
}

impl StateTest {
    fn new(state: &GpuState, mask: u8) -> StateTest {
        StateTest { texture: state.texture, first: state.first, count: state.count, mask: u32::from(mask) }
    }
}

impl Mode {
    fn post(&self) -> [f32; 16] {
        match self {
            Mode::Screen | Mode::State(_) => IDENTITY,
            Mode::Pick { post, .. } => *post,
        }
    }

    fn state(&self) -> Option<StateTest> {
        match *self {
            Mode::Screen => None,
            Mode::State(state) => Some(state),
            Mode::Pick { state, .. } => state,
        }
    }
}

/// The GL context nazca draws with: WebGL2 in a browser, OpenGL 3.3 core
/// or OpenGL ES 3.0 natively. It wraps the embedder's context and names no
/// binding, so the embedder needs no `glow` of its own and a `glow` upgrade
/// changes nothing it sees. Every call below takes it.
pub struct Context {
    raw: glow::Context,
}

impl Context {
    /// Wraps a browser's WebGL2 context. Every extension the browser offers
    /// is turned on, among them the float filtering continuous rasters use.
    #[cfg(all(target_family = "wasm", not(target_os = "emscripten")))]
    pub fn from_webgl2(context: web_sys::WebGl2RenderingContext) -> Context {
        Context { raw: glow::Context::from_webgl2_context(context) }
    }

    /// Wraps a native context through its loader, which returns the address
    /// of a GL function by name (`eglGetProcAddress`, glutin's
    /// `get_proc_address`).
    ///
    /// # Safety
    ///
    /// `load` returns OpenGL 3.3 core or OpenGL ES 3.0 functions of one
    /// context, and that context is current on the calling thread whenever
    /// this `Context` is handed to nazca.
    #[cfg(any(not(target_family = "wasm"), target_os = "emscripten"))]
    pub unsafe fn from_loader(load: impl FnMut(&CStr) -> *const c_void) -> Context {
        // SAFETY: the caller's contract is glow's.
        Context { raw: unsafe { glow::Context::from_loader_function_cstr(load) } }
    }

    /// The context's `GL_VERSION` string.
    pub fn version(&self) -> String {
        // SAFETY: a string query, on the caller's context.
        unsafe { self.raw.get_parameter_string(glow::VERSION) }
    }

    /// Reads `size` pixels from the bound framebuffer's lower left corner,
    /// as RGBA8, bottom row first.
    pub fn read_pixels(&self, size: [u32; 2]) -> Vec<u8> {
        let [w, h] = size;
        let mut pixels = vec![0u8; w as usize * h as usize * 4];
        // SAFETY: reads `w * h` RGBA8 pixels into a buffer of exactly that
        // size; their rows are whole words, so the default pack alignment
        // adds no padding.
        unsafe {
            self.raw.read_pixels(0, 0, w as i32, h as i32, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(Some(&mut pixels)));
        }
        pixels
    }
}

/// The compiled shaders and the shared unit square, built once per GL
/// context.
pub struct Renderer {
    fill: Program,
    fill_pick: Program,
    fill_state: Program,
    fill_pick_state: Program,
    line: Program,
    line_pick: Program,
    line_state: Program,
    line_pick_state: Program,
    marker: Program,
    marker_pick: Program,
    marker_state: Program,
    marker_pick_state: Program,
    raster: Program,
    stencil: Program,
    square_vao: glow::VertexArray,
    square_vbo: glow::Buffer,
    float_linear: bool,
    symbols: GpuSymbols,
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

/// A [`Palette`] as a lookup texture: 256 x 7 RGBA32F, a column per class,
/// rows fill colour, line colour, line width and point size, marker, two of
/// dash lengths, and pattern (`palette_texels`). The classes with a pattern
/// are kept on the CPU too, to place their repeats for each draw.
pub struct GpuPalette {
    texture: glow::Texture,
    patterns: Vec<(u8, Pattern)>,
}

/// [`Symbols`] on the GPU, held by the [`Renderer`]: the atlas as an RGBA8
/// texture mipmapped to level 2, and an RGBA32F table, 512 wide, of two
/// texels an id: its rectangle in atlas pixels, then its anchor and kind.
/// Each id's width and height are kept on the CPU too, by id, 0 for an id
/// not held, to size a pattern's repeat.
struct GpuSymbols {
    atlas: glow::Texture,
    table: glow::Texture,
    sizes: Vec<[u32; 2]>,
    bytes: usize,
}

/// A [`Colormap`] as a 256 x 1 RGBA8 lookup texture.
pub struct GpuColormap {
    texture: glow::Texture,
    domain: [f32; 2],
}

/// A byte of flags for each of a span of feature ids, one per bit, as an
/// R8UI texture. An id outside the span has none. [`GpuState::free`] releases
/// it.
pub struct GpuState {
    texture: glow::Texture,
    first: u32,
    count: u32,
    bytes: usize,
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

    pub fn free(self, gl: &Context) {
        let gl = &gl.raw;
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

    pub fn free(self, gl: &Context) {
        let gl = &gl.raw;
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
    pub fn free(self, gl: &Context) {
        let gl = &gl.raw;
        // SAFETY: deletes the palette's own texture.
        unsafe { gl.delete_texture(self.texture) }
    }
}

impl GpuSymbols {
    /// Uploads `symbols`' atlas and the table of where each id lies in it.
    fn new(gl: &glow::Context, symbols: &Symbols) -> Result<GpuSymbols, String> {
        let side = symbols.atlas_side() as i32;
        let (rows, table) = symbol_table_texels(symbols);
        // SAFETY: textures of their own, filled from the atlas of `side`
        // squared RGBA8 pixels and a local of `2 * SYMBOL_ROW * rows` texels.
        unsafe {
            let atlas = texture(gl, glow::RGBA8, side, side, glow::RGBA, glow::UNSIGNED_BYTE, symbols.atlas(), true)?;
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR_MIPMAP_LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAX_LEVEL, PADDING.ilog2() as i32);
            gl.generate_mipmap(glow::TEXTURE_2D);
            let table = match texture(gl, glow::RGBA32F, 2 * SYMBOL_ROW as i32, rows as i32, glow::RGBA, glow::FLOAT, &table, false) {
                Ok(table) => table,
                Err(e) => {
                    gl.delete_texture(atlas);
                    return Err(e);
                }
            };
            // Level 0 and the two mipmaps below it, then the table.
            let texels = (side * side) as usize;
            let bytes = 4 * (texels + texels / 4 + texels / 16) + 2 * SYMBOL_ROW * rows * 16;
            let mut sizes = vec![[0; 2]; symbols.iter().last().map_or(0, |(id, _, _)| id.0 as usize + 1)];
            for (id, symbol, _) in symbols.iter() {
                sizes[id.0 as usize] = [symbol.width(), symbol.height()];
            }
            Ok(GpuSymbols { atlas, table, sizes, bytes })
        }
    }

    fn free(self, gl: &glow::Context) {
        // SAFETY: deletes the symbols' own textures.
        unsafe {
            gl.delete_texture(self.atlas);
            gl.delete_texture(self.table);
        }
    }
}

impl GpuColormap {
    pub fn free(self, gl: &Context) {
        let gl = &gl.raw;
        // SAFETY: deletes the colormap's own texture.
        unsafe { gl.delete_texture(self.texture) }
    }
}

impl GpuState {
    /// The GPU memory it holds, to count against the tile budget.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn free(self, gl: &Context) {
        let gl = &gl.raw;
        // SAFETY: deletes the state's own texture.
        unsafe { gl.delete_texture(self.texture) }
    }
}

impl Renderer {
    pub fn new(gl: &Context) -> Result<Renderer, String> {
        let gl = &gl.raw;
        let embedded = gl.version().is_embedded;
        let header = if embedded { "#version 300 es\nprecision highp float;\nprecision highp int;\n" } else { "#version 330 core\n" };
        let pick = format!("{header}#define PICK\n");
        let state = format!("{header}#define STATE\n");
        let pick_state = format!("{header}#define PICK\n#define STATE\n");
        let fill_u = ["u_matrix", "u_post", "u_palette", "u_table", "u_atlas", "u_viewport", "u_turn", "u_phase", "u_layer", "u_state", "u_state_first", "u_state_count", "u_state_mask"];
        let line_u = ["u_matrix", "u_post", "u_viewport", "u_palette", "u_min_half", "u_unit_px", "u_layer", "u_state", "u_state_first", "u_state_count", "u_state_mask"];
        let marker_u = ["u_matrix", "u_post", "u_viewport", "u_palette", "u_table", "u_atlas", "u_min_half", "u_layer", "u_state", "u_state_first", "u_state_count", "u_state_mask"];
        let raster_u = ["u_matrix", "u_post", "u_band", "u_ramp", "u_kind", "u_domain", "u_opacity", "u_has_nodata", "u_nodata"];
        let renderer = Renderer {
            fill: program(gl, header, FILL_VS, FILL_FS, &fill_u)?,
            fill_pick: program(gl, &pick, FILL_VS, FILL_FS, &fill_u)?,
            fill_state: program(gl, &state, &format!("{STATE_VS}{FILL_VS}"), FILL_FS, &fill_u)?,
            fill_pick_state: program(gl, &pick_state, &format!("{STATE_VS}{FILL_VS}"), FILL_FS, &fill_u)?,
            line: program(gl, header, LINE_VS, LINE_FS, &line_u)?,
            line_pick: program(gl, &pick, LINE_VS, LINE_FS, &line_u)?,
            line_state: program(gl, &state, &format!("{STATE_VS}{LINE_VS}"), LINE_FS, &line_u)?,
            line_pick_state: program(gl, &pick_state, &format!("{STATE_VS}{LINE_VS}"), LINE_FS, &line_u)?,
            marker: program(gl, header, MARKER_VS, MARKER_FS, &marker_u)?,
            marker_pick: program(gl, &pick, MARKER_VS, MARKER_FS, &marker_u)?,
            marker_state: program(gl, &state, &format!("{STATE_VS}{MARKER_VS}"), MARKER_FS, &marker_u)?,
            marker_pick_state: program(gl, &pick_state, &format!("{STATE_VS}{MARKER_VS}"), MARKER_FS, &marker_u)?,
            raster: program(gl, header, RASTER_VS, RASTER_FS, &raster_u)?,
            stencil: program(gl, header, STENCIL_VS, STENCIL_FS, &["u_matrix", "u_post", "u_grow"])?,
            // SAFETY: creates and fills a buffer and array of its own; the
            // slice outlives the copy.
            square_vao: unsafe { gl.create_vertex_array()? },
            square_vbo: unsafe { gl.create_buffer()? },
            // Desktop GL filters float textures linearly; WebGL2 only with
            // the extension (which glow enables when it is offered).
            float_linear: !embedded || gl.supported_extensions().contains("OES_texture_float_linear"),
            symbols: GpuSymbols::new(gl, &Symbols::new())?,
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
    pub fn upload(&self, gl: &Context, mesh: &Mesh) -> Result<GpuMesh, String> {
        let gl = &gl.raw;
        let mut vertices = Vec::with_capacity(mesh.vertex_count() * VERTEX_BYTES);
        for (v, id) in mesh.vertices.chunks_exact(3).zip(&mesh.ids) {
            vertices.extend_from_slice(&f32_bytes(v));
            vertices.extend_from_slice(&id.to_ne_bytes());
        }
        let line_count = mesh.outlines.len() / 2;
        let point_count = mesh.points.len();
        let mut segments = Vec::with_capacity((line_count + point_count) * SEGMENT_BYTES);
        let mut segment = |a: u32, b: u32, base: f32, partner: Option<usize>| {
            let v = &mesh.vertices;
            let (i, j) = (a as usize, b as usize);
            segments.extend_from_slice(&f32_bytes(&[v[i * 3], v[i * 3 + 1], v[j * 3], v[j * 3 + 1], v[i * 3 + 2]]));
            segments.extend_from_slice(&mesh.ids[i].to_ne_bytes());
            let (class, id) = partner.map_or((-1.0, 0), |p| {
                let pa = mesh.outlines[p * 2] as usize;
                (v[pa * 3 + 2], mesh.ids[pa])
            });
            let offset = (mesh.distance(a) - base as f64) as f32;
            segments.extend_from_slice(&f32_bytes(&[base, offset, class]));
            segments.extend_from_slice(&id.to_ne_bytes());
        };
        // A run goes on while each segment starts where the last ended, its
        // distance rising: a ring's closing edge ends where its distance
        // starts again. The base is the run's first distance as an `f32`,
        // the offsets taking up its rounding.
        let (mut base, mut last) = (0.0f32, None);
        for (k, s) in mesh.outlines.chunks_exact(2).enumerate() {
            let d = mesh.distance(s[0]);
            if !matches!(last, Some((end, before)) if end == s[0] && d >= before) {
                base = d as f32;
            }
            segment(s[0], s[1], base, mesh.partner(k));
            last = Some((s[1], d));
        }
        for &p in &mesh.points {
            segment(p, p, 0.0, None);
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
                for (loc, size, offset) in [(0, 2, 0), (1, 2, 8), (2, 1, 16), (4, 1, 24), (5, 1, 28), (6, 1, 32)] {
                    gl.enable_vertex_attrib_array(loc);
                    gl.vertex_attrib_pointer_f32(loc, size, glow::FLOAT, false, stride, base + offset);
                    gl.vertex_attrib_divisor(loc, 1);
                }
                for (loc, offset) in [(3, 20), (7, 36)] {
                    gl.enable_vertex_attrib_array(loc);
                    gl.vertex_attrib_pointer_i32(loc, 1, glow::UNSIGNED_INT, stride, base + offset);
                    gl.vertex_attrib_divisor(loc, 1);
                }
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
    pub fn upload_raster(&self, gl: &Context, mesh: &RasterMesh, raster: &Raster) -> Result<GpuRaster, String> {
        let gl = &gl.raw;
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

    pub fn palette(&self, gl: &Context, palette: &Palette) -> Result<GpuPalette, String> {
        let gl = &gl.raw;
        // SAFETY: a texture of its own, filled from a local of 256 x 7 texels.
        let texture = unsafe { texture(gl, glow::RGBA32F, CLASSES as i32, PALETTE_ROWS, glow::RGBA, glow::FLOAT, &palette_texels(palette), false)? };
        Ok(GpuPalette { texture, patterns: patterns(palette) })
    }

    /// Restyles: rewrites `gpu` from `palette`. No tile is touched.
    pub fn update_palette(&self, gl: &Context, gpu: &mut GpuPalette, palette: &Palette) {
        let gl = &gl.raw;
        gpu.patterns = patterns(palette);
        let texels = palette_texels(palette);
        // SAFETY: rewrites the whole of the palette's own texture, at the
        // size it was made with, from a local.
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(gpu.texture));
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, CLASSES as i32, PALETTE_ROWS, glow::RGBA, glow::FLOAT, glow::PixelUnpackData::Slice(Some(&texels)));
        }
    }

    /// Uploads the symbols every draw and pick takes markers from, in place
    /// of the last ones set; a renderer starts with none. Call it after
    /// adding or removing symbols. No palette or tile is touched: palettes
    /// name symbols by id.
    pub fn set_symbols(&mut self, gl: &Context, symbols: &Symbols) -> Result<(), String> {
        let gl = &gl.raw;
        let new = GpuSymbols::new(gl, symbols)?;
        std::mem::replace(&mut self.symbols, new).free(gl);
        Ok(())
    }

    /// The GPU memory the symbols last set hold.
    pub fn symbols_bytes(&self) -> usize {
        self.symbols.bytes
    }

    /// Uploads feature flags: `flags[i]` belongs to feature id `first + i`,
    /// each bit a state of its own. A span too long for a texture is an
    /// error.
    pub fn state(&self, gl: &Context, first: u32, flags: &[u8]) -> Result<GpuState, String> {
        let gl = &gl.raw;
        let texels = state_texels(flags);
        let rows = texels.len() / STATE_ROW;
        // SAFETY: an integer query, then a texture of its own, filled from
        // a local of `STATE_ROW * rows` single-byte texels.
        unsafe {
            let limit = gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE) as usize;
            if rows > limit {
                return Err(format!("flags for {} features need {rows} rows, past the texture limit of {limit}", flags.len()));
            }
            let texture = texture(gl, glow::R8UI, STATE_ROW as i32, rows as i32, glow::RED_INTEGER, glow::UNSIGNED_BYTE, &texels, false)?;
            Ok(GpuState { texture, first, count: flags.len() as u32, bytes: texels.len() })
        }
    }

    /// Changes features' states: rewrites `gpu` from `flags`, as long as the
    /// flags it was made with. No tile is touched.
    pub fn update_state(&self, gl: &Context, gpu: &GpuState, flags: &[u8]) -> Result<(), String> {
        let gl = &gl.raw;
        if flags.len() != gpu.count as usize {
            return Err(format!("{} flags for a state of {} features", flags.len(), gpu.count));
        }
        let texels = state_texels(flags);
        let rows = (texels.len() / STATE_ROW) as i32;
        // SAFETY: rewrites the whole of the state's own texture, at the size
        // it was made with, since the length is the same, from a local.
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(gpu.texture));
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, STATE_ROW as i32, rows, glow::RED_INTEGER, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(&texels)));
        }
        Ok(())
    }

    pub fn colormap(&self, gl: &Context, colormap: &Colormap) -> Result<GpuColormap, String> {
        let gl = &gl.raw;
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
    pub fn begin_frame(&self, gl: &Context, camera: &Camera, background: [f32; 4]) {
        let gl = &gl.raw;
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
    pub fn begin_layer(&self, gl: &Context) {
        let gl = &gl.raw;
        // SAFETY: a stencil clear on the bound framebuffer.
        unsafe {
            gl.stencil_mask(0xff);
            gl.clear(glow::STENCIL_BUFFER_BIT);
        }
    }

    /// Draws `mesh` where `camera` puts it, styled by `palette`: fills, then
    /// lines, then points as dots or markers, all within the pixels the tile
    /// claims.
    pub fn draw(&self, gl: &Context, mesh: &GpuMesh, camera: &Camera, palette: &GpuPalette) {
        let gl = &gl.raw;
        self.draw_mesh(gl, mesh, camera, palette, Mode::Screen);
    }

    /// Draws only the features of `mesh` whose flags in `state` share a bit
    /// with `mask`, styled by `palette`, as [`Renderer::draw`] draws: start
    /// a layer for it and it lands over the layers before.
    pub fn draw_state(&self, gl: &Context, mesh: &GpuMesh, camera: &Camera, palette: &GpuPalette, state: &GpuState, mask: u8) {
        let gl = &gl.raw;
        self.draw_mesh(gl, mesh, camera, palette, Mode::State(StateTest::new(state, mask)));
    }

    /// Draws a raster tile through `colormap` at `opacity`, within the
    /// pixels its mesh claims.
    pub fn draw_raster(&self, gl: &Context, raster: &GpuRaster, camera: &Camera, colormap: &GpuColormap, opacity: f32) {
        let gl = &gl.raw;
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
        let state = mode.state();
        let (fill, line, marker) = match mode {
            Mode::Screen => (&self.fill, &self.line, &self.marker),
            Mode::State(_) => (&self.fill_state, &self.line_state, &self.marker_state),
            Mode::Pick { state: None, .. } => (&self.fill_pick, &self.line_pick, &self.marker_pick),
            Mode::Pick { state: Some(_), .. } => (&self.fill_pick_state, &self.line_pick_state, &self.marker_pick_state),
        };
        let state_uniforms = |gl: &glow::Context, p: &Program| {
            if let Some(StateTest { first, count, mask, .. }) = state {
                // SAFETY: sets uniforms of the program the caller bound.
                unsafe {
                    gl.uniform_1_i32(p.u("u_state"), 1);
                    gl.uniform_1_u32(p.u("u_state_first"), first);
                    gl.uniform_1_u32(p.u("u_state_count"), count);
                    gl.uniform_1_u32(p.u("u_state_mask"), mask);
                }
            }
        };
        // SAFETY: binds this renderer's programs and the mesh's own arrays,
        // and draws index and instance ranges that lie inside its buffers.
        unsafe {
            gl.bind_vertex_array(Some(self.square_vao));
            self.claim(gl, &matrix, &post, grow, square);
            gl.stencil_func(glow::EQUAL, 2, 0xff);
            gl.stencil_op(glow::KEEP, glow::KEEP, glow::KEEP);
            if let Some(state) = state {
                gl.active_texture(glow::TEXTURE1);
                gl.bind_texture(glow::TEXTURE_2D, Some(state.texture));
            }
            gl.active_texture(glow::TEXTURE2);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.symbols.table));
            gl.active_texture(glow::TEXTURE3);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.symbols.atlas));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(palette.texture));

            gl.use_program(Some(fill.program));
            gl.uniform_matrix_4_f32_slice(fill.u("u_matrix"), false, &matrix);
            gl.uniform_matrix_4_f32_slice(fill.u("u_post"), false, &post);
            gl.uniform_1_i32(fill.u("u_palette"), 0);
            state_uniforms(gl, fill);
            match mode {
                Mode::Pick { layer, .. } => gl.uniform_1_u32(fill.u("u_layer"), layer),
                // Picking ignores patterns: a fill picks wherever it covers.
                Mode::Screen | Mode::State(_) if !palette.patterns.is_empty() => {
                    gl.uniform_1_i32(fill.u("u_table"), 2);
                    gl.uniform_1_i32(fill.u("u_atlas"), 3);
                    gl.uniform_2_f32(fill.u("u_viewport"), camera.size_px[0] as f32, camera.size_px[1] as f32);
                    let (s, c) = camera.bearing.sin_cos();
                    gl.uniform_2_f32(fill.u("u_turn"), c as f32, s as f32);
                    gl.uniform_4_f32_slice(fill.u("u_phase"), &self.pattern_phases(palette, camera));
                }
                Mode::Screen | Mode::State(_) => {}
            }
            gl.bind_vertex_array(Some(mesh.fill_vao));
            gl.draw_elements(glow::TRIANGLES, mesh.fill_count, glow::UNSIGNED_INT, 0);

            gl.use_program(Some(line.program));
            gl.uniform_matrix_4_f32_slice(line.u("u_matrix"), false, &matrix);
            gl.uniform_matrix_4_f32_slice(line.u("u_post"), false, &post);
            gl.uniform_2_f32(line.u("u_viewport"), camera.size_px[0] as f32, camera.size_px[1] as f32);
            gl.uniform_1_i32(line.u("u_palette"), 0);
            state_uniforms(gl, line);
            let min_half = match mode {
                Mode::Screen | Mode::State(_) => 0.0,
                Mode::Pick { layer, min_half, .. } => {
                    gl.uniform_1_u32(line.u("u_layer"), layer);
                    min_half
                }
            };
            gl.uniform_1_f32(line.u("u_min_half"), min_half);
            gl.uniform_1_f32(line.u("u_unit_px"), (mesh.frame.half_span / camera.units_per_px) as f32);
            if mesh.line_count > 0 {
                gl.bind_vertex_array(Some(mesh.line_vao));
                gl.draw_arrays_instanced(glow::TRIANGLE_STRIP, 0, 4, mesh.line_count);
            }

            if mesh.point_count > 0 {
                gl.use_program(Some(marker.program));
                gl.uniform_matrix_4_f32_slice(marker.u("u_matrix"), false, &matrix);
                gl.uniform_matrix_4_f32_slice(marker.u("u_post"), false, &post);
                gl.uniform_2_f32(marker.u("u_viewport"), camera.size_px[0] as f32, camera.size_px[1] as f32);
                gl.uniform_1_i32(marker.u("u_palette"), 0);
                gl.uniform_1_i32(marker.u("u_table"), 2);
                gl.uniform_1_i32(marker.u("u_atlas"), 3);
                gl.uniform_1_f32(marker.u("u_min_half"), min_half);
                state_uniforms(gl, marker);
                if let Mode::Pick { layer, .. } = mode {
                    gl.uniform_1_u32(marker.u("u_layer"), layer);
                }
                gl.bind_vertex_array(Some(mesh.point_vao));
                gl.draw_arrays_instanced(glow::TRIANGLE_STRIP, 0, 4, mesh.point_count);
            }

            gl.bind_vertex_array(Some(self.square_vao));
            self.settle(gl, &matrix, &post, grow, square);
            gl.bind_vertex_array(None);
        }
    }

    /// Where the camera's centre lies in each patterned class's repeat, as
    /// `FILL_FS`'s `u_phase` takes it: two classes a `vec4`, 0 for a class
    /// that draws no pattern.
    fn pattern_phases(&self, palette: &GpuPalette, camera: &Camera) -> [f32; 2 * CLASSES] {
        let mut phases = [0.0; 2 * CLASSES];
        for &(class, pattern) in &palette.patterns {
            let size = self.symbols.sizes.get(pattern.symbol.0 as usize).copied().unwrap_or_default();
            if let Some(cell) = pattern.cell(size) {
                let [x, y] = pattern_phase(camera, cell);
                phases[2 * class as usize..][..2].copy_from_slice(&[x as f32, y as f32]);
            }
        }
        phases
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
    pub fn free(self, gl: &Context) {
        let gl = &gl.raw;
        // SAFETY: deletes objects this renderer created and alone refers to.
        unsafe {
            let markers = [self.marker, self.marker_pick, self.marker_state, self.marker_pick_state];
            for p in [self.fill, self.fill_pick, self.fill_state, self.fill_pick_state, self.line, self.line_pick, self.line_state, self.line_pick_state, self.raster, self.stencil].into_iter().chain(markers) {
                gl.delete_program(p.program);
            }
            self.symbols.free(gl);
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
/// A pick can follow a feature state ([`Picker::draw_state`]), as a screen
/// draw does. A fill picks wherever it covers, even where its class's fill
/// is transparent, and a polygon drawn only by its outline picks inside; a
/// line or point picks only where its class gives it a width or size. So
/// an embedder leaves out, or draws through a state, a feature that shouldn't
/// pick. Rasters are not picked, and do not hide the features beneath
/// them from a pick. Reading back stalls the GPU pipeline, which is free for a click but
/// wants throttling for hover.
pub struct Picker {
    framebuffer: glow::Framebuffer,
    renderbuffers: [glow::Renderbuffer; 2],
    post: [f32; 16],
    min_half: f32,
}

impl Picker {
    pub fn new(gl: &Context) -> Result<Picker, String> {
        let gl = &gl.raw;
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
    pub fn begin(&mut self, gl: &Context, camera: &Camera, px: [f64; 2], tolerance_px: f32) {
        let gl = &gl.raw;
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
    pub fn begin_layer(&self, gl: &Context) {
        let gl = &gl.raw;
        // SAFETY: a stencil clear on the picker's framebuffer.
        unsafe { gl.clear_buffer_depth_stencil(glow::DEPTH_STENCIL, 0, 1.0, 0) }
    }

    /// Draws `mesh` for picking, as `layer`. Call in the order the screen
    /// draws.
    pub fn draw(&self, gl: &Context, renderer: &Renderer, mesh: &GpuMesh, camera: &Camera, palette: &GpuPalette, layer: u32) {
        let gl = &gl.raw;
        renderer.draw_mesh(gl, mesh, camera, palette, Mode::Pick { post: self.post, layer, min_half: self.min_half, state: None });
    }

    /// Draws for picking only the features of `mesh` whose flags in `state`
    /// share a bit with `mask`, as [`Renderer::draw_state`] draws them. The
    /// rest pick nothing, so what lies beneath them is picked instead: given
    /// the state and mask the screen drew a layer with, a pick sees exactly
    /// what is on screen.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_state(&self, gl: &Context, renderer: &Renderer, mesh: &GpuMesh, camera: &Camera, palette: &GpuPalette, layer: u32, state: &GpuState, mask: u8) {
        let gl = &gl.raw;
        let state = Some(StateTest::new(state, mask));
        renderer.draw_mesh(gl, mesh, camera, palette, Mode::Pick { post: self.post, layer, min_half: self.min_half, state });
    }

    /// Reads the pixel back and rebinds the default framebuffer. `None` if
    /// nothing was drawn there. The next frame's [`Renderer::begin_frame`]
    /// restores the viewport and blending.
    pub fn finish(&self, gl: &Context) -> Option<Pick> {
        let gl = &gl.raw;
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

    pub fn free(self, gl: &Context) {
        let gl = &gl.raw;
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

/// A state texture's texels: `flags`, padded with zeros to whole rows, and
/// at least one row.
fn state_texels(flags: &[u8]) -> Vec<u8> {
    let mut texels = flags.to_vec();
    texels.resize(flags.len().div_ceil(STATE_ROW).max(1) * STATE_ROW, 0);
    texels
}

/// The palette's texels, a row each of:
///
/// 0. fill colours,
/// 1. line colours,
/// 2. (line width, point size, 0, 0),
/// 3. markers: (symbol, rotation in degrees, 1 against the map or 0 the
///    screen, 0),
/// 4. and 5. dash lengths, four each, 0 past the dash's count,
/// 6. patterns: (symbol, repeat, dash phase, dash count).
///
/// A symbol is its id + 1, so 0 is none and a default style is all zeros;
/// an `f32` holds every id exactly.
fn palette_texels(palette: &Palette) -> Vec<u8> {
    let s = palette.styles();
    (0..PALETTE_ROWS as usize).flat_map(|row| s.iter().flat_map(move |s| f32_bytes(&palette_texel(s, row)))).collect()
}

/// One class's texel in `row` of the palette.
fn palette_texel(s: &Style, row: usize) -> [f32; 4] {
    let symbol = |id: SymbolId| id.0 as f32 + 1.0;
    let mut lengths = [0.0; DASH_LENGTHS];
    if let Some(d) = s.dash {
        lengths[..d.lengths().len()].copy_from_slice(d.lengths());
    }
    match row {
        0 => s.fill,
        1 => s.line,
        2 => [s.line_width, s.point_size, 0.0, 0.0],
        3 => s.marker.map_or([0.0; 4], |m| [symbol(m.symbol), m.rotation, (m.align == Align::Map) as u8 as f32, 0.0]),
        4 | 5 => std::array::from_fn(|i| lengths[(row - 4) * 4 + i]),
        _ => {
            let (id, repeat) = s.pattern.map_or((0.0, 0.0), |p| (symbol(p.symbol), p.repeat));
            let (phase, count) = s.dash.map_or((0.0, 0.0), |d| (d.phase, d.lengths().len() as f32));
            [id, repeat, phase, count]
        }
    }
}

/// The classes of `palette` with a pattern.
fn patterns(palette: &Palette) -> Vec<(u8, Pattern)> {
    palette.styles().iter().enumerate().filter_map(|(class, s)| s.pattern.map(|p| (class as u8, p))).collect()
}

/// The symbol table's rows and texels: id `n` at column `2 * (n % 256)` of
/// row `n / 256`, as (x, y, width, height) in atlas pixels, then (anchor x,
/// anchor y, 1 for a mask or 0 for colour, 0). An id not held is all zeros,
/// so has no width. As many rows as the highest id held needs, and at least
/// one.
fn symbol_table_texels(symbols: &Symbols) -> (usize, Vec<u8>) {
    let rows = symbols.iter().last().map_or(1, |(id, _, _)| id.0 as usize / SYMBOL_ROW + 1);
    let mut texels = vec![0.0f32; rows * SYMBOL_ROW * 8];
    for (id, symbol, r) in symbols.iter() {
        let mask = (symbol.kind == Kind::Mask) as u8 as f32;
        let at = id.0 as usize * 8;
        texels[at..at + 8].copy_from_slice(&[r.x as f32, r.y as f32, r.width as f32, r.height as f32, symbol.anchor[0], symbol.anchor[1], mask, 0.0]);
    }
    (rows, f32_bytes(&texels))
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_ne_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texel(texels: &[u8], width: usize, x: usize, y: usize) -> [f32; 4] {
        let at = (y * width + x) * 16;
        std::array::from_fn(|i| f32::from_ne_bytes(texels[at + i * 4..at + i * 4 + 4].try_into().unwrap()))
    }

    #[test]
    fn palette_texels_hold_a_row_each() {
        let mut symbols = Symbols::new();
        let pin = symbols.add(crate::Symbol::new(1, 1, vec![0; 4]).unwrap()).unwrap();
        let hatch = symbols.add(crate::Symbol::new(1, 1, vec![0; 4]).unwrap()).unwrap();
        let mut palette = Palette::new();
        let s = Style {
            fill: [0.1, 0.2, 0.3, 0.4],
            line_width: 2.0,
            point_size: 6.0,
            marker: Some(crate::Marker { rotation: 45.0, align: Align::Map, ..crate::Marker::new(pin) }),
            dash: crate::Dash::new(&[4.0, 2.0, 0.0, 2.0, 1.0], 3.0),
            pattern: Some(crate::Pattern::new(hatch, 12.0)),
            ..Style::default()
        };
        palette.set(9, s);
        let texels = palette_texels(&palette);
        assert_eq!(texels.len(), CLASSES * PALETTE_ROWS as usize * 16);
        let at = |row| texel(&texels, CLASSES, 9, row);
        assert_eq!(at(0), [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(at(2), [2.0, 6.0, 0.0, 0.0]);
        assert_eq!(at(3), [1.0, 45.0, 1.0, 0.0]);
        assert_eq!(at(4), [4.0, 2.0, 0.0, 2.0]);
        assert_eq!(at(5), [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(at(6), [2.0, 12.0, 3.0, 5.0]);
        // A default style is all zeros: no marker, no dash, no pattern.
        assert!((0..PALETTE_ROWS as usize).all(|row| texel(&texels, CLASSES, 8, row) == [0.0; 4]));
    }

    #[test]
    fn the_symbol_table_grows_by_rows_of_256_ids() {
        let mut symbols = Symbols::new();
        assert_eq!(symbol_table_texels(&symbols).0, 1);
        let mut first = crate::Symbol::new(3, 5, vec![0; 60]).unwrap();
        first.anchor = [0.5, 1.0];
        first.kind = Kind::Mask;
        let first = symbols.add(first).unwrap();
        for _ in 0..SYMBOL_ROW {
            let id = symbols.add(crate::Symbol::new(1, 1, vec![0; 4]).unwrap()).unwrap();
            symbols.remove(id);
        }
        let last = symbols.add(crate::Symbol::new(1, 1, vec![0; 4]).unwrap()).unwrap();
        assert_eq!(last.0 as usize, SYMBOL_ROW + 1);
        let (rows, texels) = symbol_table_texels(&symbols);
        assert_eq!((rows, texels.len()), (2, 2 * SYMBOL_ROW * 2 * 16));
        let r = symbols.rect(first).unwrap();
        assert_eq!(texel(&texels, 2 * SYMBOL_ROW, 0, 0), [r.x as f32, r.y as f32, 3.0, 5.0]);
        assert_eq!(texel(&texels, 2 * SYMBOL_ROW, 1, 0), [0.5, 1.0, 1.0, 0.0]);
        // A removed id has no width; the last sits in row 1, column 2.
        assert_eq!(texel(&texels, 2 * SYMBOL_ROW, 2, 0), [0.0; 4]);
        assert_eq!(texel(&texels, 2 * SYMBOL_ROW, 2, 1)[2], 1.0);
    }

    #[test]
    fn state_texels_fill_whole_rows() {
        assert_eq!(state_texels(&[]).len(), STATE_ROW);
        assert_eq!(state_texels(&[1; STATE_ROW]).len(), STATE_ROW);
        let texels = state_texels(&[7; STATE_ROW + 1]);
        assert_eq!(texels.len(), 2 * STATE_ROW);
        assert_eq!((texels[STATE_ROW], texels[STATE_ROW + 1]), (7, 0));
    }
}
