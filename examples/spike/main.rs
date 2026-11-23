//! The native half of the GL spike (`plans/rendering.org` § V1 SLICE):
//! the Massachusetts counties, prepared tile by tile through nazca's camera,
//! tile selector and tile cache and drawn by its `gl` backend in a winit
//! window over an OpenGL 3.3 core context. Drag to pan, scroll to zoom about
//! the cursor, right-drag to rotate about the window's centre, click to pick
//! a county and toggle its selection, press H to toggle a hover state on the
//! county picked last, press F to show every other county, press M to cycle
//! the markers, press D to cycle the outlines (solid, dashed, dotted,
//! translucent), and press R to cycle the raster overlay.
//!
//!   cargo run -p nazca --features gl --example spike
//!   cargo run -p nazca --features gl --example spike -- --screenshot out.png
//!   cargo run -p nazca --features gl --example spike -- --bearing 30 --raster continuous
//!
//! `--screenshot` draws one frame, writes it as a PNG and exits. `--bearing`
//! opens the map turned that many degrees, east of north up. `--raster`
//! (`continuous` or `categorical`) opens with that overlay. `--pick X,Y`
//! picks at that window pixel after the first frame and prints the county.
//! `--zoom F` opens zoomed in F times about the window's centre. `--select
//! X,Y` and `--hover X,Y` (each repeatable) give the county at that pixel a
//! state after the first frame. `--filter` opens showing every other county,
//! as F toggles. `--svg out.svg` writes the first frame as an SVG, through
//! the same states, and exits. `--check-state` reads back and picks a
//! synthetic grid of features drawn by state, reports and exits. `--own-gl`
//! makes GL calls of the spike's own between two of nazca's layers, through
//! a `glow` binding of its own over the same context, as an embedder drawing
//! something of its own would: a square in the top left corner. `--markers`
//! (`colour` or `mask`) opens with the counties' markers, as M cycles them.
//! `--check-markers` checks the guard band check and picking a marker by its
//! image, reports and exits. `--centre NAME` centres the view on that
//! county's marker, and `--level Z` sets the zoom to MapLibre's zoom `Z`
//! (the grid's level, fractional), as a tile-edge check aims a view.
//! `--outlines` (`dashed`, `dotted` or `translucent`) opens with those
//! outlines, as D cycles them. `--check-dashes` checks shared borders and
//! dashes by reading the framebuffer, measures line starts' seams, the
//! phase of a line far along it and the cost of matching borders, reports
//! and exits.

mod data;
mod tiles;

use data::{Markers, Outlines, Overlay};
use glutin::config::{ConfigTemplateBuilder, GlConfig};
use glutin::context::{ContextApi, ContextAttributesBuilder, GlProfile, NotCurrentGlContext, PossiblyCurrentContext, Version};
use glutin::display::{GetGlDisplay, GlDisplay};
use glutin::surface::{GlSurface, Surface, WindowSurface};
use glutin_winit::{DisplayBuilder, GlWindow};
use glow::HasContext;
use nazca::Camera;
use nazca::gl::{Context, Renderer};
use raw_window_handle::HasWindowHandle;
use std::num::NonZeroU32;
use tiles::{HOVERED, SELECTED, Stats, TiledMap};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

struct Gfx {
    window: Window,
    surface: Surface<WindowSurface>,
    context: PossiblyCurrentContext,
    gl: Context,
    renderer: Renderer,
    map: TiledMap,
    own: Option<OwnGl>,
}

impl Gfx {
    fn draw(&mut self, camera: &Camera) -> Stats {
        match &self.own {
            Some(own) => self.map.draw_with(&self.gl, &self.renderer, camera, &mut || own.draw(camera.size_px[1])),
            None => self.map.draw(&self.gl, &self.renderer, camera),
        }
    }
}

/// The spike's own GL objects and the binding it draws them through.
struct OwnGl {
    gl: glow::Context,
    vao: glow::VertexArray,
    texture: glow::Texture,
}

impl OwnGl {
    fn new(gl: glow::Context) -> OwnGl {
        // SAFETY: creates objects of its own on the current context.
        let (vao, texture) = unsafe { (gl.create_vertex_array().unwrap(), gl.create_texture().unwrap()) };
        OwnGl { gl, vao, texture }
    }

    /// Clears a square in the top left corner of a framebuffer `height`
    /// pixels high. It leaves its own array, texture and texture unit bound
    /// and no program, which nazca's draws must not rely on, and restores the
    /// scissor test, the one part of the frame's state it changes (`gl.rs`'s
    /// module docs).
    fn draw(&self, height: u32) {
        // SAFETY: binds objects of its own and clears within a scissor, on
        // the context nazca draws with, current on this thread.
        unsafe {
            self.gl.use_program(None);
            self.gl.bind_vertex_array(Some(self.vao));
            self.gl.active_texture(glow::TEXTURE3);
            self.gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
            self.gl.enable(glow::SCISSOR_TEST);
            self.gl.scissor(16, height as i32 - 56, 40, 40);
            self.gl.clear_color(0.9, 0.1, 0.6, 1.0);
            self.gl.clear(glow::COLOR_BUFFER_BIT);
            self.gl.disable(glow::SCISSOR_TEST);
        }
    }
}

struct Options {
    screenshot: Option<String>,
    bearing: f64,
    overlay: Overlay,
    pick: Option<[f64; 2]>,
    /// Pixels whose counties get a state after the first frame.
    states: Vec<([f64; 2], u8)>,
    check_state: bool,
    filter: bool,
    own_gl: bool,
    svg: Option<String>,
    zoom: f64,
    markers: Markers,
    check_markers: bool,
    centre: Option<String>,
    level: Option<f64>,
    outlines: Outlines,
    check_dashes: bool,
}

struct App {
    options: Options,
    gfx: Option<Gfx>,
    camera: Option<Camera>,
    cursor: PhysicalPosition<f64>,
    dragging: bool,
    /// Where a left press began, to tell a click from a drag.
    pressed_at: Option<PhysicalPosition<f64>>,
    rotating: bool,
}

impl App {
    fn redraw(&mut self) {
        let (Some(g), Some(camera)) = (&mut self.gfx, &self.camera) else { return };
        if self.options.check_state {
            match tiles::check_state(&g.gl, &g.renderer, camera.size_px) {
                Ok(report) => println!("{report}"),
                Err(e) => {
                    eprintln!("state check failed: {e}");
                    std::process::exit(1);
                }
            }
            std::process::exit(0);
        }
        if self.options.filter {
            self.options.filter = false;
            g.map.set_filter(&g.gl, &g.renderer, true);
        }
        if self.options.check_dashes {
            let start = std::time::Instant::now();
            let clock = move || start.elapsed().as_secs_f64() * 1e3;
            match tiles::check_dashes(&g.gl, &g.renderer, g.map.counties(), camera.size_px, &clock) {
                Ok(report) => println!("{report}"),
                Err(e) => {
                    eprintln!("dash check failed: {e}");
                    std::process::exit(1);
                }
            }
            std::process::exit(0);
        }
        if self.options.outlines != Outlines::Solid {
            g.map.set_outlines(&g.gl, &g.renderer, std::mem::replace(&mut self.options.outlines, Outlines::Solid));
        }
        if self.options.markers != Markers::None {
            g.map.set_markers(&g.gl, &g.renderer, std::mem::replace(&mut self.options.markers, Markers::None));
        }
        if self.options.check_markers {
            match tiles::check_markers(&mut g.map, &g.gl, &g.renderer, camera) {
                Ok(report) => println!("{report}"),
                Err(e) => {
                    eprintln!("marker check failed: {e}");
                    std::process::exit(1);
                }
            }
            std::process::exit(0);
        }
        let stats = g.draw(camera);
        if stats.prepared > 0 {
            println!(
                "level {}: {} tiles drawn, {} prepared; {} cached, {} KB",
                stats.level,
                stats.drawn,
                stats.prepared,
                stats.cached,
                stats.cached_bytes / 1024
            );
        }
        if let Some(px) = self.options.pick.take() {
            report(g.map.pick(&g.gl, &g.renderer, camera, px).as_ref(), px);
        }
        for (px, bit) in std::mem::take(&mut self.options.states) {
            let pick = g.map.pick(&g.gl, &g.renderer, camera, px);
            report(pick.as_ref(), px);
            if let Some((_, id)) = pick {
                g.map.toggle(&g.gl, &g.renderer, id, bit);
            }
        }
        if let Some(path) = self.options.svg.take() {
            std::fs::write(&path, g.map.svg(camera)).unwrap();
            println!("wrote {path}");
            if self.options.screenshot.is_none() {
                std::process::exit(0);
            }
        }
        if let Some(path) = self.options.screenshot.take() {
            let [w, h] = camera.size_px;
            // The pick (if any) rebound the window; draw again so the read
            // sees a frame drawn after it.
            g.draw(camera);
            let pixels = g.gl.read_pixels([w, h]);
            // GL's rows run bottom-up; PNG's top-down.
            let flipped: Vec<u8> = pixels.chunks((w * 4) as usize).rev().flatten().copied().collect();
            let file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
            let mut encoder = png::Encoder::new(file, w, h);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.write_header().unwrap().write_image_data(&flipped).unwrap();
            println!("wrote {path} ({w}x{h})");
            std::process::exit(0);
        }
        g.surface.swap_buffers(&g.context).unwrap();
    }
}

fn report(pick: Option<&(String, u32)>, px: [f64; 2]) {
    match pick {
        Some((name, id)) => println!("picked {name} (feature {id}) at {px:?}"),
        None => println!("picked nothing at {px:?}"),
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let attributes = Window::default_attributes().with_title("nazca spike").with_inner_size(LogicalSize::new(900.0, 700.0));
        // A stencil buffer: each tile claims its pixels in it.
        let template = ConfigTemplateBuilder::new().with_multisampling(4).with_stencil_size(8);
        let (window, config) = DisplayBuilder::new()
            .with_window_attributes(Some(attributes))
            .build(event_loop, template, |configs| configs.max_by_key(|c| c.num_samples()).unwrap())
            .unwrap();
        let window = window.unwrap();
        let display = config.display();
        let context_attributes = ContextAttributesBuilder::new()
            .with_profile(GlProfile::Core)
            .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3))))
            .build(window.window_handle().ok().map(|h| h.as_raw()));
        let surface_attributes = window.build_surface_attributes(Default::default()).unwrap();
        // SAFETY: the context and surface are made from this display's own
        // config, for a window that outlives them (both live in `Gfx`); the
        // loader's functions are an OpenGL 3.3 core context's, current on
        // this thread from here on.
        let (context, surface, gl, own) = unsafe {
            let context = display.create_context(&config, &context_attributes).unwrap();
            let surface = display.create_window_surface(&config, &surface_attributes).unwrap();
            let context = context.make_current(&surface).unwrap();
            let gl = Context::from_loader(|s| display.get_proc_address(s));
            let own = self.options.own_gl.then(|| OwnGl::new(glow::Context::from_loader_function_cstr(|s| display.get_proc_address(s))));
            (context, surface, gl, own)
        };
        println!("GL: {}, stencil bits: {}", gl.version(), config.stencil_size());

        let mut renderer = Renderer::new(&gl).unwrap();
        let map = TiledMap::new(&gl, &mut renderer, 16 << 20, self.options.overlay);

        // Fit the counties, with a margin.
        let size = window.inner_size();
        let mut camera = Camera::new([0.0, 0.0], 1.0, [size.width, size.height]);
        camera.bearing = self.options.bearing.to_radians();
        camera.fit_bounds(map.bounds(), 0.04 * size.width.min(size.height) as f64);
        camera.zoom_about([size.width as f64 / 2.0, size.height as f64 / 2.0], self.options.zoom);
        if let Some(name) = &self.options.centre {
            camera.center = map.marker(name).unwrap_or_else(|| panic!("no county {name}"));
        }
        if let Some(z) = self.options.level {
            // The 512-pixel Web Mercator grid's resolution at level `z`.
            camera.units_per_px = std::f64::consts::TAU * 6_378_137.0 / 512.0 / 2f64.powf(z);
        }
        self.camera = Some(camera);
        window.request_redraw();
        self.gfx = Some(Gfx { window, surface, context, gl, renderer, map, own });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let (Some(camera), Some(g)) = (self.camera.as_mut(), self.gfx.as_mut()) else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                g.surface.resize(&g.context, NonZeroU32::new(size.width).unwrap(), NonZeroU32::new(size.height).unwrap());
                camera.resize([size.width, size.height]);
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.dragging = state == ElementState::Pressed;
                match state {
                    ElementState::Pressed => self.pressed_at = Some(self.cursor),
                    ElementState::Released => {
                        let start = self.pressed_at.take();
                        // A press and release in nearly one place is a click.
                        if start.is_some_and(|s| (s.x - self.cursor.x).abs() + (s.y - self.cursor.y).abs() < 4.0) {
                            let px = [self.cursor.x, self.cursor.y];
                            let pick = g.map.pick(&g.gl, &g.renderer, camera, px);
                            report(pick.as_ref(), px);
                            if let Some((_, id)) = pick {
                                g.map.toggle(&g.gl, &g.renderer, id, SELECTED);
                            }
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Right, .. } => {
                self.rotating = state == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let delta = [position.x - self.cursor.x, position.y - self.cursor.y];
                if self.dragging {
                    camera.pan_px(delta);
                } else if self.rotating {
                    // A full turn per window width.
                    let [w, h] = camera.size_px.map(|v| v as f64);
                    camera.rotate_about([w / 2.0, h / 2.0], delta[0] / w * std::f64::consts::TAU);
                }
                self.cursor = position;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 50.0,
                };
                camera.zoom_about([self.cursor.x, self.cursor.y], 1.15f64.powf(steps));
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => match event.logical_key {
                Key::Character(c) if c.eq_ignore_ascii_case("r") => {
                    let next = g.map.overlay().next();
                    g.map.set_overlay(&g.gl, &g.renderer, next);
                    println!("overlay: {next:?}");
                }
                Key::Character(c) if c.eq_ignore_ascii_case("h") => g.map.toggle_last(&g.gl, &g.renderer, HOVERED),
                Key::Character(c) if c.eq_ignore_ascii_case("m") => {
                    let next = g.map.markers().next();
                    g.map.set_markers(&g.gl, &g.renderer, next);
                    println!("markers: {next:?}");
                }
                Key::Character(c) if c.eq_ignore_ascii_case("d") => {
                    let next = g.map.outlines().next();
                    g.map.set_outlines(&g.gl, &g.renderer, next);
                    println!("outlines: {next:?}");
                }
                Key::Character(c) if c.eq_ignore_ascii_case("f") => {
                    let on = !g.map.filtered();
                    g.map.set_filter(&g.gl, &g.renderer, on);
                    println!("filter: {}", if on { "every other county" } else { "off" });
                }
                Key::Named(NamedKey::Escape) => event_loop.exit(),
                _ => return,
            },
            WindowEvent::RedrawRequested => {
                self.redraw();
                return;
            }
            _ => return,
        }
        if let Some(g) = &self.gfx {
            g.window.request_redraw();
        }
    }
}

fn main() {
    let usage = || -> ! {
        eprintln!("usage: spike [--screenshot out.png] [--bearing degrees] [--raster continuous|categorical] [--pick X,Y] [--select X,Y] [--hover X,Y] [--zoom F] [--filter] [--svg out.svg] [--check-state] [--own-gl] [--markers colour|mask] [--check-markers] [--centre NAME] [--level Z] [--outlines dashed|dotted|translucent] [--check-dashes]");
        std::process::exit(2);
    };
    let mut options = Options {
        screenshot: None,
        bearing: 0.0,
        overlay: Overlay::None,
        pick: None,
        states: Vec::new(),
        check_state: false,
        filter: false,
        own_gl: false,
        svg: None,
        zoom: 1.0,
        markers: Markers::None,
        check_markers: false,
        centre: None,
        level: None,
        outlines: Outlines::Solid,
        check_dashes: false,
    };
    let mut args = std::env::args().skip(1);
    let point = |value: &str| -> [f64; 2] {
        let xy: Vec<f64> = value.split(',').map(|v| v.parse().unwrap_or_else(|_| usage())).collect();
        let [x, y] = xy[..] else { usage() };
        [x, y]
    };
    while let Some(flag) = args.next() {
        if flag == "--check-state" {
            options.check_state = true;
            continue;
        }
        if flag == "--filter" {
            options.filter = true;
            continue;
        }
        if flag == "--own-gl" {
            options.own_gl = true;
            continue;
        }
        if flag == "--check-markers" {
            options.check_markers = true;
            continue;
        }
        if flag == "--check-dashes" {
            options.check_dashes = true;
            continue;
        }
        let value = args.next().unwrap_or_else(|| usage());
        match flag.as_str() {
            "--screenshot" => options.screenshot = Some(value),
            "--svg" => options.svg = Some(value),
            "--bearing" => options.bearing = value.parse().unwrap_or_else(|_| usage()),
            "--zoom" => options.zoom = value.parse().unwrap_or_else(|_| usage()),
            "--raster" => {
                options.overlay = match value.as_str() {
                    "continuous" => Overlay::Continuous,
                    "categorical" => Overlay::Categorical,
                    _ => usage(),
                }
            }
            "--markers" => {
                options.markers = match value.as_str() {
                    "colour" => Markers::Colour,
                    "mask" => Markers::Mask,
                    _ => usage(),
                }
            }
            "--outlines" => {
                options.outlines = match value.as_str() {
                    "dashed" => Outlines::Dashed,
                    "dotted" => Outlines::Dotted,
                    "translucent" => Outlines::Translucent,
                    _ => usage(),
                }
            }
            "--centre" => options.centre = Some(value),
            "--level" => options.level = Some(value.parse().unwrap_or_else(|_| usage())),
            "--pick" => options.pick = Some(point(&value)),
            "--select" => options.states.push((point(&value), SELECTED)),
            "--hover" => options.states.push((point(&value), HOVERED)),
            _ => usage(),
        }
    }
    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        options,
        gfx: None,
        camera: None,
        cursor: PhysicalPosition::new(0.0, 0.0),
        dragging: false,
        pressed_at: None,
        rotating: false,
    };
    event_loop.run_app(&mut app).unwrap();
}
