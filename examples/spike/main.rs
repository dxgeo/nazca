//! The native half of the `glow` spike (`plans/rendering.org` § V1 SLICE):
//! the Massachusetts counties, prepared tile by tile through nazca's camera,
//! tile selector and tile cache and drawn by its `gl` backend in a winit
//! window over an OpenGL 3.3 core context. Drag to pan, scroll to zoom about
//! the cursor, right-drag to rotate about the window's centre.
//!
//!   cargo run -p nazca --features gl --example spike
//!   cargo run -p nazca --features gl --example spike -- --screenshot out.png
//!   cargo run -p nazca --features gl --example spike -- --bearing 30
//!
//! `--screenshot` draws one frame, writes it as a PNG and exits. `--bearing`
//! opens the map turned that many degrees, east of north up.

mod data;
mod tiles;

use glutin::config::{ConfigTemplateBuilder, GlConfig};
use glutin::context::{ContextApi, ContextAttributesBuilder, GlProfile, NotCurrentGlContext, PossiblyCurrentContext, Version};
use glutin::display::{GetGlDisplay, GlDisplay};
use glutin::surface::{GlSurface, Surface, WindowSurface};
use glutin_winit::{DisplayBuilder, GlWindow};
use glow::HasContext;
use nazca::Camera;
use nazca::gl::Renderer;
use raw_window_handle::HasWindowHandle;
use std::num::NonZeroU32;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use tiles::TiledMap;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

struct Gfx {
    window: Window,
    surface: Surface<WindowSurface>,
    context: PossiblyCurrentContext,
    gl: glow::Context,
    renderer: Renderer,
}

struct App {
    screenshot: Option<String>,
    bearing: f64,
    map: TiledMap,
    gfx: Option<Gfx>,
    camera: Option<Camera>,
    cursor: PhysicalPosition<f64>,
    dragging: bool,
    rotating: bool,
}

impl App {
    fn redraw(&mut self) {
        let (Some(g), Some(camera)) = (&self.gfx, &self.camera) else { return };
        let [w, h] = camera.size_px;
        let [r, gr, b, a] = data::BACKGROUND;
        // SAFETY: plain state and clear calls on the current context.
        unsafe {
            g.gl.viewport(0, 0, w as i32, h as i32);
            g.gl.clear_color(r, gr, b, a);
            g.gl.clear(glow::COLOR_BUFFER_BIT);
        }
        let stats = self.map.draw(&g.gl, &g.renderer, camera);
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

        if let Some(path) = self.screenshot.take() {
            let mut pixels = vec![0u8; (w * h * 4) as usize];
            // SAFETY: reads the frame just drawn into a buffer sized for it.
            unsafe {
                g.gl.read_pixels(0, 0, w as i32, h as i32, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(Some(&mut pixels)));
            }
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

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let attributes = Window::default_attributes().with_title("nazca spike").with_inner_size(LogicalSize::new(900.0, 700.0));
        let template = ConfigTemplateBuilder::new().with_multisampling(4);
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
        // config, for a window that outlives them (both live in `Gfx`).
        let (context, surface, gl) = unsafe {
            let context = display.create_context(&config, &context_attributes).unwrap();
            let surface = display.create_window_surface(&config, &surface_attributes).unwrap();
            let context = context.make_current(&surface).unwrap();
            let gl = glow::Context::from_loader_function_cstr(|s| display.get_proc_address(s));
            (context, surface, gl)
        };
        println!("GL: {}", unsafe { gl.get_parameter_string(glow::VERSION) });

        let renderer = Renderer::new(&gl).unwrap();

        // Fit the counties, with a margin.
        let size = window.inner_size();
        let mut camera = Camera::new([0.0, 0.0], 1.0, [size.width, size.height]);
        camera.bearing = self.bearing.to_radians();
        camera.fit_bounds(self.map.bounds(), 0.04 * size.width.min(size.height) as f64);
        self.camera = Some(camera);
        window.request_redraw();
        self.gfx = Some(Gfx { window, surface, context, gl, renderer });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(camera) = self.camera.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                if let Some(g) = &self.gfx {
                    g.surface.resize(&g.context, NonZeroU32::new(size.width).unwrap(), NonZeroU32::new(size.height).unwrap());
                }
                camera.resize([size.width, size.height]);
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.dragging = state == ElementState::Pressed;
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
        eprintln!("usage: spike [--screenshot out.png] [--bearing degrees]");
        std::process::exit(2);
    };
    let (mut screenshot, mut bearing) = (None, 0.0);
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().unwrap_or_else(|| usage());
        match flag.as_str() {
            "--screenshot" => screenshot = Some(value),
            "--bearing" => bearing = value.parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
    }
    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        screenshot,
        bearing,
        map: TiledMap::new(16 << 20),
        gfx: None,
        camera: None,
        cursor: PhysicalPosition::new(0.0, 0.0),
        dragging: false,
        rotating: false,
    };
    event_loop.run_app(&mut app).unwrap();
}
