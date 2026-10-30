//! The native half of the `glow` spike (`plans/rendering.org` § V1 SLICE):
//! the Massachusetts counties, drawn by nazca's `gl` backend in a winit
//! window over an OpenGL 3.3 core context. Drag to pan, scroll to zoom about
//! the cursor.
//!
//!   cargo run -p nazca --features gl --example spike
//!   cargo run -p nazca --features gl --example spike -- --screenshot out.png
//!
//! `--screenshot` draws one frame, writes it as a PNG and exits.

mod data;

use glutin::config::{ConfigTemplateBuilder, GlConfig};
use glutin::context::{ContextApi, ContextAttributesBuilder, GlProfile, NotCurrentGlContext, PossiblyCurrentContext, Version};
use glutin::display::{GetGlDisplay, GlDisplay};
use glutin::surface::{GlSurface, Surface, WindowSurface};
use glutin_winit::{DisplayBuilder, GlWindow};
use glow::HasContext;
use nazca::gl::{GpuMesh, Renderer, View};
use nazca::grid::Rect;
use raw_window_handle::HasWindowHandle;
use std::num::NonZeroU32;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

struct Gfx {
    window: Window,
    surface: Surface<WindowSurface>,
    context: PossiblyCurrentContext,
    gl: glow::Context,
    renderer: Renderer,
    mesh: GpuMesh,
}

struct App {
    screenshot: Option<String>,
    bounds: Rect,
    gfx: Option<Gfx>,
    view: Option<View>,
    cursor: PhysicalPosition<f64>,
    dragging: bool,
}

impl App {
    fn redraw(&mut self) {
        let (Some(g), Some(view)) = (&self.gfx, &self.view) else { return };
        let [w, h] = view.size_px;
        let [r, gr, b, a] = data::BACKGROUND;
        // SAFETY: plain state and clear calls on the current context.
        unsafe {
            g.gl.viewport(0, 0, w as i32, h as i32);
            g.gl.clear_color(r, gr, b, a);
            g.gl.clear(glow::COLOR_BUFFER_BIT);
        }
        g.renderer.draw(&g.gl, &g.mesh, view, &data::PALETTE, data::OUTLINE);

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

    /// The map point under a pixel position.
    fn map_point(view: &View, px: PhysicalPosition<f64>) -> [f64; 2] {
        [
            view.center[0] + (px.x - view.size_px[0] as f64 / 2.0) * view.units_per_px,
            view.center[1] - (px.y - view.size_px[1] as f64 / 2.0) * view.units_per_px,
        ]
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

        let (mesh, _) = data::counties();
        println!(
            "{} vertices, {} triangles, {} outline segments",
            mesh.vertex_count(),
            mesh.fills.len() / 3,
            mesh.outlines.len() / 2
        );
        let renderer = Renderer::new(&gl).unwrap();
        let gpu = renderer.upload(&gl, &mesh).unwrap();

        // Fit the counties, with a margin.
        let size = window.inner_size();
        let b = self.bounds;
        let units_per_px = ((b.max[0] - b.min[0]) / size.width as f64).max((b.max[1] - b.min[1]) / size.height as f64) * 1.08;
        self.view = Some(View {
            center: [(b.min[0] + b.max[0]) / 2.0, (b.min[1] + b.max[1]) / 2.0],
            units_per_px,
            size_px: [size.width, size.height],
        });
        window.request_redraw();
        self.gfx = Some(Gfx { window, surface, context, gl, renderer, mesh: gpu });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(view) = self.view.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                if let Some(g) = &self.gfx {
                    g.surface.resize(&g.context, NonZeroU32::new(size.width).unwrap(), NonZeroU32::new(size.height).unwrap());
                }
                view.size_px = [size.width, size.height];
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.dragging = state == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.dragging {
                    view.center[0] -= (position.x - self.cursor.x) * view.units_per_px;
                    view.center[1] += (position.y - self.cursor.y) * view.units_per_px;
                }
                self.cursor = position;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 50.0,
                };
                // Zoom about the cursor: the map point under it stays put.
                let anchor = App::map_point(view, self.cursor);
                view.units_per_px *= 1.15f64.powf(-steps);
                let moved = App::map_point(view, self.cursor);
                view.center = [view.center[0] + anchor[0] - moved[0], view.center[1] + anchor[1] - moved[1]];
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
    let mut args = std::env::args().skip(1);
    let screenshot = match (args.next().as_deref(), args.next()) {
        (Some("--screenshot"), Some(path)) => Some(path),
        (None, _) => None,
        _ => {
            eprintln!("usage: spike [--screenshot out.png]");
            std::process::exit(2);
        }
    };
    let (_, bounds) = data::counties();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App { screenshot, bounds, gfx: None, view: None, cursor: PhysicalPosition::new(0.0, 0.0), dragging: false };
    event_loop.run_app(&mut app).unwrap();
}
