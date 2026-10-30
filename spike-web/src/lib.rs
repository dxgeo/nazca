//! The browser half of nazca's `glow` spike: the same counties and the same
//! `gl` backend as `../examples/spike`, in a WebGL2 canvas. Drag to pan,
//! scroll to zoom about the cursor.
//!
//!   wasm-pack build --target web --release
//!   python3 -m http.server   # then open http://localhost:8000/
//!
//! A consumer crate, as `../../wasm-plan.org` says a browser binding must be:
//! nazca itself has no `wasm-bindgen` in it.

#[path = "../../examples/spike/data.rs"]
mod data;

use glow::HasContext;
use nazca::gl::{GpuMesh, Renderer, View};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{AddEventListenerOptions, HtmlCanvasElement, MouseEvent, WebGl2RenderingContext, WheelEvent};

struct State {
    gl: glow::Context,
    renderer: Renderer,
    mesh: GpuMesh,
    view: View,
    dpr: f64,
    drag: Option<(f64, f64)>,
}

impl State {
    fn draw(&self) {
        let [w, h] = self.view.size_px;
        let [r, g, b, a] = data::BACKGROUND;
        // SAFETY: plain state and clear calls on this canvas's context.
        unsafe {
            self.gl.viewport(0, 0, w as i32, h as i32);
            self.gl.clear_color(r, g, b, a);
            self.gl.clear(glow::COLOR_BUFFER_BIT);
        }
        self.renderer.draw(&self.gl, &self.mesh, &self.view, &data::PALETTE, data::OUTLINE);
    }

    /// The map point under a pointer position in CSS pixels.
    fn map_point(&self, css_x: f64, css_y: f64) -> [f64; 2] {
        let v = &self.view;
        [
            v.center[0] + (css_x * self.dpr - v.size_px[0] as f64 / 2.0) * v.units_per_px,
            v.center[1] - (css_y * self.dpr - v.size_px[1] as f64 / 2.0) * v.units_per_px,
        ]
    }
}

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let canvas: HtmlCanvasElement = document.get_element_by_id("map").ok_or("no #map canvas")?.dyn_into()?;

    // Draw at device resolution, not CSS pixels.
    let dpr = window.device_pixel_ratio();
    let (w, h) = ((canvas.client_width() as f64 * dpr) as u32, (canvas.client_height() as f64 * dpr) as u32);
    canvas.set_width(w);
    canvas.set_height(h);

    let context: WebGl2RenderingContext = canvas.get_context("webgl2")?.ok_or("WebGL2 unavailable")?.dyn_into()?;
    let gl = glow::Context::from_webgl2_context(context);
    let renderer = Renderer::new(&gl).map_err(|e| JsValue::from_str(&e))?;
    let (mesh, b) = data::counties();
    let gpu = renderer.upload(&gl, &mesh).map_err(|e| JsValue::from_str(&e))?;

    // SAFETY: a string query on the context just created.
    let version = unsafe { gl.get_parameter_string(glow::VERSION) };
    if let Some(stats) = document.get_element_by_id("stats") {
        stats.set_text_content(Some(&format!(
            "{version} · {} vertices · {} triangles · {} outline segments · {w}x{h} px",
            mesh.vertex_count(),
            mesh.fills.len() / 3,
            mesh.outlines.len() / 2
        )));
    }

    let units_per_px = ((b.max[0] - b.min[0]) / w as f64).max((b.max[1] - b.min[1]) / h as f64) * 1.08;
    let view = View { center: [(b.min[0] + b.max[0]) / 2.0, (b.min[1] + b.max[1]) / 2.0], units_per_px, size_px: [w, h] };
    let state = Rc::new(RefCell::new(State { gl, renderer, mesh: gpu, view, dpr, drag: None }));
    state.borrow().draw();

    let on = |target: &web_sys::EventTarget, name: &str, f: Box<dyn FnMut(web_sys::Event)>| -> Result<(), JsValue> {
        let closure = Closure::<dyn FnMut(web_sys::Event)>::new(f);
        let options = AddEventListenerOptions::new();
        // Not passive, so a wheel event can stop the page from scrolling.
        options.set_passive(false);
        target.add_event_listener_with_callback_and_add_event_listener_options(name, closure.as_ref().unchecked_ref(), &options)?;
        closure.forget(); // lives as long as the page
        Ok(())
    };

    let s = state.clone();
    on(&canvas, "mousedown", Box::new(move |e| {
        let e: MouseEvent = e.unchecked_into();
        s.borrow_mut().drag = Some((e.offset_x() as f64, e.offset_y() as f64));
    }))?;
    let s = state.clone();
    on(&window, "mouseup", Box::new(move |_| s.borrow_mut().drag = None))?;
    let s = state.clone();
    on(&canvas, "mousemove", Box::new(move |e| {
        let e: MouseEvent = e.unchecked_into();
        let mut st = s.borrow_mut();
        let Some((x0, y0)) = st.drag else { return };
        let (x, y) = (e.offset_x() as f64, e.offset_y() as f64);
        let step = st.view.units_per_px * st.dpr;
        st.view.center[0] -= (x - x0) * step;
        st.view.center[1] += (y - y0) * step;
        st.drag = Some((x, y));
        st.draw();
    }))?;
    let s = state.clone();
    on(&canvas, "wheel", Box::new(move |e| {
        let e: WheelEvent = e.unchecked_into();
        e.prevent_default();
        let mut st = s.borrow_mut();
        let (x, y) = (e.offset_x() as f64, e.offset_y() as f64);
        // Zoom about the cursor: the map point under it stays put.
        let anchor = st.map_point(x, y);
        st.view.units_per_px *= 1.15f64.powf(e.delta_y() / 100.0);
        let moved = st.map_point(x, y);
        st.view.center = [st.view.center[0] + anchor[0] - moved[0], st.view.center[1] + anchor[1] - moved[1]];
        st.draw();
    }))?;
    Ok(())
}
