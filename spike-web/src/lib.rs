//! The browser half of nazca's `glow` spike: the same counties and the same
//! camera, tile selector, tile cache and `gl` backend as `../examples/spike`,
//! in a WebGL2 canvas. Drag to pan, scroll to zoom about the cursor,
//! right-drag to rotate about the canvas's centre.
//!
//!   wasm-pack build --target web --release
//!   python3 -m http.server   # then open http://localhost:8000/
//!
//! A consumer crate, as `../../wasm-plan.org` says a browser binding must be:
//! nazca itself has no `wasm-bindgen` in it.

#[path = "../../examples/spike/data.rs"]
mod data;
#[path = "../../examples/spike/tiles.rs"]
mod tiles;

use glow::HasContext;
use nazca::Camera;
use nazca::gl::Renderer;
use std::cell::RefCell;
use std::rc::Rc;
use tiles::TiledMap;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{AddEventListenerOptions, HtmlCanvasElement, MouseEvent, WebGl2RenderingContext, WheelEvent};

struct State {
    gl: glow::Context,
    renderer: Renderer,
    map: TiledMap,
    camera: Camera,
    dpr: f64,
    /// The pointer's last position while a button is held, and whether that
    /// button rotates rather than pans.
    drag: Option<(f64, f64, bool)>,
    stats: Option<web_sys::Element>,
    version: String,
}

impl State {
    fn draw(&mut self) {
        let [w, h] = self.camera.size_px;
        let [r, g, b, a] = data::BACKGROUND;
        // SAFETY: plain state and clear calls on this canvas's context.
        unsafe {
            self.gl.viewport(0, 0, w as i32, h as i32);
            self.gl.clear_color(r, g, b, a);
            self.gl.clear(glow::COLOR_BUFFER_BIT);
        }
        let s = self.map.draw(&self.gl, &self.renderer, &self.camera);
        if let Some(stats) = &self.stats {
            stats.set_text_content(Some(&format!(
                "{} · level {} · {} tiles drawn, {} just prepared · {} cached, {} KB · {w}x{h} px",
                self.version,
                s.level,
                s.drawn,
                s.prepared,
                s.cached,
                s.cached_bytes / 1024
            )));
        }
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
    // SAFETY: a string query on the context just created.
    let version = unsafe { gl.get_parameter_string(glow::VERSION) };

    let map = TiledMap::new(16 << 20);
    let mut camera = Camera::new([0.0, 0.0], 1.0, [w, h]);
    camera.fit_bounds(map.bounds(), 0.04 * w.min(h) as f64);
    let stats = document.get_element_by_id("stats");
    let state = Rc::new(RefCell::new(State { gl, renderer, map, camera, dpr, drag: None, stats, version }));
    state.borrow_mut().draw();

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
        s.borrow_mut().drag = Some((e.offset_x() as f64, e.offset_y() as f64, e.button() == 2));
    }))?;
    // The right button rotates, so it must not open the context menu.
    on(&canvas, "contextmenu", Box::new(|e| e.prevent_default()))?;
    let s = state.clone();
    on(&window, "mouseup", Box::new(move |_| s.borrow_mut().drag = None))?;
    let s = state.clone();
    on(&canvas, "mousemove", Box::new(move |e| {
        let e: MouseEvent = e.unchecked_into();
        let mut st = s.borrow_mut();
        let Some((x0, y0, rotate)) = st.drag else { return };
        let (x, y) = (e.offset_x() as f64, e.offset_y() as f64);
        let dpr = st.dpr;
        if rotate {
            // A full turn per canvas width.
            let [w, h] = st.camera.size_px.map(|v| v as f64);
            st.camera.rotate_about([w / 2.0, h / 2.0], (x - x0) * dpr / w * std::f64::consts::TAU);
        } else {
            st.camera.pan_px([(x - x0) * dpr, (y - y0) * dpr]);
        }
        st.drag = Some((x, y, rotate));
        st.draw();
    }))?;
    let s = state.clone();
    on(&canvas, "wheel", Box::new(move |e| {
        let e: WheelEvent = e.unchecked_into();
        e.prevent_default();
        let mut st = s.borrow_mut();
        let (x, y) = (e.offset_x() as f64, e.offset_y() as f64);
        let dpr = st.dpr;
        st.camera.zoom_about([x * dpr, y * dpr], 1.15f64.powf(-e.delta_y() / 100.0));
        st.draw();
    }))?;
    Ok(())
}
