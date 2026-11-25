//! The browser half of nazca's GL spike: the same counties and the same
//! camera, tile selector, tile cache and `gl` backend as `../examples/spike`,
//! in a WebGL2 canvas. Drag to pan, scroll to zoom about the cursor,
//! right-drag to rotate about the canvas's centre, click to pick a county
//! and toggle its selection, press H to toggle a hover state on the county
//! picked last, press F to show every other county, press M to cycle the
//! counties' markers, press D to cycle their outlines, press P to cycle the
//! patterns over them, press S to log the view as an SVG to the console,
//! press C to run the state check, press K to run the marker check, press X
//! to run the dash check and T the pattern check (their reports logged to
//! the console too), press B to time markers against dots (logged too), and
//! press R to cycle the raster overlay.
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

use data::Overlay;
use nazca::Camera;
use nazca::gl::{Context, Renderer};
use std::cell::RefCell;
use std::rc::Rc;
use tiles::{HOVERED, SELECTED, TiledMap};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{AddEventListenerOptions, HtmlCanvasElement, KeyboardEvent, MouseEvent, WebGl2RenderingContext, WebGlContextAttributes, WheelEvent};

struct State {
    gl: Context,
    renderer: Renderer,
    map: TiledMap,
    camera: Camera,
    dpr: f64,
    /// The pointer's last position while a button is held, and whether that
    /// button rotates rather than pans.
    drag: Option<(f64, f64, bool)>,
    /// Where a left press began, to tell a click from a drag.
    pressed_at: Option<(f64, f64)>,
    /// What the last click picked, shown with the stats.
    picked: String,
    stats: Option<web_sys::Element>,
    version: String,
}

impl State {
    fn draw(&mut self) {
        let [w, h] = self.camera.size_px;
        let s = self.map.draw(&self.gl, &self.renderer, &self.camera);
        if let Some(stats) = &self.stats {
            stats.set_text_content(Some(&format!(
                "{} · level {} · {} tiles drawn, {} just prepared · {} cached, {} KB · {w}x{h} px · overlay {:?} · markers {:?} · patterns {:?}{}",
                self.version,
                s.level,
                s.drawn,
                s.prepared,
                s.cached,
                s.cached_bytes / 1024,
                self.map.overlay(),
                self.map.markers(),
                self.map.patterns(),
                self.picked
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

    // A stencil buffer, which WebGL leaves out unless asked: each tile
    // claims its pixels in it.
    let attributes = WebGlContextAttributes::new();
    attributes.set_stencil(true);
    let context: WebGl2RenderingContext =
        canvas.get_context_with_context_options("webgl2", &attributes)?.ok_or("WebGL2 unavailable")?.dyn_into()?;
    let gl = Context::from_webgl2(context);
    let mut renderer = Renderer::new(&gl).map_err(|e| JsValue::from_str(&e))?;
    let version = gl.version();

    let map = TiledMap::new(&gl, &mut renderer, 16 << 20, Overlay::None);
    let mut camera = Camera::new([0.0, 0.0], 1.0, [w, h]);
    camera.fit_bounds(map.bounds(), 0.04 * w.min(h) as f64);
    let stats = document.get_element_by_id("stats");
    let state = Rc::new(RefCell::new(State { gl, renderer, map, camera, dpr, drag: None, pressed_at: None, picked: String::new(), stats, version }));
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
        let mut st = s.borrow_mut();
        let (x, y) = (e.offset_x() as f64, e.offset_y() as f64);
        st.drag = Some((x, y, e.button() == 2));
        st.pressed_at = (e.button() == 0).then_some((x, y));
    }))?;
    let s = state.clone();
    on(&canvas, "mouseup", Box::new(move |e| {
        let e: MouseEvent = e.unchecked_into();
        let mut st = s.borrow_mut();
        let (x, y) = (e.offset_x() as f64, e.offset_y() as f64);
        // A press and release in nearly one place is a click.
        let Some((x0, y0)) = st.pressed_at.take() else { return };
        if (x - x0).abs() + (y - y0).abs() >= 4.0 {
            return;
        }
        let st = &mut *st;
        let px = [x * st.dpr, y * st.dpr];
        st.picked = match st.map.pick(&st.gl, &st.renderer, &st.camera, px) {
            Some((name, id)) => {
                st.map.toggle(&st.gl, &st.renderer, id, SELECTED);
                format!(" · picked {name} (feature {id})")
            }
            None => " · picked nothing".to_string(),
        };
        st.draw();
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
    let s = state.clone();
    on(&window, "keydown", Box::new(move |e| {
        let e: KeyboardEvent = e.unchecked_into();
        let key = e.key().to_ascii_lowercase();
        let mut st = s.borrow_mut();
        let st = &mut *st;
        match key.as_str() {
            "r" => {
                let next = st.map.overlay().next();
                st.map.set_overlay(&st.gl, &st.renderer, next);
            }
            "h" => st.map.toggle_last(&st.gl, &st.renderer, HOVERED),
            "f" => {
                let on = !st.map.filtered();
                st.map.set_filter(&st.gl, &st.renderer, on);
            }
            "m" => {
                let next = st.map.markers().next();
                st.map.set_markers(&st.gl, &st.renderer, next);
            }
            "d" => {
                let next = st.map.outlines().next();
                st.map.set_outlines(&st.gl, &st.renderer, next);
            }
            "p" => {
                let next = st.map.patterns().next();
                st.map.set_patterns(&st.gl, &st.renderer, next);
            }
            "t" => {
                let report = match tiles::check_patterns(&st.map, &st.gl, &st.renderer, st.camera.size_px) {
                    Ok(report) => report,
                    Err(e) => format!("pattern check failed: {e}"),
                };
                web_sys::console::log_1(&report.clone().into());
                st.picked = format!(" · {}", report.lines().next().unwrap_or_default());
            }
            "s" => web_sys::console::log_1(&st.map.svg(&st.camera).into()),
            "x" => {
                let performance = web_sys::window().and_then(|w| w.performance());
                let clock = move || performance.as_ref().map_or(0.0, |p| p.now());
                let report = match tiles::check_dashes(&st.gl, &st.renderer, st.map.counties(), st.camera.size_px, &clock) {
                    Ok(report) => report,
                    Err(e) => format!("dash check failed: {e}"),
                };
                web_sys::console::log_1(&report.clone().into());
                st.picked = format!(" · {}", report.lines().next().unwrap_or_default());
            }
            "b" => {
                let performance = web_sys::window().and_then(|w| w.performance());
                let clock = move || performance.as_ref().map_or(0.0, |p| p.now());
                let report = match tiles::measure_markers(&st.map, &st.gl, &st.renderer, st.camera.size_px, &clock) {
                    Ok(report) => report,
                    Err(e) => format!("marker measurement failed: {e}"),
                };
                web_sys::console::log_1(&report.clone().into());
                st.picked = format!(" · {}", report.lines().next().unwrap_or_default());
            }
            "k" => {
                st.picked = match tiles::check_markers(&mut st.map, &st.gl, &st.renderer, &st.camera) {
                    Ok(report) => format!(" · {report}"),
                    Err(e) => format!(" · marker check failed: {e}"),
                };
            }
            "c" => {
                st.picked = match tiles::check_state(&st.gl, &st.renderer, st.camera.size_px) {
                    Ok(report) => format!(" · {report}"),
                    Err(e) => format!(" · state check failed: {e}"),
                };
            }
            _ => return,
        }
        st.draw();
    }))?;
    Ok(())
}
