//! Images a style draws with: markers at points and pattern fills
//! (`../plans/custom-symbols.org`). Standard library only, so the SVG writer
//! and a GPU backend agree on every symbol's id without a GPU in the room.
//!
//! A [`Symbol`] is an image the embedder hands in, RGBA8 with straight
//! alpha. Nazca reads no files: an embedder rasterizes its icons itself, at
//! the device's pixel ratio or more, and says so in the symbol's scale. A
//! symbol has no size of its own; the style gives it one.
//!
//! With the `symbols` feature, `Symbol::from_svg` rasterizes an icon's SVG
//! markup with `resvg` instead, and keeps `usvg`'s rewrite of it, which the
//! SVG writer writes in place of the pixels. Pixels and markup come from one
//! parsed tree, so they can't disagree. That rewrite is the only markup a
//! symbol carries (`../plans/custom-symbols.org` § OPEN QUESTIONS 8):
//!
//! - *It is one `<svg>` root of a known size*, with no prolog, no `<style>`,
//!   no scripts and no external references; an `<image>` it embeds is a data
//!   URL, and one naming a file or a URL is dropped, so nazca still reads no
//!   files. Text is dropped too: the feature has no fonts.
//! - *Its ids are prefixed by a hash of it*, so two icons that each define
//!   `id="a"` keep their own gradients and clips in one document.
//! - *A mask's `currentColor` is white*, so it masks the same in a viewer
//!   that reads masks by luminance as in one that reads `mask-type: alpha`.
//!   A colour symbol's is black, a browser's default.
//!
//! [`Symbols`] holds them by [`SymbolId`] and packs them into one atlas, in
//! shelves:
//!
//! - *Each image has 4 transparent pixels of padding, and each slot starts
//!   on a multiple of 4*, so a GPU's mipmaps up to level 2 (an aligned 4 x 4
//!   block a texel) never average two symbols together. A symbol drawn at a
//!   quarter of its pixels divided by its scale or larger is smooth; drawn
//!   smaller it can shimmer, and the embedder adds it again at a smaller
//!   size.
//! - *The atlas is premultiplied*, so filtering a colour edge beside a
//!   transparent pixel doesn't darken it. Each [`Symbol`] keeps its bytes as
//!   handed in, for repacking and for the SVG writer.
//! - *The atlas is at most [`ATLAS_MAX`] square*, the largest texture WebGL2
//!   promises. An image that can't fit is an error.
//! - *An id is never reused*, so a stale palette can't draw whatever took a
//!   removed symbol's place. Every [`Symbols::add`] takes one, a symbol
//!   added again at another size included, and after 65,536 adds to one
//!   [`Symbols`] an add is an error. Removing a symbol frees its space at the
//!   next repack, which an add that doesn't fit makes.

use std::collections::BTreeMap;

/// The largest atlas side, in pixels: WebGL2's guaranteed texture size.
pub const ATLAS_MAX: u32 = 2048;
/// Transparent pixels around each image in the atlas, and the alignment of
/// each slot. A GPU's mipmaps stop at level log2(`PADDING`).
pub const PADDING: u32 = 4;
/// The smallest atlas side, so a few small symbols don't repack at every add.
const ATLAS_MIN: u32 = 64;

/// A symbol's name in a [`Symbols`] and a [`crate::Palette`]. Never reused
/// within one [`Symbols`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolId(pub(crate) u16);

/// How a symbol's pixels are coloured. `#[non_exhaustive]`, so a kind added
/// later breaks no one: a `match` on it keeps a `_` arm.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Kind {
    /// Drawn as it is. The alpha of the class's `fill` multiplies its alpha,
    /// so fading a class's fill fades its markers too.
    #[default]
    Colour,
    /// Only its alpha is used, tinted by the class's `fill`: one hatch or one
    /// pin serves every colour a palette gives it.
    Mask,
}

/// An image a style draws: RGBA8, straight alpha, rows top first.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Symbol {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    /// Image pixels per drawn pixel, in the pixels widths are in: 2 for an
    /// icon rasterized at a device pixel ratio of 2.
    pub scale: f32,
    /// The point of the image that sits on the feature, as fractions of its
    /// width and height from its top left. The centre by default.
    pub anchor: [f32; 2],
    pub kind: Kind,
    markup: Option<Markup>,
}

/// `usvg`'s rewrite of the markup a symbol was rasterized from, and its
/// root's size, which the pixels fill.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Markup {
    pub(crate) svg: String,
    pub(crate) size: [f32; 2],
}

impl Symbol {
    /// A colour symbol of `width` x `height` pixels at scale 1, anchored at
    /// its centre. `None` unless `pixels` holds exactly `width * height`
    /// RGBA8 pixels and neither side is 0.
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Option<Symbol> {
        if width == 0 || height == 0 || pixels.len() as u64 != width as u64 * height as u64 * 4 {
            return None;
        }
        Some(Symbol { width, height, pixels, scale: 1.0, anchor: [0.5, 0.5], kind: Kind::Colour, markup: None })
    }

    /// A symbol rasterized from SVG `markup`, its longest side `size` drawn
    /// pixels at `scale` image pixels each, anchored at its centre. Its root
    /// fills the image, stretched if rounding to whole pixels changed its
    /// aspect. `kind` decides what the markup's `currentColor` is: white for
    /// a mask, black for a colour symbol. An error for a size or scale that
    /// isn't positive, markup `usvg` can't parse, or an image larger than
    /// [`ATLAS_MAX`].
    #[cfg(feature = "symbols")]
    pub fn from_svg(markup: &str, size: f32, scale: f32, kind: Kind) -> Result<Symbol, String> {
        use resvg::tiny_skia;
        if !(size.is_finite() && size > 0.0 && scale.is_finite() && scale > 0.0) {
            return Err(format!("an SVG symbol needs a positive size and scale, not {size} and {scale}"));
        }
        let mut options = usvg::Options::default();
        // Nazca reads no files: an image naming one draws nothing.
        options.image_href_resolver.resolve_string = Box::new(|_, _| None);
        if kind == Kind::Mask {
            options.style_sheet = Some("* { color: white }".into());
        }
        let tree = usvg::Tree::from_str(markup, &options).map_err(|e| format!("SVG markup nazca can't read: {e}"))?;
        let [w, h] = [tree.size().width(), tree.size().height()];
        let k = size * scale / w.max(h);
        let [pw, ph] = [w, h].map(|side| ((side * k).round() as u32).max(1));
        if pw > ATLAS_MAX || ph > ATLAS_MAX {
            return Err(format!("an SVG symbol of {pw} x {ph} pixels can't fit an atlas of {ATLAS_MAX} x {ATLAS_MAX}"));
        }
        let mut pixmap = tiny_skia::Pixmap::new(pw, ph).ok_or("an SVG symbol's pixmap couldn't be made")?;
        resvg::render(&tree, tiny_skia::Transform::from_scale(pw as f32 / w, ph as f32 / h), &mut pixmap.as_mut());
        let pixels = pixmap.pixels().iter().flat_map(|p| {
            let c = p.demultiply();
            [c.red(), c.green(), c.blue(), c.alpha()]
        });
        let mut write = usvg::WriteOptions { indent: usvg::Indent::None, attributes_indent: usvg::Indent::None, ..Default::default() };
        write.id_prefix = Some(format!("s{:016x}-", fnv1a(tree.to_string(&write).as_bytes())));
        let svg = tree.to_string(&write).trim_end().to_string();
        Ok(Symbol { width: pw, height: ph, pixels: pixels.collect(), scale, anchor: [0.5, 0.5], kind, markup: Some(Markup { svg, size: [w, h] }) })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// The pixels as handed in.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// The largest size it is drawn at without blurring, in drawn pixels:
    /// its longest side divided by its scale.
    pub fn sharp_px(&self) -> f32 {
        self.width.max(self.height) as f32 / self.scale
    }

    /// The markup it was rasterized from, for the SVG writer: only one built
    /// by `Symbol::from_svg` has it.
    pub(crate) fn markup(&self) -> Option<&Markup> {
        self.markup.as_ref()
    }
}

/// FNV-1a, 64 bits: a hash fixed across Rust versions, as `std`'s is not, so
/// a symbol's ids are the same in every build.
#[cfg(feature = "symbols")]
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

/// Where an image lies in the atlas, in atlas pixels, padding not included.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Symbols by id, packed into one premultiplied RGBA8 atlas.
#[derive(Debug, Clone)]
pub struct Symbols {
    entries: BTreeMap<u16, (Symbol, Rect)>,
    /// The next id to give, up to 65,536 when none are left.
    next: u32,
    side: u32,
    atlas: Vec<u8>,
    shelves: Vec<Shelf>,
}

/// A row of slots of at most `height`, filled left to right.
#[derive(Debug, Clone, Copy)]
struct Shelf {
    y: u32,
    height: u32,
    used: u32,
}

impl Default for Symbols {
    fn default() -> Symbols {
        Symbols { entries: BTreeMap::new(), next: 0, side: ATLAS_MIN, atlas: vec![0; (ATLAS_MIN * ATLAS_MIN * 4) as usize], shelves: Vec::new() }
    }
}

impl Symbols {
    pub fn new() -> Symbols {
        Symbols::default()
    }

    /// Adds `symbol` and returns its id, repacking the atlas (and growing
    /// it, up to [`ATLAS_MAX`]) if there is no room. An error, with nothing
    /// changed, for a scale that isn't positive, an anchor that isn't finite,
    /// an image that can't fit the largest atlas with the symbols already
    /// held, or no ids left.
    pub fn add(&mut self, symbol: Symbol) -> Result<SymbolId, String> {
        let (w, h) = (symbol.width, symbol.height);
        if !(symbol.scale.is_finite() && symbol.scale > 0.0) {
            return Err(format!("a symbol of {w} x {h} pixels has scale {}, not a positive number", symbol.scale));
        }
        if !symbol.anchor.iter().all(|a| a.is_finite()) {
            return Err(format!("a symbol of {w} x {h} pixels has anchor {:?}, not finite", symbol.anchor));
        }
        let Ok(id) = u16::try_from(self.next) else {
            return Err("no symbol ids left: a Symbols gives 65,536, and never one twice".into());
        };
        let (sw, sh) = slot(w, h);
        if sw > ATLAS_MAX || sh > ATLAS_MAX {
            return Err(format!("a symbol of {w} x {h} pixels can't fit an atlas of {ATLAS_MAX} x {ATLAS_MAX}"));
        }
        let rect = match place(&mut self.shelves, self.side, w, h) {
            Some(rect) => rect,
            None => {
                let mut all: Vec<(u16, u32, u32)> = self.entries.iter().map(|(&id, (s, _))| (id, s.width, s.height)).collect();
                all.push((id, w, h));
                let Some((side, shelves, rects)) = pack(&all) else {
                    return Err(format!("a symbol of {w} x {h} pixels can't fit an atlas of {ATLAS_MAX} x {ATLAS_MAX} with the {} symbols held", self.entries.len()));
                };
                self.side = side;
                self.shelves = shelves;
                self.atlas = vec![0; (side * side * 4) as usize];
                for (&id, entry) in self.entries.iter_mut() {
                    entry.1 = rects[&id];
                    blit(&mut self.atlas, side, &entry.0, entry.1);
                }
                rects[&id]
            }
        };
        blit(&mut self.atlas, self.side, &symbol, rect);
        self.entries.insert(id, (symbol, rect));
        self.next += 1;
        Ok(SymbolId(id))
    }

    /// Removes a symbol, returning it. Its space is cleared and freed at the
    /// next repack; its id is not given again.
    pub fn remove(&mut self, id: SymbolId) -> Option<Symbol> {
        let (symbol, rect) = self.entries.remove(&id.0)?;
        let blank = Symbol { pixels: vec![0; symbol.pixels.len()], ..symbol.clone() };
        blit(&mut self.atlas, self.side, &blank, rect);
        Some(symbol)
    }

    pub fn get(&self, id: SymbolId) -> Option<&Symbol> {
        self.entries.get(&id.0).map(|e| &e.0)
    }

    /// Where `id`'s image lies in the atlas.
    pub fn rect(&self, id: SymbolId) -> Option<Rect> {
        self.entries.get(&id.0).map(|e| e.1)
    }

    /// The symbols held, in id order.
    pub fn iter(&self) -> impl Iterator<Item = (SymbolId, &Symbol, Rect)> {
        self.entries.iter().map(|(&id, (s, r))| (SymbolId(id), s, *r))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The atlas's side in pixels: a power of two from 64 to [`ATLAS_MAX`].
    pub fn atlas_side(&self) -> u32 {
        self.side
    }

    /// The atlas, `atlas_side()` squared RGBA8 pixels, premultiplied, rows
    /// top first.
    pub fn atlas(&self) -> &[u8] {
        &self.atlas
    }
}

/// A slot's size: the image, its padding on every side, rounded up to the
/// alignment.
fn slot(w: u32, h: u32) -> (u32, u32) {
    let round = |n: u32| (n + 2 * PADDING).div_ceil(PADDING) * PADDING;
    (round(w), round(h))
}

/// Places a `w` x `h` image's slot on the shelf it wastes least height on,
/// or on a new shelf, returning the image's rectangle inside it.
fn place(shelves: &mut Vec<Shelf>, side: u32, w: u32, h: u32) -> Option<Rect> {
    let (sw, sh) = slot(w, h);
    let fits = shelves.iter_mut().filter(|s| s.height >= sh && s.used + sw <= side).min_by_key(|s| s.height);
    let shelf = match fits {
        Some(shelf) => shelf,
        None => {
            let y = shelves.last().map_or(0, |s| s.y + s.height);
            if y + sh > side || sw > side {
                return None;
            }
            shelves.push(Shelf { y, height: sh, used: 0 });
            shelves.last_mut()?
        }
    };
    let x = shelf.used;
    shelf.used += sw;
    Some(Rect { x: x + PADDING, y: shelf.y + PADDING, width: w, height: h })
}

/// Packs `(id, width, height)` images tallest first into the smallest square
/// atlas they fit, or `None` past [`ATLAS_MAX`].
#[allow(clippy::type_complexity)]
fn pack(images: &[(u16, u32, u32)]) -> Option<(u32, Vec<Shelf>, BTreeMap<u16, Rect>)> {
    let mut order = images.to_vec();
    order.sort_by_key(|&(id, w, h)| (std::cmp::Reverse(slot(w, h).1), id));
    let mut side = ATLAS_MIN;
    'sides: while side <= ATLAS_MAX {
        let mut shelves = Vec::new();
        let mut rects = BTreeMap::new();
        for &(id, w, h) in &order {
            let Some(r) = place(&mut shelves, side, w, h) else {
                side *= 2;
                continue 'sides;
            };
            rects.insert(id, r);
        }
        return Some((side, shelves, rects));
    }
    None
}

/// Writes `symbol`'s pixels into `rect` of the atlas, premultiplied.
fn blit(atlas: &mut [u8], side: u32, symbol: &Symbol, rect: Rect) {
    let premultiply = |c: u8, a: u8| ((c as u32 * a as u32 + 127) / 255) as u8;
    for row in 0..symbol.height {
        let src = &symbol.pixels[(row * symbol.width * 4) as usize..][..(symbol.width * 4) as usize];
        let start = (((rect.y + row) * side + rect.x) * 4) as usize;
        let dst = &mut atlas[start..][..src.len()];
        for (d, s) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
            let a = s[3];
            d.copy_from_slice(&[premultiply(s[0], a), premultiply(s[1], a), premultiply(s[2], a), a]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(w: u32, h: u32, rgba: [u8; 4]) -> Symbol {
        Symbol::new(w, h, rgba.repeat((w * h) as usize)).unwrap()
    }

    fn slot_of(r: Rect) -> (u32, u32, u32, u32) {
        let (sw, sh) = slot(r.width, r.height);
        (r.x - PADDING, r.y - PADDING, sw, sh)
    }

    fn check_layout(s: &Symbols) {
        let slots: Vec<_> = s.iter().map(|(_, _, r)| slot_of(r)).collect();
        for &(x, y, w, h) in &slots {
            assert_eq!((x % PADDING, y % PADDING, w % PADDING, h % PADDING), (0, 0, 0, 0));
            assert!(x + w <= s.atlas_side() && y + h <= s.atlas_side());
        }
        for (i, a) in slots.iter().enumerate() {
            for b in &slots[i + 1..] {
                let apart = a.0 + a.2 <= b.0 || b.0 + b.2 <= a.0 || a.1 + a.3 <= b.1 || b.1 + b.3 <= a.1;
                assert!(apart, "slots {a:?} and {b:?} overlap");
            }
        }
    }

    #[test]
    fn symbols_must_be_whole_images() {
        assert!(Symbol::new(2, 2, vec![0; 16]).is_some());
        assert!(Symbol::new(2, 2, vec![0; 15]).is_none());
        assert!(Symbol::new(0, 2, vec![]).is_none());
        let mut s = Symbols::new();
        let mut bad = image(2, 2, [0; 4]);
        bad.scale = 0.0;
        assert!(s.add(bad).is_err());
        let mut bad = image(2, 2, [0; 4]);
        bad.anchor = [f32::NAN, 0.5];
        assert!(s.add(bad).is_err());
        assert_eq!(s.add(image(2, 2, [0; 4])), Ok(SymbolId(0)));
    }

    #[test]
    fn ids_stay_stable_across_a_repack() {
        let mut s = Symbols::new();
        let mut ids = Vec::new();
        // Enough 20-pixel images (28-pixel slots) to outgrow 64 and 128.
        for k in 0..30u8 {
            ids.push(s.add(image(20, 20 + (k % 3) as u32, [k, 0, 0, 255])).unwrap());
        }
        assert!(s.atlas_side() >= 256, "{}", s.atlas_side());
        check_layout(&s);
        for (k, &id) in ids.iter().enumerate() {
            assert_eq!(id, SymbolId(k as u16));
            let r = s.rect(id).unwrap();
            assert_eq!((r.width, r.height), (20, 20 + (k % 3) as u32));
            // Its pixels moved with it.
            let at = ((r.y * s.atlas_side() + r.x) * 4) as usize;
            assert_eq!(s.atlas()[at], k as u8);
        }
    }

    #[test]
    fn padding_is_transparent() {
        let mut s = Symbols::new();
        let a = s.add(image(5, 3, [255; 4])).unwrap();
        let b = s.add(image(7, 7, [255; 4])).unwrap();
        check_layout(&s);
        let side = s.atlas_side();
        let opaque = |x: u32, y: u32| s.atlas()[((y * side + x) * 4 + 3) as usize] != 0;
        for id in [a, b] {
            let r = s.rect(id).unwrap();
            let (sx, sy, sw, sh) = slot_of(r);
            for y in sy..sy + sh {
                for x in sx..sx + sw {
                    let inside = x >= r.x && x < r.x + r.width && y >= r.y && y < r.y + r.height;
                    assert_eq!(opaque(x, y), inside, "({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn the_atlas_is_premultiplied_and_the_symbol_is_not() {
        let mut s = Symbols::new();
        let id = s.add(image(1, 1, [255, 100, 0, 128])).unwrap();
        let r = s.rect(id).unwrap();
        let at = ((r.y * s.atlas_side() + r.x) * 4) as usize;
        assert_eq!(&s.atlas()[at..at + 4], &[128, 50, 0, 128]);
        assert_eq!(s.get(id).unwrap().pixels(), &[255, 100, 0, 128]);
    }

    #[test]
    fn an_image_too_large_is_an_error() {
        let mut s = Symbols::new();
        let err = s.add(image(ATLAS_MAX, 1, [0; 4])).unwrap_err();
        assert!(err.contains("2048 x 1"), "{err}");
        // Room for one 1,000-pixel image a shelf, two shelves high, but not
        // a fifth.
        for _ in 0..4 {
            s.add(image(1000, 1000, [0; 4])).unwrap();
        }
        assert_eq!(s.atlas_side(), ATLAS_MAX);
        assert!(s.add(image(1000, 1000, [0; 4])).is_err());
        assert_eq!(s.len(), 4);
        check_layout(&s);
    }

    #[test]
    fn removing_frees_space_at_the_next_repack_but_never_an_id() {
        let mut s = Symbols::new();
        let first = s.add(image(1000, 1000, [0, 0, 0, 255])).unwrap();
        for _ in 0..3 {
            s.add(image(1000, 1000, [0; 4])).unwrap();
        }
        let r = s.rect(first).unwrap();
        assert!(s.remove(first).is_some());
        assert_eq!(s.atlas()[((r.y * s.atlas_side() + r.x) * 4 + 3) as usize], 0);
        assert!(s.remove(first).is_none());
        let again = s.add(image(1000, 1000, [0; 4])).unwrap();
        assert_eq!(again, SymbolId(4));
        assert_eq!(s.get(first), None);
        check_layout(&s);
    }

    #[cfg(feature = "symbols")]
    #[test]
    fn an_svg_symbol_is_its_pixels_and_usvgs_rewrite() {
        let red = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 12"><rect width="24" height="12" fill="red"/></svg>"#;
        // Longest side 8 drawn pixels at 2 image pixels each.
        let s = Symbol::from_svg(red, 8.0, 2.0, Kind::Colour).unwrap();
        assert_eq!((s.width(), s.height(), s.scale, s.anchor), (16, 8, 2.0, [0.5, 0.5]));
        assert!(s.pixels().chunks_exact(4).all(|p| p == [255, 0, 0, 255]));
        let markup = s.markup().unwrap();
        assert_eq!(markup.size, [24.0, 12.0]);
        assert!(markup.svg.starts_with(r#"<svg width="24" height="12" xmlns="http://www.w3.org/2000/svg">"#), "{}", markup.svg);
        assert!(Symbol::new(1, 1, vec![0; 4]).unwrap().markup().is_none());
    }

    #[cfg(feature = "symbols")]
    #[test]
    fn an_svg_symbols_markup_keeps_only_what_it_draws() {
        let icon = r#"<?xml version="1.0"?><!DOCTYPE svg><svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><style>.c{opacity:0.5}</style><path class="c" fill="currentColor" d="M0 0H10V10Z"/><image href="icon.png" width="5" height="5"/><text x="1" y="9">hi</text><script>alert(1)</script></svg>"#;
        let colour = Symbol::from_svg(icon, 10.0, 1.0, Kind::Colour).unwrap();
        let svg = &colour.markup().unwrap().svg;
        for gone in ["<?xml", "DOCTYPE", "<style", "<image", "<text", "<script", "icon.png"] {
            assert!(!svg.contains(gone), "{gone} in {svg}");
        }
        // The style sheet applied, and currentColor black for a colour symbol,
        // white for a mask, in the markup and the pixels alike.
        assert!(svg.contains(r##"<g opacity="0.5"><path fill="#000000""##), "{svg}");
        let mask = Symbol::from_svg(icon, 10.0, 1.0, Kind::Mask).unwrap();
        assert!(mask.markup().unwrap().svg.contains(r##"fill="#ffffff""##), "{:?}", mask.markup());
        // Top right, inside the triangle.
        assert_eq!(&colour.pixels()[9 * 4 + 40..][..4], &[0, 0, 0, 128]);
        assert_eq!(&mask.pixels()[9 * 4 + 40..][..4], &[255, 255, 255, 128]);
    }

    #[cfg(feature = "symbols")]
    #[test]
    fn an_svg_symbol_can_be_refused() {
        let square = r#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>"#;
        for (size, scale) in [(0.0, 1.0), (10.0, -1.0), (f32::NAN, 1.0), (10.0, f32::INFINITY)] {
            assert!(Symbol::from_svg(square, size, scale, Kind::Colour).is_err(), "{size} {scale}");
        }
        assert!(Symbol::from_svg("<svg", 10.0, 1.0, Kind::Colour).is_err());
        let err = Symbol::from_svg(square, 1100.0, 2.0, Kind::Colour).unwrap_err();
        assert!(err.contains("2200 x 2200"), "{err}");
        // Rounding never leaves a side empty.
        let thin = r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="1"/>"#;
        let s = Symbol::from_svg(thin, 10.0, 1.0, Kind::Colour).unwrap();
        assert_eq!((s.width(), s.height()), (10, 1));
    }

    #[test]
    fn ids_run_out_after_65536_adds() {
        let mut s = Symbols::new();
        for k in 0..=u16::MAX {
            let id = s.add(image(1, 1, [0; 4])).unwrap();
            assert_eq!(id, SymbolId(k));
            s.remove(id);
        }
        let err = s.add(image(1, 1, [0; 4])).unwrap_err();
        assert!(err.contains("65,536"), "{err}");
    }
}
