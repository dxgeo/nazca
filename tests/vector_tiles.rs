//! The vector tile decoders against independent answers.
//!
//! - MLT: every v1 conformance fixture of the MLT specification decodes to
//!   the geometry and ids its expected GeoJSON gives
//!   (`fixtures/fetch_mlt_fixtures.sh`; ignored until fetched).
//! - MVT and MLT: one Massachusetts basemap tile, cut by the same tiler from
//!   the same counties in both encodings (`fixtures/basemap/`, z6/19/23 from
//!   DuckMap's PMTiles basemaps, census-derived), decodes to the same
//!   geometry, feature for feature.

use nazca::grid::Rect;
use nazca::{Sink, mlt, mvt};
use serde_json::{Value, json};

/// Rebuilds a feature's geometry as GeoJSON coordinates in tile units.
#[derive(Default)]
struct Collect {
    extent: f64,
    points: Vec<Value>,
    lines: Vec<Value>,
    polygons: Vec<Vec<Value>>,
    current: Vec<Value>,
}

impl Collect {
    fn new(extent: u32) -> Collect {
        Collect { extent: extent as f64, ..Collect::default() }
    }

    /// Map units back to tile units (the bounds are 0..extent, Y flipped).
    fn xy(&self, p: [f64; 2]) -> Value {
        json!([p[0] as i64, (self.extent - p[1]) as i64])
    }

    fn geometry(self, kind: &str) -> Value {
        let first = |v: Vec<Value>| v.into_iter().next().unwrap_or(Value::Null);
        let polygons = || self.polygons.iter().map(|p| Value::Array(p.clone())).collect::<Vec<_>>();
        let coordinates = match kind {
            "Point" => first(self.points.clone()),
            "MultiPoint" => Value::Array(self.points.clone()),
            "LineString" => first(self.lines.clone()),
            "MultiLineString" => Value::Array(self.lines.clone()),
            "Polygon" => first(polygons()),
            _ => Value::Array(polygons()),
        };
        json!({ "type": kind, "coordinates": coordinates })
    }
}

impl Sink for Collect {
    fn point(&mut self, xy: [f64; 2]) {
        let p = self.xy(xy);
        self.points.push(p);
    }
    fn points(&mut self, xy: &[[f64; 2]]) {
        for &p in xy {
            let p = self.xy(p);
            self.current.push(p);
        }
    }
    fn begin_line(&mut self) {
        self.current.clear();
    }
    fn end_line(&mut self) {
        self.lines.push(Value::Array(std::mem::take(&mut self.current)));
    }
    fn begin_polygon(&mut self) {
        self.polygons.push(Vec::new());
    }
    fn begin_ring(&mut self) {
        self.current.clear();
    }
    fn end_ring(&mut self) {
        let ring = Value::Array(std::mem::take(&mut self.current));
        self.polygons.last_mut().unwrap().push(ring);
    }
}

fn mlt_type_name(t: mlt::GeomType) -> &'static str {
    match t {
        mlt::GeomType::Point => "Point",
        mlt::GeomType::LineString => "LineString",
        mlt::GeomType::Polygon => "Polygon",
        mlt::GeomType::MultiPoint => "MultiPoint",
        mlt::GeomType::MultiLineString => "MultiLineString",
        mlt::GeomType::MultiPolygon => "MultiPolygon",
    }
}

/// One decoded feature: layer name, extent, id, GeoJSON geometry.
type Decoded = (String, u32, Option<u64>, Value);

/// Every feature of an MLT tile.
fn decode_mlt(tile: &[u8]) -> Result<Vec<Decoded>, mlt::Error> {
    let mut out = Vec::new();
    for layer in mlt::layers(tile) {
        let layer = layer?;
        let features = layer.decode()?;
        let e = layer.extent();
        let bounds = Rect { min: [0.0, 0.0], max: [e as f64, e as f64] };
        for i in 0..features.len() {
            let mut c = Collect::new(e);
            features.read(i, bounds, &mut c)?;
            let kind = mlt_type_name(features.geom_type(i).unwrap());
            out.push((layer.name().to_string(), e, features.id(i), c.geometry(kind)));
        }
    }
    Ok(out)
}

#[test]
#[ignore = "needs tests/fixtures/local/mlt (fixtures/fetch_mlt_fixtures.sh)"]
fn mlt_conformance() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/local/mlt/synthetic");
    let mut names: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "mlt")).collect();
    names.sort();
    let mut failures = Vec::new();
    for path in &names {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let tile = std::fs::read(path).unwrap();
        let want: Value = serde_json::from_slice(&std::fs::read(path.with_extension("json")).unwrap()).unwrap();
        let got = match decode_mlt(&tile) {
            Ok(g) => g,
            Err(e) => {
                failures.push(format!("{name}: {e}"));
                continue;
            }
        };
        let want = want["features"].as_array().unwrap();
        if want.len() != got.len() {
            failures.push(format!("{name}: {} features, want {}", got.len(), want.len()));
            continue;
        }
        for (k, (w, (layer, extent, id, geometry))) in want.iter().zip(&got).enumerate() {
            let props = &w["properties"];
            if props["_layer"] != json!(layer) || props["_extent"] != json!(extent) {
                failures.push(format!("{name} #{k}: layer {layer}/{extent}, want {}/{}", props["_layer"], props["_extent"]));
            }
            if w.get("id").and_then(Value::as_u64) != *id {
                failures.push(format!("{name} #{k}: id {id:?}, want {:?}", w.get("id")));
            }
            if &w["geometry"] != geometry {
                failures.push(format!("{name} #{k}: geometry {geometry}, want {}", w["geometry"]));
            }
        }
    }
    assert!(failures.is_empty(), "{} of {} fixtures:\n{}", failures.len(), names.len(), failures.join("\n"));
    assert!(names.len() > 300, "only {} fixtures", names.len());
}

/// Every feature of an MVT tile as (layer, extent, id, polygons), with each
/// geometry as the list of polygons (or lines, or points) it holds, since MVT
/// does not tell `Polygon` from `MultiPolygon`.
fn decode_mvt(tile: &[u8]) -> Vec<Decoded> {
    let mut out = Vec::new();
    for layer in mvt::layers(tile) {
        let layer = layer.unwrap();
        let e = layer.extent();
        let bounds = Rect { min: [0.0, 0.0], max: [e as f64, e as f64] };
        for f in layer.features() {
            let f = f.unwrap();
            let mut c = Collect::new(e);
            f.read(bounds, &mut c).unwrap();
            out.push((layer.name().to_string(), e, f.id(), c.geometry("MultiPolygon")));
        }
    }
    out
}

const MA_MVT: &[u8] = include_bytes!("fixtures/basemap/ma_6_19_23.mvt");
const MA_MLT: &[u8] = include_bytes!("fixtures/basemap/ma_6_19_23.mlt");

#[test]
fn the_same_tile_in_mvt_and_mlt_decodes_alike() {
    assert_same_geometry(MA_MVT, MA_MLT);
}

/// Every pair `fixtures/make_basemap_fixtures.sh` cut, z0 to z6.
#[test]
#[ignore = "needs tests/fixtures/local/basemap (fixtures/make_basemap_fixtures.sh)"]
fn every_local_basemap_tile_decodes_alike() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/local/basemap");
    let mut n = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|x| x == "mvt") {
            assert_same_geometry(&std::fs::read(&path).unwrap(), &std::fs::read(path.with_extension("mlt")).unwrap());
            n += 1;
        }
    }
    assert!(n >= 8, "{n} pairs");
}

fn assert_same_geometry(mvt_tile: &[u8], mlt_tile: &[u8]) {
    let from_mvt = decode_mvt(mvt_tile);
    let from_mlt: Vec<_> = decode_mlt(mlt_tile)
        .unwrap()
        .into_iter()
        .map(|(layer, e, id, g)| {
            // As MultiPolygon, to compare with MVT's one polygon type.
            let g = if g["type"] == "Polygon" { json!({ "type": "MultiPolygon", "coordinates": [g["coordinates"]] }) } else { g };
            (layer, e, id, g)
        })
        .collect();
    assert!(from_mvt.len() > 20, "{} features", from_mvt.len());
    assert_eq!(from_mvt.len(), from_mlt.len());
    // Geometry only, as sets: the tiler (freestiler 0.3.0) orders and numbers
    // features differently per format. Its MLT writer also stores the 64-bit
    // id deltas without the zigzag encoding MLT v1 requires, so a conforming
    // decoder reads those ids as wrapped negatives.
    let key = |f: &Decoded| format!("{} {} {}", f.0, f.1, f.3);
    let mut a: Vec<String> = from_mvt.iter().map(key).collect();
    let mut b: Vec<String> = from_mlt.iter().map(key).collect();
    a.sort();
    b.sort();
    for (k, (a, b)) in a.iter().zip(&b).enumerate() {
        assert_eq!(a, b, "feature {k} in sorted order");
    }
}
