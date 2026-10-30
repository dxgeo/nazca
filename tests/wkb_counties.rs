//! Reads real county geometry (DuckDB's `ST_AsWKB` output, from DuckMap's
//! catalog) and checks every count and bound against DuckDB's own answer for
//! the same geometry. See `fixtures/make_fixtures.sh` for how the TSVs are
//! made and what each column is.

use nazca::{Sink, wkb};

/// Folds a geometry into the same numbers DuckDB reports.
#[derive(Default)]
struct Tally {
    polygons: usize,
    rings: usize,
    points: usize,
    bounds: Option<[f64; 4]>,
}

impl Tally {
    fn extend(&mut self, [x, y]: [f64; 2]) {
        let b = self.bounds.get_or_insert([x, y, x, y]);
        *b = [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)];
    }
}

impl Sink for Tally {
    fn point(&mut self, xy: [f64; 2]) {
        self.points += 1;
        self.extend(xy);
    }
    fn points(&mut self, xy: &[[f64; 2]]) {
        self.points += xy.len();
        for &p in xy {
            self.extend(p);
        }
    }
    fn begin_polygon(&mut self) { self.polygons += 1; }
    fn begin_ring(&mut self) { self.rings += 1; }
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

/// Checks every row of a fixture; returns how many rows it checked.
fn check(tsv: &str) -> usize {
    let mut lines = tsv.lines();
    assert_eq!(
        lines.next().unwrap(),
        "geoid\tname\ttype\tparts\trings\tpoints\txmin\tymin\txmax\tymax\twkb_hex"
    );
    let mut rows = 0;
    for line in lines {
        let f: Vec<&str> = line.split('\t').collect();
        let (geoid, name) = (f[0], f[1]);
        let num = |i: usize| f[i].parse::<usize>().unwrap();
        let float = |i: usize| f[i].parse::<f64>().unwrap();

        let mut t = Tally::default();
        wkb::read(&hex(f[10]), &mut t).unwrap_or_else(|e| panic!("{geoid} {name}: {e}"));

        assert_eq!(t.polygons, num(3), "{geoid} {name}: parts");
        assert_eq!(t.rings, num(4), "{geoid} {name}: rings");
        assert_eq!(t.points, num(5), "{geoid} {name}: points");
        // DuckDB prints the shortest decimal that round-trips, so these are
        // exact comparisons, not tolerances.
        let expected = [float(6), float(7), float(8), float(9)];
        assert_eq!(t.bounds, Some(expected), "{geoid} {name}: bounds");
        rows += 1;
    }
    rows
}

#[test]
fn counties_sample_matches_duckdb() {
    let rows = check(include_str!("fixtures/counties_sample.tsv"));
    assert_eq!(rows, 28);
}

/// All 3,222 US counties. Not checked in; generate it with
/// `tests/fixtures/make_fixtures.sh`, then run `cargo test -- --ignored`.
#[test]
#[ignore = "needs tests/fixtures/local/us_counties.tsv"]
fn us_counties_match_duckdb() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/local/us_counties.tsv");
    let tsv = std::fs::read_to_string(path).expect("run tests/fixtures/make_fixtures.sh first");
    assert_eq!(check(&tsv), 3222);
}
