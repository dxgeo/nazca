#!/bin/sh
# Regenerates nazca's WKB test fixtures from DuckMap's DuckLake catalog.
#
#   tests/fixtures/make_fixtures.sh [path/to/duckmap.ducklake]
#
# The default path assumes DuckMap is checked out beside geocore.
#
# Needs the `duckdb` CLI (1.5.x) with the spatial extension. DuckMap's tile
# server must be stopped: a second process cannot ATTACH the catalog while it
# holds it.
#
# Writes one TSV per fixture, one row per county. Every column but the last
# is DuckDB's own answer for that geometry, so the WKB reader is checked
# against an independent implementation rather than against itself:
#
#   geoid name type parts rings points xmin ymin xmax ymax wkb_hex
#
# - counties_sample.tsv (checked in): Massachusetts, plus every county
#   nationally whose polygon has holes, so rings past the first are covered.
# - local/us_counties.tsv (not checked in): all 3,222 counties.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
catalog=${1:-$here/../../../../DuckMap/ingestion/data/duckmap.ducklake}
catalog=$(cd "$(dirname "$catalog")" && pwd)/$(basename "$catalog")
mkdir -p "$here/local"

columns="geoid, name, ST_GeometryType(geom) AS type,
    ST_NumGeometries(geom) AS parts,
    (SELECT sum(1 + ST_NumInteriorRings(d.geom))::BIGINT
       FROM (SELECT unnest(ST_Dump(geom)) AS d)) AS rings,
    ST_NPoints(geom) AS points,
    ST_XMin(geom)::VARCHAR AS xmin, ST_YMin(geom)::VARCHAR AS ymin,
    ST_XMax(geom)::VARCHAR AS xmax, ST_YMax(geom)::VARCHAR AS ymax,
    lower(hex(ST_AsWKB(geom))) AS wkb_hex"

duckdb -c "
LOAD spatial;
ATTACH 'ducklake:$catalog' AS duckmap (READ_ONLY);
COPY (
    SELECT $columns FROM duckmap.counties
    WHERE state_fips = '25'
       OR (SELECT sum(ST_NumInteriorRings(d.geom))
             FROM (SELECT unnest(ST_Dump(geom)) AS d)) > 0
    ORDER BY geoid
) TO '$here/counties_sample.tsv' (HEADER, DELIMITER '\t');
COPY (
    SELECT $columns FROM duckmap.counties ORDER BY geoid
) TO '$here/local/us_counties.tsv' (HEADER, DELIMITER '\t');
"
