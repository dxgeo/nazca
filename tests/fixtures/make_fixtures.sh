#!/bin/sh
# Regenerates nazca's WKB test fixtures from a DuckLake catalog with a
# `counties` table (geoid, name, state_fips, geom).
#
#   tests/fixtures/make_fixtures.sh path/to/catalog.ducklake
#
# Needs the `duckdb` CLI (1.5.x) with the spatial extension. Nothing else may
# have the catalog open: a second process cannot ATTACH it while another holds
# it.
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
catalog=${1:?usage: make_fixtures.sh path/to/catalog.ducklake}
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
ATTACH 'ducklake:$catalog' AS catalog (READ_ONLY);
COPY (
    SELECT $columns FROM catalog.counties
    WHERE state_fips = '25'
       OR (SELECT sum(ST_NumInteriorRings(d.geom))
             FROM (SELECT unnest(ST_Dump(geom)) AS d)) > 0
    ORDER BY geoid
) TO '$here/counties_sample.tsv' (HEADER, DELIMITER '\t');
COPY (
    SELECT $columns FROM catalog.counties ORDER BY geoid
) TO '$here/local/us_counties.tsv' (HEADER, DELIMITER '\t');
"
