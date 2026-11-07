#!/bin/sh
# Cuts matching MVT and MLT tiles from PMTiles basemaps into
# tests/fixtures/local/basemap/ (not checked in), for the ignored
# `every_local_basemap_tile_decodes_alike` test.
#
#   tests/fixtures/make_basemap_fixtures.sh path/to/basemaps
#
# The directory holds counties_basemap_mvt.pmtiles and
# counties_basemap_mlt.pmtiles.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
src=${1:?usage: make_basemap_fixtures.sh path/to/basemaps}
out=$here/local/basemap
mkdir -p "$out"
for zxy in "0 0 0" "1 0 0" "2 1 1" "3 2 2" "4 4 5" "5 9 11" "6 19 23" "6 13 24"; do
    set -- $zxy
    for fmt in mvt mlt; do
        python3 -I "$here/pmtiles_tile.py" "$src/counties_basemap_$fmt.pmtiles" "$1" "$2" "$3" "$out/${1}_${2}_${3}.$fmt" >/dev/null
    done
done
ls "$out"
