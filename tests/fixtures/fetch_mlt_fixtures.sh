#!/bin/sh
# Fetches the MLT specification's v1 conformance fixtures into
# tests/fixtures/local/mlt/ (not checked in: the spec repository's test data
# is licensed per source, so nazca does not redistribute it).
#
#   tests/fixtures/fetch_mlt_fixtures.sh
#
# Pinned to one commit of https://github.com/maplibre/maplibre-tile-spec, so
# the expected output does not move under the tests:
# - synthetic/: every tile under test/synthetic/0x01/ with the GeoJSON a
#   conforming decoder must produce from it;
# - fastpfor/: the cross-language FastPFOR vectors (test/fixtures/fastpfor/).
#
# Then `cargo test -p nazca --features mvt,mlt -- --ignored`.
set -eu

commit=71f20ee897a285f54bc461974c07713af04aeb7b
repo=maplibre/maplibre-tile-spec
here=$(cd "$(dirname "$0")" && pwd)
out=$here/local/mlt
mkdir -p "$out/synthetic" "$out/fastpfor"

curl -sfL "https://api.github.com/repos/$repo/git/trees/$commit?recursive=1" |
    python3 -I -c '
import json, sys
for t in json.load(sys.stdin)["tree"]:
    p = t["path"]
    if p.startswith("test/synthetic/0x01/") and p.endswith((".mlt", ".json")):
        print(p)
' > "$out/list.txt"

while read -r path; do
    curl -sfL "https://raw.githubusercontent.com/$repo/$commit/$path" -o "$out/synthetic/$(basename "$path")"
done < "$out/list.txt"

for n in 1 2 3 4; do
    for kind in encoded decoded; do
        f=vector${n}_$kind.bin
        curl -sfL "https://raw.githubusercontent.com/$repo/$commit/test/fixtures/fastpfor/$f" -o "$out/fastpfor/$f"
    done
done
echo "fetched $(wc -l < "$out/list.txt") files into $out"
