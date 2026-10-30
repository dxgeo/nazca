#!/usr/bin/env python3
"""Extracts one tile from a PMTiles v3 archive, decompressed (std lib only).

    python3 -I tests/fixtures/pmtiles_tile.py ARCHIVE Z X Y OUT

How tests/fixtures/basemap/ma_6_19_23.{mvt,mlt} were cut from DuckMap's
basemaps (ingestion/data/basemaps/counties_basemap_{mvt,mlt}.pmtiles), and
how `make_basemap_fixtures.sh` fills tests/fixtures/local/basemap/. nazca
itself reads no containers: this is test tooling.
"""
import gzip, struct, sys

def varint(b, i):
    v = s = 0
    while True:
        c = b[i]; i += 1
        v |= (c & 0x7f) << s; s += 7
        if c < 0x80: return v, i

def rotate(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1: x, y = n - 1 - x, n - 1 - y
        x, y = y, x
    return x, y

def tile_id(z, x, y):
    acc = sum(4 ** i for i in range(z))
    n = 1 << z; d = 0; s = n // 2
    while s > 0:
        rx = 1 if (x & s) else 0; ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = rotate(n, x, y, rx, ry)
        s //= 2
    return acc + d

def decomp(data, c):
    return gzip.decompress(data) if c == 2 else data

def directory(b):
    n, i = varint(b, 0)
    ids, runs, lens, offs = [], [], [], []
    last = 0
    for _ in range(n):
        v, i = varint(b, i); last += v; ids.append(last)
    for _ in range(n):
        v, i = varint(b, i); runs.append(v)
    for _ in range(n):
        v, i = varint(b, i); lens.append(v)
    for k in range(n):
        v, i = varint(b, i)
        offs.append(offs[k - 1] + lens[k - 1] if v == 0 and k > 0 else v - 1)
    return list(zip(ids, runs, lens, offs))

def find(f, h, tid):
    root_off, root_len = struct.unpack_from('<QQ', h, 8)
    leaf_off = struct.unpack_from('<Q', h, 40)[0]
    data_off = struct.unpack_from('<Q', h, 56)[0]
    ic = h[97]
    off, ln = root_off, root_len
    for _ in range(4):
        f.seek(off); entries = directory(decomp(f.read(ln), ic))
        hit = None
        for e in entries:
            if e[0] <= tid: hit = e
            else: break
        if hit is None: return None
        eid, run, l, o = hit
        if run == 0:
            off, ln = leaf_off + o, l; continue
        if tid < eid + run:
            f.seek(data_off + o); return f.read(l)
        return None

path, z, x, y, out = sys.argv[1], *map(int, sys.argv[2:5]), sys.argv[5]
with open(path, 'rb') as f:
    h = f.read(127)
    assert h[:7] == b'PMTiles' and h[7] == 3
    print('tile_compression', h[98], 'tile_type', h[99], 'zooms', h[100], h[101])
    t = find(f, h, tile_id(z, x, y))
    if t is None: sys.exit('no tile')
    t = decomp(t, h[98])
    open(out, 'wb').write(t)
    print(out, len(t), 'bytes')
