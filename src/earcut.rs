//! Polygon triangulation: a port of mapbox/earcut (v3).
//!
//! Ear clipping over a circular linked list, with holes bridged into the
//! outer ring first (Eberly's method), points hashed along a z-order curve
//! once a polygon has more than 80 of them so an ear test only visits nearby
//! points, and three fallbacks for bad input: filter collinear and repeated
//! points, cure small self-intersections, and finally split the polygon in
//! two along a valid diagonal. The nodes live in an arena (`Vec<Node>`)
//! that an [`Earcut`] keeps between polygons, so a long-lived one allocates
//! nothing per polygon once warmed up.
//!
//! The functions keep earcut's names and orientation conventions, so this
//! file can be read against the original: [`area`] is *negative* for a
//! counter-clockwise turn, and the outer ring is linked counter-clockwise
//! (Y up), holes clockwise.
//!
//! Ported from <https://github.com/mapbox/earcut>, under its licence:
//!
//! > ISC License
//! >
//! > Copyright (c) 2016, Mapbox
//! >
//! > Permission to use, copy, modify, and/or distribute this software for any
//! > purpose with or without fee is hereby granted, provided that the above
//! > copyright notice and this permission notice appear in all copies.
//! >
//! > THE SOFTWARE IS PROVIDED "AS IS" AND ISC DISCLAIMS ALL WARRANTIES WITH
//! > REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
//! > MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
//! > SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
//! > WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
//! > ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
//! > OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

use std::cmp::Ordering;
use std::ops::Range;

const NONE: usize = usize::MAX;

#[derive(Debug, Clone, Copy)]
struct Node {
    /// The point's index in the caller's slice.
    i: u32,
    x: f64,
    y: f64,
    prev: usize,
    next: usize,
    /// Z-order curve value, 0 until computed.
    z: u32,
    prev_z: usize,
    next_z: usize,
    /// A hole of a single point.
    steiner: bool,
}

/// A reusable triangulator. See the module docs.
#[derive(Debug, Default)]
pub struct Earcut {
    nodes: Vec<Node>,
    queue: Vec<usize>,
    order: Vec<usize>,
}

/// The z-order hashing parameters: the outer ring's minimum corner and the
/// scale to 15-bit integers.
#[derive(Clone, Copy)]
struct Hash {
    min_x: f64,
    min_y: f64,
    inv_size: f64,
}

impl Earcut {
    pub fn new() -> Earcut {
        Earcut::default()
    }

    /// Triangulates one polygon. `pts` holds every ring's points; `rings`
    /// gives each ring's range in it, the exterior first, each without a
    /// closing repeat of its first point. Appends counter-clockwise triangles
    /// to `out` as indices into `pts`. Either winding is accepted for any
    /// ring.
    pub fn triangulate(&mut self, pts: &[[f64; 2]], rings: &[Range<usize>], out: &mut Vec<u32>) {
        self.nodes.clear();
        let Some(outer) = rings.first() else { return };
        let mut outer_node = self.linked_list(pts, outer.clone(), true);
        if outer_node == NONE || self.nodes[outer_node].next == self.nodes[outer_node].prev {
            return;
        }
        if rings.len() > 1 {
            outer_node = self.eliminate_holes(pts, &rings[1..], outer_node);
        }

        // Past 80 points, hash along a z-order curve over the outer ring's
        // bounding box.
        let total: usize = rings.iter().map(|r| r.len()).sum();
        let mut hash = None;
        if total > 80 {
            let (mut min_x, mut min_y, mut max_x, mut max_y) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
            for p in &pts[outer.clone()] {
                min_x = min_x.min(p[0]);
                min_y = min_y.min(p[1]);
                max_x = max_x.max(p[0]);
                max_y = max_y.max(p[1]);
            }
            let size = (max_x - min_x).max(max_y - min_y);
            if size != 0.0 {
                hash = Some(Hash { min_x, min_y, inv_size: 32767.0 / size });
            }
        }
        self.earcut_linked(outer_node, out, hash, 0);
    }

    // --- the linked list --------------------------------------------------

    /// Links a ring's points in the requested winding.
    fn linked_list(&mut self, pts: &[[f64; 2]], ring: Range<usize>, clockwise: bool) -> usize {
        let mut last = NONE;
        if clockwise == (signed_area(&pts[ring.clone()]) > 0.0) {
            for i in ring {
                last = self.insert_node(i as u32, pts[i], last);
            }
        } else {
            for i in ring.rev() {
                last = self.insert_node(i as u32, pts[i], last);
            }
        }
        if last != NONE && self.equals(last, self.nodes[last].next) {
            self.remove_node(last);
            last = self.nodes[last].next;
        }
        last
    }

    fn insert_node(&mut self, i: u32, [x, y]: [f64; 2], last: usize) -> usize {
        let p = self.nodes.len();
        let mut node = Node { i, x, y, prev: p, next: p, z: 0, prev_z: NONE, next_z: NONE, steiner: false };
        if last != NONE {
            node.next = self.nodes[last].next;
            node.prev = last;
            let after = self.nodes[last].next;
            self.nodes[after].prev = p;
            self.nodes[last].next = p;
        }
        self.nodes.push(node);
        p
    }

    /// Unlinks `p` from both lists. `p`'s own links are left as they were,
    /// which the callers rely on to step past it.
    fn remove_node(&mut self, p: usize) {
        let Node { prev, next, prev_z, next_z, .. } = self.nodes[p];
        self.nodes[next].prev = prev;
        self.nodes[prev].next = next;
        if prev_z != NONE {
            self.nodes[prev_z].next_z = next_z;
        }
        if next_z != NONE {
            self.nodes[next_z].prev_z = prev_z;
        }
    }

    fn equals(&self, a: usize, b: usize) -> bool {
        self.nodes[a].x == self.nodes[b].x && self.nodes[a].y == self.nodes[b].y
    }

    fn area(&self, p: usize, q: usize, r: usize) -> f64 {
        let (p, q, r) = (&self.nodes[p], &self.nodes[q], &self.nodes[r]);
        area([p.x, p.y], [q.x, q.y], [r.x, r.y])
    }

    /// Removes repeated and collinear points between `start` and `end`.
    fn filter_points(&mut self, start: usize, end: usize) -> usize {
        if start == NONE {
            return start;
        }
        let mut end = if end == NONE { start } else { end };
        let mut p = start;
        loop {
            let mut again = false;
            let (prev, next) = (self.nodes[p].prev, self.nodes[p].next);
            if !self.nodes[p].steiner && (self.equals(p, next) || self.area(prev, p, next) == 0.0) {
                self.remove_node(p);
                p = self.nodes[p].prev;
                end = p;
                if p == self.nodes[p].next {
                    break;
                }
                again = true;
            } else {
                p = next;
            }
            if !(again || p != end) {
                break;
            }
        }
        end
    }

    // --- ear slicing ------------------------------------------------------

    fn earcut_linked(&mut self, ear: usize, out: &mut Vec<u32>, hash: Option<Hash>, pass: u8) {
        if ear == NONE {
            return;
        }
        if pass == 0
            && let Some(h) = hash
        {
            self.index_curve(ear, h);
        }
        let mut ear = ear;
        let mut stop = ear;
        while self.nodes[ear].prev != self.nodes[ear].next {
            let (prev, next) = (self.nodes[ear].prev, self.nodes[ear].next);
            let is_ear = match hash {
                Some(h) => self.is_ear_hashed(ear, h),
                None => self.is_ear(ear),
            };
            if is_ear {
                out.extend_from_slice(&[self.nodes[prev].i, self.nodes[ear].i, self.nodes[next].i]);
                self.remove_node(ear);
                // Skipping the next vertex leaves fewer sliver triangles.
                ear = self.nodes[next].next;
                stop = ear;
                continue;
            }
            ear = next;
            if ear == stop {
                match pass {
                    // No ear left: filter repeated and collinear points and
                    // try again,
                    0 => {
                        let e = self.filter_points(ear, NONE);
                        self.earcut_linked(e, out, hash, 1);
                    }
                    // then cure small self-intersections,
                    1 => {
                        let e = self.filter_points(ear, NONE);
                        let e = self.cure_local_intersections(e, out);
                        self.earcut_linked(e, out, hash, 2);
                    }
                    // then split the polygon in two.
                    _ => self.split_earcut(ear, out, hash),
                }
                break;
            }
        }
    }

    /// Whether `ear` and its neighbours form a triangle with no other point
    /// of the polygon inside it.
    fn is_ear(&self, ear: usize) -> bool {
        let (a, b, c) = (self.nodes[ear].prev, ear, self.nodes[ear].next);
        if self.area(a, b, c) >= 0.0 {
            return false; // reflex
        }
        let (pa, pb, pc) = (self.xy(a), self.xy(b), self.xy(c));
        let (x0, y0, x1, y1) = tri_bounds(pa, pb, pc);
        let mut p = self.nodes[c].next;
        while p != a {
            let pp = self.xy(p);
            if pp[0] >= x0
                && pp[0] <= x1
                && pp[1] >= y0
                && pp[1] <= y1
                && point_in_triangle_except_first(pa, pb, pc, pp)
                && self.area(self.nodes[p].prev, p, self.nodes[p].next) >= 0.0
            {
                return false;
            }
            p = self.nodes[p].next;
        }
        true
    }

    /// [`Self::is_ear`], visiting only the points whose z-order lies within
    /// the triangle's bounding box.
    fn is_ear_hashed(&self, ear: usize, h: Hash) -> bool {
        let (a, b, c) = (self.nodes[ear].prev, ear, self.nodes[ear].next);
        if self.area(a, b, c) >= 0.0 {
            return false;
        }
        let (pa, pb, pc) = (self.xy(a), self.xy(b), self.xy(c));
        let (x0, y0, x1, y1) = tri_bounds(pa, pb, pc);
        let min_z = z_order(x0, y0, h);
        let max_z = z_order(x1, y1, h);
        let blocks = |p: usize| {
            let pp = self.xy(p);
            pp[0] >= x0
                && pp[0] <= x1
                && pp[1] >= y0
                && pp[1] <= y1
                && p != a
                && p != c
                && point_in_triangle_except_first(pa, pb, pc, pp)
                && self.area(self.nodes[p].prev, p, self.nodes[p].next) >= 0.0
        };
        let mut p = self.nodes[ear].prev_z;
        let mut n = self.nodes[ear].next_z;
        // Both directions at once while both are in range,
        while p != NONE && self.nodes[p].z >= min_z && n != NONE && self.nodes[n].z <= max_z {
            if blocks(p) {
                return false;
            }
            p = self.nodes[p].prev_z;
            if blocks(n) {
                return false;
            }
            n = self.nodes[n].next_z;
        }
        // then whichever is left.
        while p != NONE && self.nodes[p].z >= min_z {
            if blocks(p) {
                return false;
            }
            p = self.nodes[p].prev_z;
        }
        while n != NONE && self.nodes[n].z <= max_z {
            if blocks(n) {
                return false;
            }
            n = self.nodes[n].next_z;
        }
        true
    }

    /// Computes z-order values and links the ring's nodes in z-order.
    fn index_curve(&mut self, start: usize, h: Hash) {
        self.order.clear();
        let mut p = start;
        loop {
            if self.nodes[p].z == 0 {
                self.nodes[p].z = z_order(self.nodes[p].x, self.nodes[p].y, h);
            }
            self.order.push(p);
            p = self.nodes[p].next;
            if p == start {
                break;
            }
        }
        let mut order = std::mem::take(&mut self.order);
        order.sort_by_key(|&k| self.nodes[k].z);
        for w in order.windows(2) {
            self.nodes[w[0]].next_z = w[1];
            self.nodes[w[1]].prev_z = w[0];
        }
        self.nodes[order[0]].prev_z = NONE;
        self.nodes[order[order.len() - 1]].next_z = NONE;
        self.order = order;
    }

    /// Clips the ear of every small self-intersection: a `p`, `p.next` pair
    /// whose neighbours' edges cross.
    fn cure_local_intersections(&mut self, start: usize, out: &mut Vec<u32>) -> usize {
        let mut start = start;
        let mut p = start;
        loop {
            let a = self.nodes[p].prev;
            let pn = self.nodes[p].next;
            let b = self.nodes[pn].next;
            if !self.equals(a, b) && self.intersects(a, p, pn, b) && self.locally_inside(a, b) && self.locally_inside(b, a) {
                out.extend_from_slice(&[self.nodes[a].i, self.nodes[p].i, self.nodes[b].i]);
                self.remove_node(p);
                self.remove_node(pn);
                p = b;
                start = b;
            }
            p = self.nodes[p].next;
            if p == start {
                break;
            }
        }
        self.filter_points(p, NONE)
    }

    /// Splits the polygon along the first valid diagonal found and
    /// triangulates each half.
    fn split_earcut(&mut self, start: usize, out: &mut Vec<u32>, hash: Option<Hash>) {
        let mut a = start;
        loop {
            let mut b = self.nodes[self.nodes[a].next].next;
            while b != self.nodes[a].prev {
                if self.nodes[a].i != self.nodes[b].i && self.is_valid_diagonal(a, b) {
                    let c = self.split_polygon(a, b);
                    let a = self.filter_points(a, self.nodes[a].next);
                    let c = self.filter_points(c, self.nodes[c].next);
                    self.earcut_linked(a, out, hash, 0);
                    self.earcut_linked(c, out, hash, 0);
                    return;
                }
                b = self.nodes[b].next;
            }
            a = self.nodes[a].next;
            if a == start {
                break;
            }
        }
    }

    // --- holes ------------------------------------------------------------

    fn eliminate_holes(&mut self, pts: &[[f64; 2]], holes: &[Range<usize>], mut outer: usize) -> usize {
        let mut queue = std::mem::take(&mut self.queue);
        queue.clear();
        for hole in holes {
            let list = self.linked_list(pts, hole.clone(), false);
            if list == NONE {
                continue;
            }
            if list == self.nodes[list].next {
                self.nodes[list].steiner = true;
            }
            queue.push(self.get_leftmost(list));
        }
        queue.sort_by(|&a, &b| self.compare_x_y_slope(a, b));
        // Left to right.
        for &hole in &queue {
            outer = self.eliminate_hole(hole, outer);
        }
        self.queue = queue;
        outer
    }

    fn compare_x_y_slope(&self, a: usize, b: usize) -> Ordering {
        let (na, nb) = (&self.nodes[a], &self.nodes[b]);
        let by_xy = na.x.total_cmp(&nb.x).then(na.y.total_cmp(&nb.y));
        if by_xy != Ordering::Equal {
            return by_xy;
        }
        let slope = |n: &Node| {
            let next = &self.nodes[n.next];
            (next.y - n.y) / (next.x - n.x)
        };
        slope(na).partial_cmp(&slope(nb)).unwrap_or(Ordering::Equal)
    }

    /// Bridges a hole into the outer ring.
    fn eliminate_hole(&mut self, hole: usize, outer: usize) -> usize {
        let bridge = self.find_hole_bridge(hole, outer);
        if bridge == NONE {
            return outer;
        }
        let bridge_reverse = self.split_polygon(bridge, hole);
        // Filter collinear points around the cuts.
        self.filter_points(bridge_reverse, self.nodes[bridge_reverse].next);
        self.filter_points(bridge, self.nodes[bridge].next)
    }

    /// David Eberly's algorithm for a bridge from a hole's leftmost point to
    /// a visible outer point.
    fn find_hole_bridge(&self, hole: usize, outer: usize) -> usize {
        let mut p = outer;
        let (hx, hy) = (self.nodes[hole].x, self.nodes[hole].y);
        let mut qx = f64::NEG_INFINITY;
        let mut m = NONE;

        // The nearest outer segment a ray from the hole leftward meets; its
        // endpoint with the smaller x is the candidate, unless the ray meets
        // a vertex, which then is the bridge.
        if self.equals(hole, p) {
            return p;
        }
        loop {
            let next = self.nodes[p].next;
            if self.equals(hole, next) {
                return next;
            }
            let (px, py, nx, ny) = (self.nodes[p].x, self.nodes[p].y, self.nodes[next].x, self.nodes[next].y);
            if hy <= py && hy >= ny && ny != py {
                let x = px + (hy - py) * (nx - px) / (ny - py);
                if x <= hx && x > qx {
                    qx = x;
                    m = if px < nx { p } else { next };
                    if x == hx {
                        return m; // the hole touches the segment: its leftmost endpoint
                    }
                }
            }
            p = next;
            if p == outer {
                break;
            }
        }
        if m == NONE {
            return NONE;
        }

        // If any point lies inside the triangle of the hole point, the
        // intersection and the candidate, take the one with the smallest
        // angle to the ray instead.
        let stop = m;
        let (mx, my) = (self.nodes[m].x, self.nodes[m].y);
        let mut tan_min = f64::INFINITY;
        p = m;
        loop {
            let (px, py) = (self.nodes[p].x, self.nodes[p].y);
            let (ax, cx) = if hy < my { (hx, qx) } else { (qx, hx) };
            if hx >= px && px >= mx && hx != px && point_in_triangle([ax, hy], [mx, my], [cx, hy], [px, py]) {
                let tan = (hy - py).abs() / (hx - px);
                if self.locally_inside(p, hole)
                    && (tan < tan_min
                        || (tan == tan_min && (px > self.nodes[m].x || (px == self.nodes[m].x && self.sector_contains_sector(m, p)))))
                {
                    m = p;
                    tan_min = tan;
                }
            }
            p = self.nodes[p].next;
            if p == stop {
                break;
            }
        }
        m
    }

    /// Whether `m`'s sector contains `p`'s.
    fn sector_contains_sector(&self, m: usize, p: usize) -> bool {
        self.area(self.nodes[m].prev, m, self.nodes[p].prev) < 0.0 && self.area(self.nodes[p].next, m, self.nodes[m].next) < 0.0
    }

    fn get_leftmost(&self, start: usize) -> usize {
        let mut p = start;
        let mut leftmost = start;
        loop {
            let (n, l) = (&self.nodes[p], &self.nodes[leftmost]);
            if n.x < l.x || (n.x == l.x && n.y < l.y) {
                leftmost = p;
            }
            p = self.nodes[p].next;
            if p == start {
                break;
            }
        }
        leftmost
    }

    // --- diagonals --------------------------------------------------------

    /// Whether `a`--`b` can split the polygon in two.
    fn is_valid_diagonal(&self, a: usize, b: usize) -> bool {
        let (na, nb) = (&self.nodes[a], &self.nodes[b]);
        self.nodes[na.next].i != nb.i
            && self.nodes[na.prev].i != nb.i
            && !self.intersects_polygon(a, b)
            && ((self.locally_inside(a, b)
                && self.locally_inside(b, a)
                && self.middle_inside(a, b)
                && (self.area(na.prev, a, nb.prev) != 0.0 || self.area(a, nb.prev, b) != 0.0))
                || (self.equals(a, b) && self.area(na.prev, a, na.next) > 0.0 && self.area(nb.prev, b, nb.next) > 0.0))
    }

    /// Whether segments `p1`--`q1` and `p2`--`q2` intersect, touching
    /// included.
    fn intersects(&self, p1: usize, q1: usize, p2: usize, q2: usize) -> bool {
        let o1 = sign(self.area(p1, q1, p2));
        let o2 = sign(self.area(p1, q1, q2));
        let o3 = sign(self.area(p2, q2, p1));
        let o4 = sign(self.area(p2, q2, q1));
        if o1 != o2 && o3 != o4 {
            return true;
        }
        (o1 == 0 && self.on_segment(p1, p2, q1))
            || (o2 == 0 && self.on_segment(p1, q2, q1))
            || (o3 == 0 && self.on_segment(p2, p1, q2))
            || (o4 == 0 && self.on_segment(p2, q1, q2))
    }

    /// For collinear `p`, `q`, `r`: whether `q` lies on segment `p`--`r`.
    fn on_segment(&self, p: usize, q: usize, r: usize) -> bool {
        let (p, q, r) = (&self.nodes[p], &self.nodes[q], &self.nodes[r]);
        q.x <= p.x.max(r.x) && q.x >= p.x.min(r.x) && q.y <= p.y.max(r.y) && q.y >= p.y.min(r.y)
    }

    /// Whether `a`--`b` crosses any polygon edge not touching `a` or `b`.
    fn intersects_polygon(&self, a: usize, b: usize) -> bool {
        let (ai, bi) = (self.nodes[a].i, self.nodes[b].i);
        let mut p = a;
        loop {
            let next = self.nodes[p].next;
            let (pi, ni) = (self.nodes[p].i, self.nodes[next].i);
            if pi != ai && ni != ai && pi != bi && ni != bi && self.intersects(p, next, a, b) {
                return true;
            }
            p = next;
            if p == a {
                break;
            }
        }
        false
    }

    /// Whether the diagonal `a`--`b` starts into the polygon's interior at
    /// `a`.
    fn locally_inside(&self, a: usize, b: usize) -> bool {
        let (prev, next) = (self.nodes[a].prev, self.nodes[a].next);
        if self.area(prev, a, next) < 0.0 {
            self.area(a, b, next) >= 0.0 && self.area(a, prev, b) >= 0.0
        } else {
            self.area(a, b, prev) < 0.0 || self.area(a, next, b) < 0.0
        }
    }

    /// Whether the midpoint of `a`--`b` lies inside the polygon.
    fn middle_inside(&self, a: usize, b: usize) -> bool {
        let (px, py) = ((self.nodes[a].x + self.nodes[b].x) / 2.0, (self.nodes[a].y + self.nodes[b].y) / 2.0);
        let mut p = a;
        let mut inside = false;
        loop {
            let n = &self.nodes[p];
            let next = &self.nodes[n.next];
            if ((n.y > py) != (next.y > py)) && next.y != n.y && px < (next.x - n.x) * (py - n.y) / (next.y - n.y) + n.x {
                inside = !inside;
            }
            p = n.next;
            if p == a {
                break;
            }
        }
        inside
    }

    /// Links `a` to `b` with a bridge, splitting the ring in two; or, if they
    /// were in separate rings, merging them. Returns the copy of `b`.
    fn split_polygon(&mut self, a: usize, b: usize) -> usize {
        let (na, nb) = (self.nodes[a], self.nodes[b]);
        let a2 = self.nodes.len();
        let b2 = a2 + 1;
        let fresh = |n: Node| Node { prev: NONE, next: NONE, prev_z: NONE, next_z: NONE, z: 0, steiner: false, ..n };
        self.nodes.push(fresh(na));
        self.nodes.push(fresh(nb));
        let (an, bp) = (na.next, nb.prev);

        self.nodes[a].next = b;
        self.nodes[b].prev = a;
        self.nodes[a2].next = an;
        self.nodes[an].prev = a2;
        self.nodes[b2].next = a2;
        self.nodes[a2].prev = b2;
        self.nodes[bp].next = b2;
        self.nodes[b2].prev = bp;
        b2
    }

    fn xy(&self, p: usize) -> [f64; 2] {
        [self.nodes[p].x, self.nodes[p].y]
    }
}

/// Earcut's orientation: *negative* for a counter-clockwise turn.
fn area(p: [f64; 2], q: [f64; 2], r: [f64; 2]) -> f64 {
    (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
}

/// Earcut's ring area: positive for a counter-clockwise ring.
fn signed_area(ring: &[[f64; 2]]) -> f64 {
    let mut sum = 0.0;
    let mut j = ring.len().wrapping_sub(1);
    for i in 0..ring.len() {
        sum += (ring[j][0] - ring[i][0]) * (ring[i][1] + ring[j][1]);
        j = i;
    }
    sum
}

fn sign(v: f64) -> i8 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

fn tri_bounds(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> (f64, f64, f64, f64) {
    (a[0].min(b[0]).min(c[0]), a[1].min(b[1]).min(c[1]), a[0].max(b[0]).max(c[0]), a[1].max(b[1]).max(c[1]))
}

/// Whether `p` lies inside triangle `abc`, edges included.
fn point_in_triangle(a: [f64; 2], b: [f64; 2], c: [f64; 2], p: [f64; 2]) -> bool {
    (c[0] - p[0]) * (a[1] - p[1]) >= (a[0] - p[0]) * (c[1] - p[1])
        && (a[0] - p[0]) * (b[1] - p[1]) >= (b[0] - p[0]) * (a[1] - p[1])
        && (b[0] - p[0]) * (c[1] - p[1]) >= (c[0] - p[0]) * (b[1] - p[1])
}

fn point_in_triangle_except_first(a: [f64; 2], b: [f64; 2], c: [f64; 2], p: [f64; 2]) -> bool {
    a != p && point_in_triangle(a, b, c, p)
}

/// A point's position on a 15-bit z-order curve over the hash's box.
fn z_order(x: f64, y: f64, h: Hash) -> u32 {
    // `as` saturates, so a point outside the box clamps to its edge.
    let spread = |v: f64| {
        let mut v = v as u32;
        v = (v | (v << 8)) & 0x00FF_00FF;
        v = (v | (v << 4)) & 0x0F0F_0F0F;
        v = (v | (v << 2)) & 0x3333_3333;
        (v | (v << 1)) & 0x5555_5555
    };
    spread((x - h.min_x) * h.inv_size) | (spread((y - h.min_y) * h.inv_size) << 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The triangles' total area, positive when counter-clockwise.
    fn tri_area(pts: &[[f64; 2]], tris: &[u32]) -> f64 {
        tris.chunks(3).map(|t| -area(pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]) / 2.0).sum()
    }

    fn polygon(rings: &[&[[f64; 2]]]) -> (Vec<[f64; 2]>, Vec<Range<usize>>) {
        let mut pts = Vec::new();
        let mut ranges = Vec::new();
        for r in rings {
            let start = pts.len();
            pts.extend_from_slice(r);
            ranges.push(start..pts.len());
        }
        (pts, ranges)
    }

    fn run(rings: &[&[[f64; 2]]]) -> (Vec<[f64; 2]>, Vec<u32>) {
        let (pts, ranges) = polygon(rings);
        let mut out = Vec::new();
        Earcut::new().triangulate(&pts, &ranges, &mut out);
        (pts, out)
    }

    fn all_ccw(pts: &[[f64; 2]], tris: &[u32]) -> bool {
        tris.chunks(3).all(|t| area(pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]) < 0.0)
    }

    #[test]
    fn square_either_winding() {
        for sq in [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]] {
            let (pts, out) = run(&[&sq]);
            assert_eq!(out.len(), 6);
            assert!((tri_area(&pts, &out) - 1.0).abs() < 1e-12);
            assert!(all_ccw(&pts, &out));
        }
    }

    #[test]
    fn concave_polygon() {
        let u = [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [2.0, 3.0], [2.0, 1.0], [1.0, 1.0], [1.0, 3.0], [0.0, 3.0]];
        let (pts, out) = run(&[&u]);
        assert!((tri_area(&pts, &out) - 7.0).abs() < 1e-12);
        assert_eq!(out.len() / 3, u.len() - 2);
        assert!(all_ccw(&pts, &out));
    }

    #[test]
    fn holes_in_either_winding() {
        let outer = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let h1 = [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
        let h2 = [[6.0, 6.0], [6.0, 9.0], [9.0, 9.0], [9.0, 6.0]];
        let (pts, out) = run(&[&outer, &h1, &h2]);
        assert!((tri_area(&pts, &out) - (100.0 - 4.0 - 9.0)).abs() < 1e-9);
        assert_eq!(out.len() / 3, 12 + 2 * 2 - 2);
        assert!(all_ccw(&pts, &out));
    }

    #[test]
    fn a_hole_touching_the_outer_ring() {
        let outer = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let hole = [[0.0, 1.0], [1.0, 2.0], [0.0, 3.0]]; // one edge on the outer ring's
        let (pts, out) = run(&[&outer, &hole]);
        assert!((tri_area(&pts, &out) - (16.0 - 1.0)).abs() < 1e-9);
        assert!(all_ccw(&pts, &out));
    }

    #[test]
    fn many_points_take_the_hashed_path() {
        // A 2000-point circle with a 500-point hole: well past 80 points.
        let circle = |r: f64, n: usize, ccw: bool| -> Vec<[f64; 2]> {
            (0..n).map(|k| {
                let t = 2.0 * std::f64::consts::PI * k as f64 / n as f64 * if ccw { 1.0 } else { -1.0 };
                [r * t.cos(), r * t.sin()]
            }).collect()
        };
        let (outer, hole) = (circle(10.0, 2000, true), circle(4.0, 500, false));
        let (pts, out) = run(&[&outer, &hole]);
        let expected = (signed_area(&outer) - signed_area(&hole).abs()) / 2.0;
        assert!((tri_area(&pts, &out) - expected).abs() < 1e-9 * expected, "{} vs {expected}", tri_area(&pts, &out));
        assert_eq!(out.len() / 3, 2500 + 2 - 2);
        assert!(all_ccw(&pts, &out));
    }

    #[test]
    fn repeated_and_collinear_points() {
        let ring = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 2.0], [0.0, 1.0]];
        let (pts, out) = run(&[&ring]);
        assert!((tri_area(&pts, &out) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn degenerate_input_terminates() {
        let (_, out) = run(&[&[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]);
        assert!(out.is_empty());
        // A bow tie: self-intersecting; earcut cures what it can.
        run(&[&[[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]]]);
        // Fewer than three points.
        let (_, out) = run(&[&[[0.0, 0.0], [1.0, 1.0]]]);
        assert!(out.is_empty());
        let (_, out) = run(&[]);
        assert!(out.is_empty());
    }
}
