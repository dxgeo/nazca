//! The tile cache: prepared tiles kept between frames, under one byte budget
//! (`../plans/rendering.org` § TILE CACHE).
//!
//! The cache is generic over what it holds, so one cache and one budget
//! cover vector and raster alike: the embedder's value is typically an enum
//! of the backend's uploaded mesh and texture. Once a tile is uploaded its
//! CPU copy is dropped, and the GPU handle is all the cache keeps; GPU
//! objects are not freed by `Drop`, so every value the cache lets go of is
//! handed back to the caller to free.
//!
//! A frame runs:
//!
//! 1. [`TileCache::begin_frame`];
//! 2. per source, a [`TileSelector`](crate::select::TileSelector) lists the
//!    tiles the camera needs and [`TileCache::plan`] splits them into what to
//!    draw now and what to fetch;
//! 3. the embedder fetches and prepares those tiles, at its own pace, and
//!    hands each to [`TileCache::insert`] (or [`TileCache::fail`]).
//!
//! A needed tile not yet in the cache is stood in for by its nearest cached
//! ancestor, drawn scaled, or failing that by whichever of its children are
//! cached, so zooming in or out shows something while a level loads.
//!
//! Eviction is least-recently-used, by the order tiles were last planned or
//! inserted, but never takes a tile the current frame draws: when those
//! alone exceed the budget the cache holds them and runs over, rather than
//! leave a hole in what is on screen.

use crate::grid::{TileGrid, TileId};
use std::collections::{BTreeMap, HashMap, HashSet};

/// A tile of one source. A source is whatever the embedder fetches tiles
/// from (a layer's tile route, a basemap, a raster): its numbering is the
/// embedder's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileKey {
    pub source: u32,
    pub tile: TileId,
}

/// One frame's work for one source, written by [`TileCache::plan`]. Kept by
/// the caller and reused, so planning allocates nothing in steady state.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Plan {
    /// Cached tiles to draw, finest level first. A backend gives each pixel
    /// to the first tile that covers it (the `gl` backend by stencil), so a
    /// loaded tile keeps its area and an ancestor standing in around it fills
    /// only the rest; neither draws over the other.
    pub draw: Vec<TileId>,
    /// Needed tiles neither cached nor already being fetched, most important
    /// first. Planning marks them pending.
    pub fetch: Vec<TileId>,
}

struct Entry<V> {
    value: V,
    bytes: usize,
    /// The tile's place in `TileCache::order`.
    tick: u64,
    /// The frame that last used it.
    frame: u64,
}

/// See the module docs.
pub struct TileCache<V> {
    budget: usize,
    used: usize,
    frame: u64,
    tick: u64,
    entries: HashMap<TileKey, Entry<V>>,
    /// Every entry by its last use, oldest first.
    order: BTreeMap<u64, TileKey>,
    pending: HashSet<TileKey>,
}

impl<V> TileCache<V> {
    /// An empty cache that holds up to `budget` bytes, as counted by the
    /// sizes passed to [`TileCache::insert`].
    pub fn new(budget: usize) -> TileCache<V> {
        TileCache {
            budget,
            used: 0,
            frame: 0,
            tick: 0,
            entries: HashMap::new(),
            order: BTreeMap::new(),
            pending: HashSet::new(),
        }
    }

    /// Starts a frame. The tiles the previous frame drew become evictable
    /// again, unless this frame plans them too.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
    }

    /// Splits `needed` (one source's tiles, as a selector lists them) into
    /// what to draw now and what to fetch, writing both into `plan`. `grid`
    /// is the source's grid, which says which tiles are a needed tile's
    /// ancestors and children.
    ///
    /// Every tile planned for drawing is marked used, which protects it from
    /// eviction for the rest of the frame.
    pub fn plan(&mut self, source: u32, grid: &TileGrid, needed: &[TileId], plan: &mut Plan) {
        plan.draw.clear();
        plan.fetch.clear();
        for &tile in needed {
            let key = TileKey { source, tile };
            if self.entries.contains_key(&key) {
                self.touch(key);
                plan.draw.push(tile);
                continue;
            }
            if self.pending.insert(key) {
                plan.fetch.push(tile);
            }
            self.stand_in(source, grid, tile, &mut plan.draw);
        }
        // Finest first; within a level, any order. A stand-in shared by
        // several needed tiles is drawn once.
        plan.draw.sort_by_key(|t| (std::cmp::Reverse(t.z), t.y, t.x));
        plan.draw.dedup();
    }

    /// Pushes onto `draw` what can stand in for the missing `tile`: its
    /// nearest level of cached ancestors, or else its cached children.
    fn stand_in(&mut self, source: u32, grid: &TileGrid, tile: TileId, draw: &mut Vec<TileId>) {
        let Some(bounds) = grid.bounds(tile) else { return };
        // In a quadtree each level holds one ancestor; a grid whose levels
        // do not nest can give several, each covering part of the tile.
        for z in (0..tile.z).rev() {
            let before = draw.len();
            for parent in grid.tiles_in(z, bounds) {
                let key = TileKey { source, tile: parent };
                if self.entries.contains_key(&key) {
                    self.touch(key);
                    draw.push(parent);
                }
            }
            if draw.len() > before {
                return;
            }
        }
        if tile.z < grid.max_zoom() {
            for child in grid.tiles_in(tile.z + 1, bounds) {
                let key = TileKey { source, tile: child };
                if self.entries.contains_key(&key) {
                    self.touch(key);
                    draw.push(child);
                }
            }
        }
    }

    /// Adds a fetched tile, `bytes` large, and clears its pending mark.
    /// Returns whatever this pushed out to stay within budget, with the
    /// value it replaced if `key` was already cached: the caller frees them.
    ///
    /// A tile that turned out empty is worth inserting too (an empty mesh,
    /// a few bytes), so it is not fetched again.
    pub fn insert(&mut self, key: TileKey, value: V, bytes: usize) -> Vec<(TileKey, V)> {
        self.pending.remove(&key);
        let mut freed = Vec::new();
        if let Some(old) = self.take(key) {
            freed.push((key, old));
        }
        self.tick += 1;
        self.order.insert(self.tick, key);
        self.entries.insert(key, Entry { value, bytes, tick: self.tick, frame: self.frame });
        self.used += bytes;
        self.evict(&mut freed);
        freed
    }

    /// Clears `key`'s pending mark after a fetch that produced nothing to
    /// insert, so a later plan asks for it again. When to retry is the
    /// embedder's call: a tile that will never exist is better inserted
    /// empty.
    pub fn fail(&mut self, key: TileKey) {
        self.pending.remove(&key);
    }

    /// The cached value for `key`, if any. Does not count as a use.
    pub fn get(&self, key: TileKey) -> Option<&V> {
        self.entries.get(&key).map(|e| &e.value)
    }

    /// Removes one tile, returning its value to free.
    pub fn remove(&mut self, key: TileKey) -> Option<V> {
        self.pending.remove(&key);
        self.take(key)
    }

    /// Removes every tile of `source`, as when its data changes or its layer
    /// goes away, and forgets its pending fetches. Returns the values to free.
    pub fn remove_source(&mut self, source: u32) -> Vec<(TileKey, V)> {
        self.pending.retain(|k| k.source != source);
        let keys: Vec<TileKey> = self.entries.keys().filter(|k| k.source == source).copied().collect();
        keys.into_iter().filter_map(|k| self.take(k).map(|v| (k, v))).collect()
    }

    /// Changes the budget, evicting down to it at once. Returns the values to
    /// free.
    pub fn set_budget(&mut self, budget: usize) -> Vec<(TileKey, V)> {
        self.budget = budget;
        let mut freed = Vec::new();
        self.evict(&mut freed);
        freed
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    /// The bytes held, which exceeds the budget only while the current
    /// frame's own tiles do.
    pub fn used(&self) -> usize {
        self.used
    }

    /// How many tiles are cached.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether `key` has been handed out to fetch and not yet inserted or
    /// failed.
    pub fn is_pending(&self, key: TileKey) -> bool {
        self.pending.contains(&key)
    }

    fn touch(&mut self, key: TileKey) {
        let entry = self.entries.get_mut(&key).unwrap();
        self.order.remove(&entry.tick);
        self.tick += 1;
        entry.tick = self.tick;
        entry.frame = self.frame;
        self.order.insert(self.tick, key);
    }

    fn take(&mut self, key: TileKey) -> Option<V> {
        let entry = self.entries.remove(&key)?;
        self.order.remove(&entry.tick);
        self.used -= entry.bytes;
        Some(entry.value)
    }

    /// Evicts the least recently used tiles until within budget, stopping at
    /// the current frame's. Those are all newer than every other tile, since
    /// using a tile moves it to the back, so the first one reached means
    /// nothing older is left.
    fn evict(&mut self, freed: &mut Vec<(TileKey, V)>) {
        while self.used > self.budget {
            let Some((_, &key)) = self.order.first_key_value() else { break };
            if self.entries[&key].frame == self.frame {
                break;
            }
            let value = self.take(key).unwrap();
            freed.push((key, value));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(z: u8, x: u32, y: u32) -> TileId {
        TileId { z, x, y }
    }

    fn k(tile: TileId) -> TileKey {
        TileKey { source: 0, tile }
    }

    fn grid() -> TileGrid {
        TileGrid::web_mercator(256, 10).unwrap()
    }

    #[test]
    fn missing_tiles_are_fetched_once() {
        let g = grid();
        let mut cache: TileCache<&str> = TileCache::new(1000);
        let mut plan = Plan::default();
        let needed = [t(2, 1, 1), t(2, 2, 1)];
        cache.begin_frame();
        cache.plan(0, &g, &needed, &mut plan);
        assert_eq!(plan.fetch, needed);
        assert!(plan.draw.is_empty());
        // Still on its way: not asked for again.
        cache.begin_frame();
        cache.plan(0, &g, &needed, &mut plan);
        assert!(plan.fetch.is_empty());
        // One arrives, one fails: the first draws, the second is asked for again.
        assert!(cache.insert(k(t(2, 1, 1)), "a", 10).is_empty());
        cache.fail(k(t(2, 2, 1)));
        cache.begin_frame();
        cache.plan(0, &g, &needed, &mut plan);
        assert_eq!(plan.draw, [t(2, 1, 1)]);
        assert_eq!(plan.fetch, [t(2, 2, 1)]);
    }

    #[test]
    fn sources_do_not_share_tiles() {
        let g = grid();
        let mut cache: TileCache<&str> = TileCache::new(1000);
        let mut plan = Plan::default();
        cache.insert(k(t(1, 0, 0)), "a", 10);
        cache.plan(7, &g, &[t(1, 0, 0)], &mut plan);
        assert_eq!(plan.fetch, [t(1, 0, 0)]);
        assert!(plan.draw.is_empty());
    }

    #[test]
    fn an_ancestor_stands_in_while_a_level_loads() {
        let g = grid();
        let mut cache: TileCache<&str> = TileCache::new(1000);
        let mut plan = Plan::default();
        cache.insert(k(t(1, 0, 0)), "grandparent", 10);
        cache.insert(k(t(3, 1, 1)), "loaded", 10);
        // Four level-3 tiles inside level 1's (0, 0), one of them loaded.
        let needed = [t(3, 1, 1), t(3, 2, 1), t(3, 1, 2), t(3, 2, 2)];
        cache.plan(0, &g, &needed, &mut plan);
        assert_eq!(plan.fetch, [t(3, 2, 1), t(3, 1, 2), t(3, 2, 2)]);
        // The loaded tile, then the grandparent once around it.
        assert_eq!(plan.draw, [t(3, 1, 1), t(1, 0, 0)]);
        // A nearer ancestor wins over a farther one.
        cache.insert(k(t(2, 1, 1)), "parent", 10);
        cache.plan(0, &g, &[t(3, 2, 2)], &mut plan);
        assert_eq!(plan.draw, [t(2, 1, 1)]);
    }

    #[test]
    fn children_stand_in_when_zooming_out() {
        let g = grid();
        let mut cache: TileCache<&str> = TileCache::new(1000);
        let mut plan = Plan::default();
        cache.insert(k(t(4, 2, 2)), "a", 10);
        cache.insert(k(t(4, 3, 3)), "b", 10);
        cache.insert(k(t(4, 9, 9)), "elsewhere", 10);
        cache.plan(0, &g, &[t(3, 1, 1)], &mut plan);
        assert_eq!(plan.draw, [t(4, 2, 2), t(4, 3, 3)]);
    }

    #[test]
    fn the_least_recently_used_go_first() {
        let g = grid();
        let mut cache: TileCache<u32> = TileCache::new(30);
        let mut plan = Plan::default();
        cache.begin_frame();
        cache.insert(k(t(5, 0, 0)), 0, 10);
        cache.insert(k(t(5, 1, 0)), 1, 10);
        cache.insert(k(t(5, 2, 0)), 2, 10);
        // A new frame uses the oldest, so the second is now least recent.
        cache.begin_frame();
        cache.plan(0, &g, &[t(5, 0, 0)], &mut plan);
        cache.begin_frame();
        let freed = cache.insert(k(t(5, 3, 0)), 3, 10);
        assert_eq!(freed, [(k(t(5, 1, 0)), 1)]);
        assert_eq!(cache.used(), 30);
        assert_eq!(cache.len(), 3);
        assert!(cache.get(k(t(5, 0, 0))).is_some());
    }

    #[test]
    fn the_current_frame_is_never_evicted() {
        let g = grid();
        let mut cache: TileCache<u32> = TileCache::new(25);
        let mut plan = Plan::default();
        cache.begin_frame();
        cache.insert(k(t(5, 0, 0)), 0, 10);
        cache.insert(k(t(5, 1, 0)), 1, 10);
        cache.begin_frame();
        cache.plan(0, &g, &[t(5, 0, 0), t(5, 1, 0)], &mut plan);
        // Over budget, but both are on screen: the cache runs over.
        assert!(cache.insert(k(t(5, 2, 0)), 2, 10).is_empty());
        assert_eq!(cache.used(), 30);
        // Next frame they are fair game, oldest first.
        cache.begin_frame();
        let freed = cache.set_budget(20);
        assert_eq!(freed, [(k(t(5, 0, 0)), 0)]);
        assert_eq!(cache.used(), 20);
    }

    #[test]
    fn replacing_and_removing_hand_values_back() {
        let mut cache: TileCache<u32> = TileCache::new(100);
        cache.insert(k(t(1, 0, 0)), 1, 10);
        assert_eq!(cache.insert(k(t(1, 0, 0)), 2, 15), [(k(t(1, 0, 0)), 1)]);
        assert_eq!(cache.used(), 15);
        cache.insert(TileKey { source: 3, tile: t(1, 1, 0) }, 3, 5);
        cache.insert(TileKey { source: 3, tile: t(1, 1, 1) }, 4, 5);
        let mut gone = cache.remove_source(3);
        gone.sort();
        assert_eq!(gone.iter().map(|g| g.1).collect::<Vec<_>>(), [3, 4]);
        assert_eq!(cache.remove(k(t(1, 0, 0))), Some(2));
        assert!(cache.is_empty());
        assert_eq!(cache.used(), 0);
    }

    #[test]
    fn removing_a_source_forgets_its_fetches() {
        let g = grid();
        let mut cache: TileCache<u32> = TileCache::new(100);
        let mut plan = Plan::default();
        cache.plan(2, &g, &[t(1, 0, 0)], &mut plan);
        assert!(cache.is_pending(TileKey { source: 2, tile: t(1, 0, 0) }));
        cache.remove_source(2);
        cache.plan(2, &g, &[t(1, 0, 0)], &mut plan);
        assert_eq!(plan.fetch, [t(1, 0, 0)]);
    }
}
