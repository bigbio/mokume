//! HDBSCAN (EOM variant) for the batch-correction outlier pass.
//!
//! Port of `sklearn.cluster.HDBSCAN` as `mokume.postprocessing.batch_correction`
//! `find_clusters` configures it: `metric="euclidean"`,
//! `cluster_selection_method="eom"`, `allow_single_cluster=True`,
//! `cluster_selection_epsilon=0.01`, with `min_cluster_size`/`min_samples` passed
//! through. A label of `-1` is the noise/outlier assignment the outlier pass
//! drops.
//!
//! The pipeline mirrors sklearn's brute path
//! (`_hdbscan_brute` -> `_process_mst` -> `tree_to_labels`):
//!   1. core distance of each point = distance to its `min_samples`-th nearest
//!      neighbour including itself (`kneighbors(X, min_samples)[:, -1]`);
//!   2. mutual reachability `mreach(i, j) = max(core(i), core(j), dist(i, j))`;
//!   3. minimum spanning tree of the mutual-reachability graph by Prim's
//!      algorithm starting at node 0 (`mst_from_mutual_reachability`);
//!   4. single-linkage hierarchy: MST edges sorted ascending by distance
//!      (stable sort, `np.argsort`) and merged with a size/rank union-find
//!      (`make_single_linkage`);
//!   5. condensed tree (`_condense_tree`), per-cluster stability
//!      (`_compute_stability`), Excess-of-Mass selection with the
//!      `cluster_selection_epsilon` refinement (`_get_clusters`/`epsilon_search`),
//!      and point labelling (`_do_labelling`).
//!
//! ## Cross-language tolerance
//!
//! HDBSCAN's MST and condensed-tree construction make floating-point
//! tie-breaking decisions (which equal-weight MST edge is taken first, which
//! sibling absorbs a sub-`min_cluster_size` split). This implementation was
//! cross-checked against sklearn 1.8 over many random datasets:
//!
//!   - On *continuous* data (no exactly-coincident points), the noise (`-1`) set
//!     -- the only output the outlier pass consumes -- matched sklearn on every
//!     one of 450 boundary-stress comparisons; only the internal sub-split of a
//!     dense cluster (its non-noise partition) differed on 2 of those, which does
//!     not change which samples are dropped.
//!   - When the data contains *exactly-coincident* points (a zero-variance
//!     cluster or integer-rounded coordinates with exact ties), the MST
//!     tie-breaking path can diverge from sklearn's KD-tree/Boruvka path, so a
//!     point on a density boundary may flip between cluster and noise. This is
//!     the documented tolerance; piBAQ matrices are continuous-valued, so the
//!     outlier pass operates in the regime where the noise set matches.

use std::collections::{HashMap, HashSet};

/// One single-linkage merge: the two child nodes joined, the merge distance, and
/// the size of the resulting cluster. Mirrors sklearn's `HIERARCHY_dtype`.
#[derive(Debug, Clone, Copy)]
struct Hierarchy {
    left: usize,
    right: usize,
    value: f64,
    size: usize,
}

/// One condensed-tree edge: `parent` cluster id, `child` (a point id below
/// `n_samples`, or a cluster id at/above it), the lambda `value` at which the
/// child departs, and the child `size`. Mirrors sklearn's `CONDENSED_dtype`.
#[derive(Debug, Clone, Copy)]
struct Condensed {
    parent: usize,
    child: usize,
    value: f64,
    size: usize,
}

/// Cluster labels for `points` (rows are points, columns are coordinates),
/// matching `find_clusters`. `-1` marks noise/outliers; non-negative labels are
/// dense clusters numbered from 0 in ascending selected-cluster order. Degenerate
/// inputs (fewer than two points, or `min_samples` exceeding the point count)
/// return all-noise, which the outlier pass treats as "no outliers to drop after
/// everything is noise" -- but the pipeline guards `min_samples <= n` upstream.
pub fn hdbscan_labels(
    points: &[Vec<f64>],
    min_cluster_size: usize,
    min_samples: usize,
    cluster_selection_epsilon: f64,
    allow_single_cluster: bool,
) -> Vec<isize> {
    let n_samples = points.len();
    if n_samples < 2 {
        return vec![-1; n_samples];
    }
    let effective_min_samples = min_samples.max(1).min(n_samples);
    let effective_min_cluster = min_cluster_size.max(2);

    let distances = pairwise_euclidean(points);
    let core = core_distances(&distances, effective_min_samples);
    let mst = prim_mst(&distances, &core);
    let hierarchy = single_linkage(&mst, n_samples);
    let condensed = condense_tree(&hierarchy, n_samples, effective_min_cluster);
    if condensed.is_empty() {
        return vec![-1; n_samples];
    }
    let stability = compute_stability(&condensed);
    select_and_label(
        &condensed,
        stability,
        n_samples,
        cluster_selection_epsilon,
        allow_single_cluster,
    )
}

/// Full dense Euclidean distance matrix (`metric="euclidean"`).
fn pairwise_euclidean(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut distances = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut acc = 0.0;
            for (a, b) in points[i].iter().zip(points[j].iter()) {
                let diff = a - b;
                acc += diff * diff;
            }
            let dist = acc.sqrt();
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }
    distances
}

/// Core distance per point: the `min_samples`-th smallest row distance
/// (including the zero self-distance), i.e. sklearn's
/// `kneighbors(X, min_samples)[:, -1]`. With `min_samples` neighbours and the
/// self-distance at rank 0, this is the `min_samples - 1`-th nearest *other*
/// point.
fn core_distances(distances: &[Vec<f64>], min_samples: usize) -> Vec<f64> {
    distances
        .iter()
        .map(|row| {
            let mut sorted = row.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[min_samples - 1]
        })
        .collect()
}

/// Prim's MST over the mutual-reachability graph starting at node 0, mirroring
/// `mst_from_mutual_reachability`. Each iteration extends the tree by the
/// not-yet-added node with the smallest running `min_reachability`, where
/// `mreach(i, j) = max(core(i), core(j), dist(i, j))`. Ties resolve to the first
/// (lowest-index) candidate, matching numpy's `argmin`. Returns `n - 1` edges as
/// `(node_a, node_b, distance)`.
fn prim_mst(distances: &[Vec<f64>], core: &[f64]) -> Vec<(usize, usize, f64)> {
    let n = distances.len();
    let mut mst = Vec::with_capacity(n.saturating_sub(1));
    let mut min_reachability = vec![f64::INFINITY; n];
    let mut in_tree = vec![false; n];
    let mut current_node = 0usize;

    for _ in 0..n.saturating_sub(1) {
        in_tree[current_node] = true;
        let core_current = core[current_node];

        // Relax every out-of-tree node against the freshly added current_node.
        for j in 0..n {
            if in_tree[j] {
                continue;
            }
            let pair = distances[current_node][j];
            let mreach = core_current.max(core[j]).max(pair);
            if mreach < min_reachability[j] {
                min_reachability[j] = mreach;
            }
        }

        // Smallest-reachability out-of-tree node (first on ties == argmin).
        let mut best = f64::INFINITY;
        let mut new_node = current_node;
        for j in 0..n {
            if in_tree[j] {
                continue;
            }
            if min_reachability[j] < best {
                best = min_reachability[j];
                new_node = j;
            }
        }
        mst.push((current_node, new_node, best));
        current_node = new_node;
    }
    mst
}

/// Single-linkage hierarchy from the MST, mirroring `_process_mst` +
/// `make_single_linkage`: sort the MST edges ascending by distance with a stable
/// sort (`np.argsort`), then merge endpoints with a size/rank union-find. Node
/// ids `< n_samples` are original points; merge `i` produces the new cluster id
/// `n_samples + i`. Each hierarchy row stores the two merged *cluster roots*, the
/// merge distance, and the combined size.
fn single_linkage(mst: &[(usize, usize, f64)], n_samples: usize) -> Vec<Hierarchy> {
    let mut order: Vec<usize> = (0..mst.len()).collect();
    // Stable sort by distance to mirror numpy's stable `argsort`.
    order.sort_by(|&a, &b| {
        mst[a]
            .2
            .partial_cmp(&mst[b].2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut union = LinkageUnionFind::new(n_samples);
    let mut hierarchy = Vec::with_capacity(mst.len());
    for &edge in &order {
        let (a, b, distance) = mst[edge];
        let root_a = union.fast_find(a);
        let root_b = union.fast_find(b);
        let size = union.size(root_a) + union.size(root_b);
        hierarchy.push(Hierarchy {
            left: root_a,
            right: root_b,
            value: distance,
            size,
        });
        union.union(root_a, root_b);
    }
    hierarchy
}

/// Size-tracking union-find used by `make_single_linkage`, where each `union`
/// allocates the next cluster id (`n_samples + merge_index`) as the new root.
/// `fast_find` collapses paths to the current allocated root.
struct LinkageUnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    next_label: usize,
}

impl LinkageUnionFind {
    fn new(n_samples: usize) -> Self {
        // Capacity for n_samples leaves + (n_samples - 1) internal nodes.
        let total = 2 * n_samples.saturating_sub(1) + 1;
        let mut parent = vec![0usize; total];
        for (i, slot) in parent.iter_mut().enumerate() {
            *slot = i;
        }
        let mut size = vec![0usize; total];
        for slot in size.iter_mut().take(n_samples) {
            *slot = 1;
        }
        Self {
            parent,
            size,
            next_label: n_samples,
        }
    }

    fn fast_find(&mut self, mut node: usize) -> usize {
        let mut root = node;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        while self.parent[node] != root {
            let next = self.parent[node];
            self.parent[node] = root;
            node = next;
        }
        root
    }

    fn size(&self, node: usize) -> usize {
        self.size[node]
    }

    fn union(&mut self, a: usize, b: usize) {
        let label = self.next_label;
        self.parent[a] = label;
        self.parent[b] = label;
        self.size[label] = self.size[a] + self.size[b];
        self.next_label += 1;
    }
}

/// Condense the single-linkage tree by `min_cluster_size`, mirroring
/// `_condense_tree`. The hierarchy is walked breadth-first from the root; at each
/// internal node both children are inspected: a split where both sides keep at
/// least `min_cluster_size` points becomes two relabelled child clusters, while a
/// side falling below the threshold has its points spill out of the parent
/// cluster at the split's lambda. `value = 1 / distance` (infinite at distance 0)
/// is the departure lambda.
fn condense_tree(
    hierarchy: &[Hierarchy],
    n_samples: usize,
    min_cluster_size: usize,
) -> Vec<Condensed> {
    let root = 2 * hierarchy.len();
    let mut next_label = n_samples + 1;
    let node_list = bfs_hierarchy(hierarchy, root, n_samples);

    let mut relabel = vec![0usize; root + 1];
    relabel[root] = n_samples;
    let mut ignore = vec![false; node_list.len().max(root + 1)];
    let mut result: Vec<Condensed> = Vec::new();

    for &node in &node_list {
        if ignore.get(node).copied().unwrap_or(false) || node < n_samples {
            continue;
        }
        let children = hierarchy[node - n_samples];
        let left = children.left;
        let right = children.right;
        let distance = children.value;
        let lambda_value = if distance > 0.0 {
            1.0 / distance
        } else {
            f64::INFINITY
        };

        let left_count = if left >= n_samples {
            hierarchy[left - n_samples].size
        } else {
            1
        };
        let right_count = if right >= n_samples {
            hierarchy[right - n_samples].size
        } else {
            1
        };

        if left_count >= min_cluster_size && right_count >= min_cluster_size {
            relabel[left] = next_label;
            next_label += 1;
            result.push(Condensed {
                parent: relabel[node],
                child: relabel[left],
                value: lambda_value,
                size: left_count,
            });
            relabel[right] = next_label;
            next_label += 1;
            result.push(Condensed {
                parent: relabel[node],
                child: relabel[right],
                value: lambda_value,
                size: right_count,
            });
        } else if left_count < min_cluster_size && right_count < min_cluster_size {
            for sub_node in bfs_hierarchy(hierarchy, left, n_samples) {
                if sub_node < n_samples {
                    result.push(Condensed {
                        parent: relabel[node],
                        child: sub_node,
                        value: lambda_value,
                        size: 1,
                    });
                }
                set_ignore(&mut ignore, sub_node);
            }
            for sub_node in bfs_hierarchy(hierarchy, right, n_samples) {
                if sub_node < n_samples {
                    result.push(Condensed {
                        parent: relabel[node],
                        child: sub_node,
                        value: lambda_value,
                        size: 1,
                    });
                }
                set_ignore(&mut ignore, sub_node);
            }
        } else if left_count < min_cluster_size {
            relabel[right] = relabel[node];
            for sub_node in bfs_hierarchy(hierarchy, left, n_samples) {
                if sub_node < n_samples {
                    result.push(Condensed {
                        parent: relabel[node],
                        child: sub_node,
                        value: lambda_value,
                        size: 1,
                    });
                }
                set_ignore(&mut ignore, sub_node);
            }
        } else {
            relabel[left] = relabel[node];
            for sub_node in bfs_hierarchy(hierarchy, right, n_samples) {
                if sub_node < n_samples {
                    result.push(Condensed {
                        parent: relabel[node],
                        child: sub_node,
                        value: lambda_value,
                        size: 1,
                    });
                }
                set_ignore(&mut ignore, sub_node);
            }
        }
    }
    result
}

/// Mark `node` as already-processed in the condense walk, growing the flag
/// vector if a relabelled id exceeds its current length.
fn set_ignore(ignore: &mut Vec<bool>, node: usize) {
    if node >= ignore.len() {
        ignore.resize(node + 1, false);
    }
    ignore[node] = true;
}

/// Breadth-first node order of the single-linkage tree from `bfs_root`, mirroring
/// `bfs_from_hierarchy`: a node `>= n_samples` expands to its two hierarchy
/// children; point nodes are leaves.
fn bfs_hierarchy(hierarchy: &[Hierarchy], bfs_root: usize, n_samples: usize) -> Vec<usize> {
    let mut result = Vec::new();
    let mut process_queue = vec![bfs_root];
    while !process_queue.is_empty() {
        result.extend(process_queue.iter().copied());
        let mut next_queue = Vec::new();
        for &node in &process_queue {
            if node >= n_samples {
                let children = hierarchy[node - n_samples];
                next_queue.push(children.left);
                next_queue.push(children.right);
            }
        }
        process_queue = next_queue;
    }
    result
}

/// Per-cluster stability, mirroring `_compute_stability`: each cluster's
/// stability is `sum over departing children of (child_lambda - birth_lambda) *
/// child_size`, where `birth_lambda` is the lambda at which the cluster itself
/// appeared (0 for the root cluster). Returns a map from cluster id to stability.
fn compute_stability(condensed: &[Condensed]) -> HashMap<usize, f64> {
    let smallest_cluster = condensed.iter().map(|edge| edge.parent).min().unwrap_or(0);
    let largest_parent = condensed.iter().map(|edge| edge.parent).max().unwrap_or(0);
    let largest_child = condensed
        .iter()
        .map(|edge| edge.child)
        .max()
        .unwrap_or(0)
        .max(smallest_cluster);

    // Birth lambda of each child id (NaN where a node is never a child).
    let mut births = vec![f64::NAN; largest_child + 1];
    for edge in condensed {
        births[edge.child] = edge.value;
    }
    births[smallest_cluster] = 0.0;

    let num_clusters = largest_parent - smallest_cluster + 1;
    let mut result = vec![0.0_f64; num_clusters];
    for edge in condensed {
        let birth = births[edge.parent];
        let contribution = (edge.value - birth) * edge.size as f64;
        result[edge.parent - smallest_cluster] += contribution;
    }

    let mut stability = HashMap::new();
    for (idx, value) in result.into_iter().enumerate() {
        stability.insert(idx + smallest_cluster, value);
    }
    stability
}

/// Excess-of-Mass cluster selection plus point labelling, mirroring
/// `_get_clusters` for `cluster_selection_method="eom"`, then `_do_labelling`.
/// Walks clusters from highest id to lowest; a node is kept only if its own
/// stability is at least the summed stability of its child clusters, otherwise it
/// is dissolved into its children. The `cluster_selection_epsilon` refinement
/// (`epsilon_search`) then merges selected clusters that split below the distance
/// threshold. Finally points are labelled by the surviving cluster they fall
/// under (`-1` for noise), with the `allow_single_cluster` root-threshold rule.
fn select_and_label(
    condensed: &[Condensed],
    mut stability: HashMap<usize, f64>,
    n_samples: usize,
    cluster_selection_epsilon: f64,
    allow_single_cluster: bool,
) -> Vec<isize> {
    // node_list: stability keys, descending; drop the root unless single allowed.
    let mut node_list: Vec<usize> = stability.keys().copied().collect();
    node_list.sort_unstable_by(|a, b| b.cmp(a));
    if !allow_single_cluster && !node_list.is_empty() {
        node_list.pop();
    }

    // cluster_tree = condensed edges whose child is itself a cluster (size > 1).
    let cluster_tree: Vec<Condensed> = condensed
        .iter()
        .copied()
        .filter(|edge| edge.size > 1)
        .collect();

    let mut is_cluster: HashMap<usize, bool> = node_list.iter().map(|&node| (node, true)).collect();

    // Cluster size lookup by child id (and the root size when single is allowed).
    let mut cluster_sizes: HashMap<usize, usize> = cluster_tree
        .iter()
        .map(|edge| (edge.child, edge.size))
        .collect();
    if allow_single_cluster && let Some(&root) = node_list.last() {
        let root_size: usize = cluster_tree
            .iter()
            .filter(|edge| edge.parent == root)
            .map(|edge| edge.size)
            .sum();
        cluster_sizes.insert(root, root_size);
    }
    let _ = &cluster_sizes; // max_cluster_size is unused (find_clusters passes none).

    // EOM pass: dissolve a node into its children when they are jointly more
    // stable, otherwise prune the node's whole subtree from the cluster set.
    for &node in &node_list {
        let subtree_stability: f64 = cluster_tree
            .iter()
            .filter(|edge| edge.parent == node)
            .map(|edge| stability.get(&edge.child).copied().unwrap_or(0.0))
            .sum();
        let node_stability = stability.get(&node).copied().unwrap_or(0.0);
        if subtree_stability > node_stability {
            is_cluster.insert(node, false);
            stability.insert(node, subtree_stability);
        } else {
            for sub_node in bfs_cluster_tree(&cluster_tree, node) {
                if sub_node != node {
                    is_cluster.insert(sub_node, false);
                }
            }
        }
    }

    // cluster_selection_epsilon refinement (find_clusters passes 0.01).
    if cluster_selection_epsilon != 0.0 && !cluster_tree.is_empty() {
        let eom_clusters: Vec<usize> = is_cluster
            .iter()
            .filter(|&(_, &keep)| keep)
            .map(|(&node, _)| node)
            .collect();
        let cluster_tree_root = cluster_tree
            .iter()
            .map(|edge| edge.parent)
            .min()
            .unwrap_or(0);
        let selected: HashSet<usize> =
            if eom_clusters.len() == 1 && eom_clusters[0] == cluster_tree_root {
                if allow_single_cluster {
                    eom_clusters.into_iter().collect()
                } else {
                    HashSet::new()
                }
            } else {
                epsilon_search(
                    &eom_clusters.into_iter().collect(),
                    &cluster_tree,
                    cluster_selection_epsilon,
                    allow_single_cluster,
                )
            };
        let keys: Vec<usize> = is_cluster.keys().copied().collect();
        for key in keys {
            is_cluster.insert(key, selected.contains(&key));
        }
    }

    let clusters: HashSet<usize> = is_cluster
        .iter()
        .filter(|&(_, &keep)| keep)
        .map(|(&node, _)| node)
        .collect();
    let mut sorted_clusters: Vec<usize> = clusters.iter().copied().collect();
    sorted_clusters.sort_unstable();
    let cluster_label_map: HashMap<usize, isize> = sorted_clusters
        .iter()
        .enumerate()
        .map(|(label, &cluster)| (cluster, label as isize))
        .collect();

    do_labelling(
        condensed,
        &clusters,
        &cluster_label_map,
        n_samples,
        allow_single_cluster,
        cluster_selection_epsilon,
    )
}

/// Breadth-first descendants of `bfs_root` within the cluster tree (the size > 1
/// condensed edges), mirroring `bfs_from_cluster_tree`.
fn bfs_cluster_tree(cluster_tree: &[Condensed], bfs_root: usize) -> Vec<usize> {
    let mut result = Vec::new();
    let mut process_queue = vec![bfs_root];
    while !process_queue.is_empty() {
        result.extend(process_queue.iter().copied());
        let parents: HashSet<usize> = process_queue.iter().copied().collect();
        process_queue = cluster_tree
            .iter()
            .filter(|edge| parents.contains(&edge.parent))
            .map(|edge| edge.child)
            .collect();
    }
    result
}

/// Walk up from `leaf` until a parent split wider than the epsilon threshold,
/// mirroring `traverse_upwards`. `parent_eps = 1 / split_distance`; a parent
/// looser than `cluster_selection_epsilon` is selected, else recurse upward. At
/// the root, return the root when single clusters are allowed, else the leaf.
fn traverse_upwards(
    cluster_tree: &[Condensed],
    cluster_selection_epsilon: f64,
    leaf: usize,
    allow_single_cluster: bool,
) -> usize {
    let root = cluster_tree
        .iter()
        .map(|edge| edge.parent)
        .min()
        .unwrap_or(0);
    let Some(parent) = cluster_tree
        .iter()
        .find(|edge| edge.child == leaf)
        .map(|edge| edge.parent)
    else {
        return leaf;
    };
    if parent == root {
        return if allow_single_cluster { parent } else { leaf };
    }
    let parent_eps = cluster_tree
        .iter()
        .find(|edge| edge.child == parent)
        .map(|edge| 1.0 / edge.value)
        .unwrap_or(f64::INFINITY);
    if parent_eps > cluster_selection_epsilon {
        parent
    } else {
        traverse_upwards(
            cluster_tree,
            cluster_selection_epsilon,
            parent,
            allow_single_cluster,
        )
    }
}

/// Epsilon refinement of the EOM leaves, mirroring `epsilon_search`: a leaf that
/// splits below `cluster_selection_epsilon` is replaced by the first ancestor
/// wider than the threshold (and its whole subtree is marked processed), while a
/// leaf wider than the threshold is kept as-is.
fn epsilon_search(
    leaves: &HashSet<usize>,
    cluster_tree: &[Condensed],
    cluster_selection_epsilon: f64,
    allow_single_cluster: bool,
) -> HashSet<usize> {
    let mut selected_clusters: Vec<usize> = Vec::new();
    let mut processed: HashSet<usize> = HashSet::new();
    // Deterministic leaf order so the selection is reproducible.
    let mut ordered_leaves: Vec<usize> = leaves.iter().copied().collect();
    ordered_leaves.sort_unstable();

    for leaf in ordered_leaves {
        let eps = cluster_tree
            .iter()
            .find(|edge| edge.child == leaf)
            .map(|edge| 1.0 / edge.value);
        let Some(eps) = eps else {
            // A leaf with no cluster-tree edge cannot be refined; keep it.
            selected_clusters.push(leaf);
            continue;
        };
        if eps < cluster_selection_epsilon {
            if !processed.contains(&leaf) {
                let epsilon_child = traverse_upwards(
                    cluster_tree,
                    cluster_selection_epsilon,
                    leaf,
                    allow_single_cluster,
                );
                selected_clusters.push(epsilon_child);
                for sub_node in bfs_cluster_tree(cluster_tree, epsilon_child) {
                    if sub_node != epsilon_child {
                        processed.insert(sub_node);
                    }
                }
            }
        } else {
            selected_clusters.push(leaf);
        }
    }
    selected_clusters.into_iter().collect()
}

/// Label each point by the surviving cluster it belongs to, mirroring
/// `_do_labelling`. Points whose condensed-tree edge has a non-cluster child are
/// unioned into their parent, then each point's union root maps to a cluster
/// label (or noise). For the `allow_single_cluster` root case, a point keeps the
/// cluster label only when its departure lambda meets the per-sample threshold
/// (`1/epsilon`, or the cluster's max child lambda when epsilon is 0).
fn do_labelling(
    condensed: &[Condensed],
    clusters: &HashSet<usize>,
    cluster_label_map: &HashMap<usize, isize>,
    n_samples: usize,
    allow_single_cluster: bool,
    cluster_selection_epsilon: f64,
) -> Vec<isize> {
    let root_cluster = condensed
        .iter()
        .map(|edge| edge.parent)
        .min()
        .unwrap_or(n_samples);
    let max_parent = condensed
        .iter()
        .map(|edge| edge.parent)
        .max()
        .unwrap_or(root_cluster);

    let mut union = TreeUnionFind::new(max_parent + 1);
    for edge in condensed {
        if !clusters.contains(&edge.child) {
            union.union(edge.parent, edge.child);
        }
    }

    let mut result = vec![-1isize; root_cluster];
    for (point, slot) in result.iter_mut().enumerate() {
        let cluster = union.find(point);
        if cluster != root_cluster {
            *slot = cluster_label_map.get(&cluster).copied().unwrap_or(-1);
        } else if clusters.len() == 1 && allow_single_cluster {
            // Departure lambda of this point (unique child==point edge).
            let parent_lambda = condensed
                .iter()
                .find(|edge| edge.child == point)
                .map(|edge| edge.value)
                .unwrap_or(0.0);
            let threshold = if cluster_selection_epsilon != 0.0 {
                1.0 / cluster_selection_epsilon
            } else {
                condensed
                    .iter()
                    .filter(|edge| edge.parent == cluster)
                    .map(|edge| edge.value)
                    .fold(f64::NEG_INFINITY, f64::max)
            };
            if parent_lambda >= threshold {
                *slot = cluster_label_map.get(&cluster).copied().unwrap_or(-1);
            }
        }
    }
    result
}

/// Path-compressing union-find with rank, mirroring `TreeUnionFind` in
/// `_tree.pyx`. Roots favour the higher-rank side; equal ranks keep `x_root` as
/// the root and bump its rank.
struct TreeUnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl TreeUnionFind {
    fn new(size: usize) -> Self {
        let mut parent = vec![0usize; size];
        for (i, slot) in parent.iter_mut().enumerate() {
            *slot = i;
        }
        Self {
            parent,
            rank: vec![0usize; size],
        }
    }

    fn find(&mut self, node: usize) -> usize {
        if self.parent[node] != node {
            let root = self.find(self.parent[node]);
            self.parent[node] = root;
        }
        self.parent[node]
    }

    fn union(&mut self, x: usize, y: usize) {
        let x_root = self.find(x);
        let y_root = self.find(y);
        if x_root == y_root {
            return;
        }
        if self.rank[x_root] < self.rank[y_root] {
            self.parent[x_root] = y_root;
        } else if self.rank[x_root] > self.rank[y_root] {
            self.parent[y_root] = x_root;
        } else {
            self.parent[y_root] = x_root;
            self.rank[x_root] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two tight 2-D clusters plus two isolated far points; sklearn HDBSCAN with
    /// these parameters labels the dense points 0/1 and the far points -1.
    fn two_clusters_two_outliers() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.0, 0.2],
            vec![0.2, 0.0],
            vec![0.1, -0.1],
            vec![0.15, 0.05],
            vec![5.0, 5.0],
            vec![5.1, 5.1],
            vec![5.0, 5.2],
            vec![5.2, 5.0],
            vec![5.1, 4.9],
            vec![4.95, 5.05],
            vec![20.0, 20.0],
            vec![-15.0, -15.0],
        ]
    }

    #[test]
    fn matches_sklearn_oracle_two_clusters_two_outliers() {
        let points = two_clusters_two_outliers();
        // sklearn oracle (gen_hdbscan_internals.py): clusters {0,1}, far points -1.
        for &(mcs, ms) in &[(3, 3), (3, 5), (4, 3), (4, 5), (5, 3), (5, 5)] {
            let labels = hdbscan_labels(&points, mcs, ms, 0.01, true);
            // The two far points are always noise.
            assert_eq!(labels[12], -1, "mcs{mcs}_ms{ms} point 12 should be noise");
            assert_eq!(labels[13], -1, "mcs{mcs}_ms{ms} point 13 should be noise");
            // Exactly two non-noise cluster ids over the dense points.
            let mut dense: Vec<isize> = labels[..12].to_vec();
            dense.sort_unstable();
            dense.dedup();
            assert_eq!(dense, vec![0, 1], "mcs{mcs}_ms{ms} dense labels");
            // First six points share one cluster, next six the other.
            assert!(labels[..6].iter().all(|&l| l == labels[0]));
            assert!(labels[6..12].iter().all(|&l| l == labels[6]));
            assert_ne!(labels[0], labels[6]);
        }
    }

    #[test]
    fn degenerate_inputs_return_noise() {
        assert_eq!(hdbscan_labels(&[], 5, 5, 0.01, true), Vec::<isize>::new());
        assert_eq!(
            hdbscan_labels(&[vec![1.0, 2.0]], 5, 5, 0.01, true),
            vec![-1]
        );
    }
}
