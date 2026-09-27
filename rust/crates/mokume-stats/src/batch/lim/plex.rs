//! TMT plex groups for LIM's plex block `P[k,g]`.
//!
//! Preferred source: an explicit plex / mixture id per profile (from the SDRF),
//! passed by the caller. When none is given, plexes are inferred as in the
//! prototype (`plex_groups` in `pli.py` / `lim.py`): profiles of one TMT plex
//! share the plex's missingness pattern, so profiles are clustered by average
//! linkage (UPGMA) on the Jaccard distance of their observed-gene sets and the
//! dendrogram is cut at distance `t` (SciPy `fcluster(..., t, "distance")`).
//! Clusters smaller than `min_size` are left unassigned.

/// Label returned for profiles that belong to no plex.
pub const NO_PLEX: i64 = -1;

/// Cluster the rows of a boolean observed mask (`rows x genes`, row-major) by
/// shared missingness. Returns one label per row: `0..k` for clustered rows
/// (numbered by first occurrence) and [`NO_PLEX`] for rows in clusters smaller
/// than `min_size`.
pub fn jaccard_plex_groups(observed: &[Vec<bool>], t: f64, min_size: usize) -> Vec<i64> {
    let n = observed.len();
    if n == 0 {
        return Vec::new();
    }
    let bits: Vec<Vec<u64>> = observed.iter().map(|row| pack(row)).collect();
    let counts: Vec<u32> = bits
        .iter()
        .map(|b| b.iter().map(|w| w.count_ones()).sum())
        .collect();

    // Pairwise Jaccard distances (condensed as a full symmetric matrix).
    let mut dist = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let inter: u32 = bits[i]
                .iter()
                .zip(&bits[j])
                .map(|(a, b)| (a & b).count_ones())
                .sum();
            let union = counts[i] + counts[j] - inter;
            let jac = if union == 0 {
                0.0
            } else {
                f64::from(inter) / f64::from(union)
            };
            let d = (1.0 - jac).clamp(0.0, 1.0);
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }

    // UPGMA: repeatedly merge the closest pair while its distance <= t
    // (average linkage is monotone, so this equals cutting the full tree at t).
    let mut members: Vec<Option<Vec<usize>>> = (0..n).map(|i| Some(vec![i])).collect();
    let mut active: Vec<usize> = (0..n).collect();
    loop {
        let mut best: Option<(f64, usize, usize)> = None;
        for (ai, &a) in active.iter().enumerate() {
            for &b in &active[(ai + 1)..] {
                let d = dist[a * n + b];
                if best.is_none_or(|(bd, _, _)| d < bd) {
                    best = Some((d, a, b));
                }
            }
        }
        let Some((d, a, b)) = best else { break };
        if d > t {
            break;
        }
        let size_a = members[a].as_ref().map_or(0, Vec::len) as f64;
        let size_b = members[b].as_ref().map_or(0, Vec::len) as f64;
        for &c in &active {
            if c == a || c == b {
                continue;
            }
            let merged = (size_a * dist[a * n + c] + size_b * dist[b * n + c]) / (size_a + size_b);
            dist[a * n + c] = merged;
            dist[c * n + a] = merged;
        }
        let moved = members[b].take().unwrap_or_default();
        if let Some(m) = members[a].as_mut() {
            m.extend(moved);
        }
        active.retain(|&c| c != b);
    }

    let mut label_of = vec![NO_PLEX; n];
    let mut clusters: Vec<Vec<usize>> = members
        .into_iter()
        .flatten()
        .filter(|m| m.len() >= min_size.max(1))
        .map(|mut m| {
            m.sort_unstable();
            m
        })
        .collect();
    clusters.sort_by_key(|m| m[0]);
    for (k, m) in clusters.iter().enumerate() {
        for &i in m {
            label_of[i] = k as i64;
        }
    }
    label_of
}

fn pack(row: &[bool]) -> Vec<u64> {
    let mut out = vec![0_u64; row.len().div_ceil(64)];
    for (g, &o) in row.iter().enumerate() {
        if o {
            out[g / 64] |= 1 << (g % 64);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{jaccard_plex_groups, NO_PLEX};

    fn row(genes: &[usize], g: usize) -> Vec<bool> {
        (0..g).map(|i| genes.contains(&i)).collect()
    }

    #[test]
    fn profiles_sharing_missingness_cluster_together() {
        let g = 200;
        let a: Vec<usize> = (0..150).collect();
        let b: Vec<usize> = (50..200).collect();
        let mut obs = Vec::new();
        for _ in 0..4 {
            obs.push(row(&a, g));
        }
        for _ in 0..5 {
            obs.push(row(&b, g));
        }
        obs.push(row(&(0..30).collect::<Vec<_>>(), g)); // singleton
        let cl = jaccard_plex_groups(&obs, 0.05, 4);
        assert!(cl[..4].iter().all(|&c| c == 0));
        assert!(cl[4..9].iter().all(|&c| c == 1));
        assert_eq!(cl[9], NO_PLEX);
    }

    #[test]
    fn small_clusters_are_unassigned() {
        let g = 64;
        let obs = vec![row(&[1, 2, 3], g), row(&[1, 2, 3], g), row(&[40, 41], g)];
        assert_eq!(jaccard_plex_groups(&obs, 0.05, 4), vec![NO_PLEX; 3]);
        assert_eq!(jaccard_plex_groups(&obs, 0.05, 2), vec![0, 0, NO_PLEX]);
    }
}
