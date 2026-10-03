//! BRIDLE: Batch Removal via Intrinsic Detectability and Latent Estimation,
//! for multi-dataset collections (linear `f`).
//!
//! Integrates many proteomics datasets that measure overlapping sets of
//! biological units onto one scale. A unit measured by >= 2 datasets is an
//! *anchor sample*: a cell line, a reference material, a pooled QC or the same
//! patient across cohorts (the code calls every unit a "line", `l`). Each
//! dataset's offset `A` is predicted from intrinsic protein detectability
//! features (`f`) and refined on its anchors (`r`); the shared biology is a
//! latent low-rank estimate (`theta`). Port of the
//! benchmark-winning `lim_lin` variant of the Cell Line Collection prototype
//! (`research/cellline-integration/lim_lin/lim.py`; Gaussian likelihood, linear
//! `f`, joint plex block, sample loading on, `theta` shared across datasets).
//!
//! Model, for profile `i` = (dataset `s(i)`, line `l(i)`, plex `k(i)`) and gene `g`:
//!
//! ```text
//! y[i,g]     = theta[l,g] + A[s,g] + c[i] + P[k,g] + eps,  eps ~ N(0, sig2[s,g])
//! log sig2   = a_s + b_s * abund_g                          (per-dataset noise trend)
//! theta[l,g] = m_g + Lin[lineage(l),g] + U_l . V_g + R[l,g] (biology; rank-`rank` U.V)
//! A[s,g]     = a0[s,g] + r[s,g]                            (A[reference] = 0)
//!   a0       = graph prior offset where one exists (`graph_prior`, see
//!              [`graph`]: plex-aware anchor offsets chained to the reference),
//!              else f(s, x_g)
//!   f        = per-dataset weighted ridge on [1, x_g] (technical protein features)
//!   r        ~ N(0, tau_s^2), empirical-Bayes residual around a0, only for
//!              anchored datasets
//! P[k,g]     ~ N(0, tauP^2), TMT plex effects, centred within each dataset
//! c[i]       ~ N(0, sd_c^2), per-profile sample loading
//! ```
//!
//! Fit: block-coordinate closed-form weighted ridge (`m`, `Lin`, ALS for
//! `U`/`V`, `R`, `f`, `r`, `P`, `c`) with EM moment updates for the variance
//! components (`tauR`, `tau_s`, `tauP`) and a per-dataset noise trend estimated
//! from leverage-corrected anchor-row residuals, for at most `sweeps` sweeps
//! ([`StopRule`]: by default until the output stops changing, i.e. the mean
//! |change| of `A + c + P` over observed cells between two sweeps falls below
//! `converge_tol`, after `min_sweeps` sweeps; a `holdout_frac` sample of observed
//! cells is withheld from the fit as an MSE monitor).
//!
//! Output: `v = y - A_out - P` (plus `- c` with `keep_sample_loading` off) on
//! observed cells only; nothing is imputed
//! and no protein is dropped for having missing values. `A_out` is cross-fitted
//! for datasets with `1 <= n_anchor < cf_max_nb`: an anchor profile's own
//! residual offset `r` is estimated only from the other anchor-sample folds
//! against a leave-dataset-out `theta`, so a single-line dataset keeps its own
//! signal (`r = 0`, `A = a0`). `theta` is returned separately. With
//! `anchor_scale`, each dataset's corrected values are then rescaled onto the
//! reference's spread by a slope estimated on shared anchors ([`anchor_scale`]).
//! With `plex_rescale` (default), every (dataset x plex, gene) is finally
//! rescaled around its own mean by an empirical-Bayes ComBat-style scale
//! estimated label-blind from all its values ([`rescale`]).
//!
//! Deviations from the prototype (no-ops on the benchmark, see the PR):
//! plex effects are centred per dataset (the prototype centres across all
//! plexes, identical when one dataset has plexes), and the cross-fit residual
//! subtracts the profile's plex effect (the prototype omits it; it only matters
//! for a plexed dataset with fewer than `cf_max_nb` anchor samples).
//!
//! Numerics: all matrices are stored gene-major and every block is a per-gene
//! (or per-line / per-dataset) rayon loop whose reductions run in a fixed order,
//! so results do not depend on the thread count. Random draws use a NumPy
//! `RandomState`-compatible stream ([`NumpyRandomState`]).

mod features;
mod graph;
mod linalg;
mod plex;
mod rescale;
mod rng;
mod scale;

use std::collections::{BTreeMap, BTreeSet, HashMap};

use mokume_core::{MokumeError, Result};
use rayon::prelude::*;

pub use features::{build_design, sequence_features, SequenceFeatures, FEATURE_NAMES};
pub use plex::{attach_to_nearest_plex, jaccard_plex_groups, NO_PLEX};
pub use rescale::PlexRescale;
pub use rng::NumpyRandomState;
pub use scale::{anchor_scale, AnchorScale};

/// Genes per work block for row reductions (fixed, so sums are deterministic).
const GENE_BLOCK: usize = 256;
/// Lower / upper clamp of `sig2[s,g]`.
const SIG2_RANGE: (f64, f64) = (0.02, 25.0);
/// Clamp of `tau_s^2`.
const TAUS2_RANGE: (f64, f64) = (0.02 * 0.02, 2.0 * 2.0);
/// Abundance bins of the noise trend.
const NOISE_BINS: usize = 10;

/// How plex groups are obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlexMode {
    /// No plex block.
    Off,
    /// Use `BridleData::plexes` (explicit, e.g. SDRF TMT mixture ids).
    Explicit,
    /// Infer plexes by shared-missingness Jaccard clustering (prototype rule).
    Inferred,
}

/// When the fit stops before `sweeps`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopRule {
    /// After `min_sweeps` sweeps, stop when the mean |change| of the output
    /// offsets `A + c + P` over observed cells since the previous sweep is
    /// below `converge_tol` (benchmark: 60 sweeps were not converged; 200-400
    /// sweeps improved every metric).
    OutputChange,
    /// Previous default: after `min_sweeps` sweeps, stop when the monitor MSE
    /// changed less than `converge_tol` over the last 3 sweeps.
    MonitorMse,
    /// Always run `sweeps` sweeps.
    Never,
}

/// Input profiles in long-to-wide form.
#[derive(Debug, Clone)]
pub struct BridleData {
    /// Dataset of each profile.
    pub datasets: Vec<String>,
    /// Anchor / biological unit (e.g. cell line) of each profile.
    pub lines: Vec<String>,
    /// Lineage of each profile's line, when known.
    pub lineages: Vec<Option<String>>,
    /// Explicit plex id of each profile (used with [`PlexMode::Explicit`]).
    /// Ids are scoped to the profile's dataset.
    pub plexes: Option<Vec<Option<String>>>,
    /// Gene / protein identifiers (columns).
    pub genes: Vec<String>,
    /// Row-major `profiles x genes`; `NaN` = not observed.
    pub values: Vec<f64>,
}

impl BridleData {
    pub fn n_profiles(&self) -> usize {
        self.datasets.len()
    }
}

/// Model hyper-parameters (defaults = the benchmark-winning configuration).
#[derive(Debug, Clone)]
pub struct BridleParams {
    /// Reference dataset (`A = 0`).
    pub reference: String,
    /// Rank of the biological low-rank term `U.V`.
    pub rank: usize,
    /// Maximum number of sweeps.
    pub sweeps: usize,
    pub seed: u32,
    pub tau_r: f64,
    pub tau_s: f64,
    pub tau_p: f64,
    pub tau_l: f64,
    pub sd_c: f64,
    pub lam_uv: f64,
    /// Ridge penalty of the per-dataset feature regression `f`.
    pub ridge_f: f64,
    /// Minimum number of informative genes to fit `f` for a dataset.
    pub min_f_genes: usize,
    /// Cross-fit datasets with `1 <= n_anchor < cf_max_nb`.
    pub cf_max_nb: usize,
    pub folds: usize,
    /// Fraction of observed cells withheld as a convergence monitor.
    pub holdout_frac: f64,
    pub sample_loading: bool,
    /// Keep the per-profile sample loading `c` in the output (`v = y - A -
    /// P`). Benchmark (graph prior on): removing `c` is 0.011 more accurate
    /// on held-out lines (median |error|) but loses 0.010 CORUM AUROC and
    /// some cis/deletion signal, i.e. `c` carries biology as well as loading.
    pub keep_sample_loading: bool,
    pub plex_mode: PlexMode,
    /// Inferred plexes: Jaccard-distance cut, minimum cluster size, minimum
    /// profiles in a dataset, minimum assigned fraction and minimum #plexes.
    pub plex_cut: f64,
    pub plex_min_size: usize,
    pub plex_min_profiles: usize,
    pub plex_min_assigned: f64,
    pub plex_min_groups: usize,
    /// Explicit plexes: attach profiles without a plex id to the nearest
    /// explicit plex of their dataset by missingness (Jaccard <= `plex_cut`).
    pub plex_attach_unlabelled: bool,
    /// Early stop rule, its minimum number of sweeps and its tolerance.
    pub stop_rule: StopRule,
    pub min_sweeps: usize,
    pub converge_tol: f64,
    /// Post-fit per-dataset scale from anchor samples shared with the
    /// reference (see [`anchor_scale`]); off by default.
    pub anchor_scale: bool,
    /// Minimum anchor samples shared with the reference for the scale step.
    pub anchor_scale_min_anchors: usize,
    /// Graph prior: plex-aware anchor offsets chained to the reference are
    /// the initial value and prior mean of `A` where they exist (else `f`).
    pub graph_prior: bool,
    /// Graph prior: anchor lines two batches must share to be linked, and
    /// alternating-means iterations.
    pub graph_min_shared: usize,
    pub graph_iters: usize,
    /// Post-fit label-blind ComBat-style scale per (dataset x plex, gene),
    /// see [`rescale`].
    pub plex_rescale: bool,
}

impl BridleParams {
    /// The previous convergence defaults: at most 60 sweeps, stop on the
    /// monitor MSE (2e-4 over 3 sweeps, after 8 sweeps).
    pub fn monitor_stop(self) -> Self {
        Self {
            sweeps: 60,
            stop_rule: StopRule::MonitorMse,
            min_sweeps: 8,
            converge_tol: 2e-4,
            ..self
        }
    }
}

impl BridleParams {
    /// The configuration before the 2026-10 benchmark (the `lim_lin`
    /// prototype): [`Self::monitor_stop`] convergence and no graph prior.
    pub fn legacy() -> Self {
        Self {
            graph_prior: false,
            keep_sample_loading: false,
            plex_rescale: false,
            ..Self::default().monitor_stop()
        }
    }
}

impl Default for BridleParams {
    fn default() -> Self {
        Self {
            reference: String::new(),
            rank: 16,
            sweeps: 400,
            seed: 0,
            tau_r: 0.8,
            tau_s: 0.3,
            tau_p: 0.3,
            tau_l: 1.0,
            sd_c: 0.3,
            lam_uv: 10.0,
            ridge_f: 1.0,
            min_f_genes: 50,
            cf_max_nb: 20,
            folds: 5,
            holdout_frac: 0.01,
            sample_loading: true,
            keep_sample_loading: true,
            plex_mode: PlexMode::Inferred,
            plex_cut: 0.05,
            plex_min_size: 4,
            plex_min_profiles: 20,
            plex_min_assigned: 0.5,
            plex_min_groups: 5,
            plex_attach_unlabelled: true,
            stop_rule: StopRule::OutputChange,
            min_sweeps: 200,
            converge_tol: 1e-5,
            anchor_scale: false,
            anchor_scale_min_anchors: 20,
            graph_prior: true,
            graph_min_shared: 3,
            graph_iters: 400,
            plex_rescale: true,
        }
    }
}

/// Per-dataset fit summary.
#[derive(Debug, Clone)]
pub struct DatasetReport {
    pub name: String,
    pub n_profiles: usize,
    pub n_anchor_samples: usize,
    pub anchored: bool,
    pub cross_fitted: bool,
    pub n_plexes: usize,
    pub n_plexed_profiles: usize,
    pub tau_s: f64,
    pub sig2: f64,
    pub sig_slope: f64,
    pub f_fitted: bool,
    /// Genes whose offset prior is the graph prior (else `f`).
    pub graph_prior_genes: usize,
}

/// Per-sweep monitor.
#[derive(Debug, Clone)]
pub struct SweepStats {
    pub sweep: usize,
    pub hold_mse: f64,
    /// Mean |change| of `A + c + P` over observed cells since the previous
    /// sweep (`NaN` on the first sweep).
    pub out_change: f64,
    pub tau_r: f64,
    pub tau_p: f64,
}

/// Fit summary.
#[derive(Debug, Clone)]
pub struct BridleReport {
    pub reference: String,
    pub n_profiles: usize,
    pub n_genes: usize,
    pub n_lines: usize,
    pub n_lineages: usize,
    pub n_plexes: usize,
    pub design_columns: usize,
    pub observed_cells: usize,
    pub holdout_cells: usize,
    pub converged: bool,
    pub datasets: Vec<DatasetReport>,
    pub history: Vec<SweepStats>,
    pub sample_loading_sd: f64,
    /// Per-dataset anchor scale (empty unless `BridleParams::anchor_scale`).
    pub anchor_scale: Vec<AnchorScale>,
    /// Per-batch plex rescale (empty unless `BridleParams::plex_rescale`).
    pub plex_rescale: Vec<PlexRescale>,
}

/// Fit result.
#[derive(Debug, Clone)]
pub struct BridleResult {
    /// Row-major `profiles x genes`: `v = y - A_out - P` (`- c` unless
    /// `keep_sample_loading`), `NaN` exactly where the input is `NaN`.
    pub corrected: Vec<f64>,
    /// Line names (sorted), the rows of `theta`.
    pub line_names: Vec<String>,
    /// Row-major `lines x genes` fitted biology `theta` (defined everywhere).
    pub theta: Vec<f64>,
    /// Row-major `lines x genes`: the line has >= 1 observed profile value.
    pub theta_observed: Vec<bool>,
    /// Dataset names (sorted), the rows of `offsets` / `feature_offsets`.
    pub dataset_names: Vec<String>,
    /// Row-major `datasets x genes` pooled technical offset `A = f + r` (the
    /// output of a cross-fitted dataset uses fold-specific `r`, see module doc).
    pub offsets: Vec<f64>,
    /// Row-major `datasets x genes` feature-explained part `f`.
    pub feature_offsets: Vec<f64>,
    /// Per-profile sample loading `c`.
    pub sample_loading: Vec<f64>,
    /// Plex index of each profile (`None` = no plex).
    pub profile_plex: Vec<Option<usize>>,
    pub report: BridleReport,
}

/// Per-gene abundance axis: median of the reference dataset's observed values,
/// falling back to the median over all profiles.
pub fn reference_abundance(data: &BridleData, reference: &str) -> Vec<f64> {
    let g_n = data.genes.len();
    let n = data.n_profiles();
    let ref_rows: Vec<usize> = (0..n).filter(|&i| data.datasets[i] == reference).collect();
    (0..g_n)
        .into_par_iter()
        .map(|g| {
            let m = features::nan_median(ref_rows.iter().map(|&i| data.values[i * g_n + g]));
            if m.is_finite() {
                m
            } else {
                features::nan_median((0..n).map(|i| data.values[i * g_n + g]))
            }
        })
        .collect()
}

fn invalid(message: impl Into<String>) -> MokumeError {
    MokumeError::InvalidInput {
        message: message.into(),
    }
}

/// Index structure shared by all blocks.
struct Layout {
    n: usize,
    g_n: usize,
    s_n: usize,
    nl: usize,
    nlin: usize,
    k: usize,
    srow: Vec<usize>,
    li: Vec<usize>,
    lin_of_line: Vec<usize>,
    prow: Vec<usize>,
    plex_ds: Vec<usize>,
    /// Plex id (explicit) or cluster number (inferred) of each plex.
    plex_names: Vec<String>,
    anchor_row: Vec<bool>,
    nb: Vec<usize>,
    anchored: Vec<bool>,
    uv_free: Vec<bool>,
    ref_s: usize,
    studies: Vec<String>,
    line_names: Vec<String>,
    line_rows: Vec<Vec<usize>>,
    n_lineages: usize,
}

fn build_layout(data: &BridleData, params: &BridleParams) -> Result<(Layout, Vec<String>)> {
    let n = data.n_profiles();
    let g_n = data.genes.len();
    if n == 0 || g_n == 0 {
        return Err(invalid("BRIDLE needs at least one profile and one gene"));
    }
    if data.lines.len() != n || data.lineages.len() != n || data.values.len() != n * g_n {
        return Err(invalid("BRIDLE input arrays have inconsistent lengths"));
    }
    let studies: Vec<String> = data
        .datasets
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let sidx: HashMap<&str, usize> = studies
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let ref_s = *sidx.get(params.reference.as_str()).ok_or_else(|| {
        invalid(format!(
            "reference dataset '{}' not found in the input",
            params.reference
        ))
    })?;
    let srow: Vec<usize> = data.datasets.iter().map(|d| sidx[d.as_str()]).collect();
    let line_names: Vec<String> = data
        .lines
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let lidx: HashMap<&str, usize> = line_names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let li: Vec<usize> = data.lines.iter().map(|l| lidx[l.as_str()]).collect();
    let nl = line_names.len();
    let s_n = studies.len();

    let mut seen_pairs = BTreeSet::new();
    for i in 0..n {
        if !seen_pairs.insert((srow[i], li[i])) {
            return Err(invalid(format!(
                "duplicate profile (dataset '{}', line '{}'); collapse replicates first",
                data.datasets[i], data.lines[i]
            )));
        }
    }
    let mut line_rows = vec![Vec::new(); nl];
    for i in 0..n {
        line_rows[li[i]].push(i);
    }
    let lcount: Vec<usize> = line_rows.iter().map(Vec::len).collect();
    let anchor_row: Vec<bool> = (0..n).map(|i| lcount[li[i]] >= 2).collect();
    let mut nb = vec![0_usize; s_n];
    for i in 0..n {
        if anchor_row[i] {
            nb[srow[i]] += 1;
        }
    }
    // lineage of each line: first known lineage in row order
    let mut lin_line: Vec<Option<&str>> = vec![None; nl];
    for i in 0..n {
        if lin_line[li[i]].is_none() {
            lin_line[li[i]] = data.lineages[i].as_deref().filter(|s| !s.is_empty());
        }
    }
    let lins: Vec<&str> = lin_line
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let lin_pos: HashMap<&str, usize> = lins.iter().enumerate().map(|(i, s)| (*s, i)).collect();
    let lin_of_line: Vec<usize> = lin_line
        .iter()
        .map(|l| l.map_or(lins.len(), |s| lin_pos[s]))
        .collect();
    let ds_anchored: Vec<bool> = nb.iter().map(|&b| b > 0).collect();
    let uv_free: Vec<bool> = line_rows
        .iter()
        .map(|rows| rows.iter().any(|&i| ds_anchored[srow[i]]))
        .collect();
    let anchored: Vec<bool> = (0..s_n).map(|s| nb[s] > 0 && s != ref_s).collect();

    if data.plexes.as_ref().is_some_and(|p| p.len() != n) {
        return Err(invalid("plex ids must have one entry per profile"));
    }
    let (prow, plex_ds, plex_names, plex_info) = assign_plexes(data, params, &srow, &studies, g_n)?;
    let k = plex_ds.len();
    let prow: Vec<usize> = prow.into_iter().map(|p| p.unwrap_or(k)).collect();
    Ok((
        Layout {
            n,
            g_n,
            s_n,
            nl,
            nlin: lins.len() + 1,
            k,
            srow,
            li,
            lin_of_line,
            prow,
            plex_ds,
            plex_names,
            anchor_row,
            nb,
            anchored,
            uv_free,
            ref_s,
            studies,
            line_names,
            line_rows,
            n_lineages: lins.len(),
        },
        plex_info,
    ))
}

type PlexAssignment = (Vec<Option<usize>>, Vec<usize>, Vec<String>, Vec<String>);

fn assign_plexes(
    data: &BridleData,
    params: &BridleParams,
    srow: &[usize],
    studies: &[String],
    g_n: usize,
) -> Result<PlexAssignment> {
    let n = srow.len();
    let mut prow: Vec<Option<usize>> = vec![None; n];
    let mut plex_ds = Vec::new();
    let mut plex_names: Vec<String> = Vec::new();
    let mut info = Vec::new();
    for (s, name) in studies.iter().enumerate() {
        let rows: Vec<usize> = (0..n).filter(|&i| srow[i] == s).collect();
        let mut names: Vec<String> = Vec::new();
        let labels: Vec<i64> = match params.plex_mode {
            PlexMode::Off => continue,
            PlexMode::Explicit => {
                let plexes = data
                    .plexes
                    .as_ref()
                    .ok_or_else(|| invalid("explicit plex mode requires plex ids"))?;
                let mut ids: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
                for (pos, &i) in rows.iter().enumerate() {
                    if let Some(p) = plexes[i].as_deref().filter(|p| !p.is_empty()) {
                        ids.entry(p).or_default().push(pos);
                    }
                }
                let mut labels = vec![NO_PLEX; rows.len()];
                let groups: Vec<&Vec<usize>> = ids.values().filter(|m| m.len() >= 2).collect();
                if groups.len() < 2 {
                    continue;
                }
                names = ids
                    .iter()
                    .filter(|(_, m)| m.len() >= 2)
                    .map(|(id, _)| (*id).to_owned())
                    .collect();
                for (kk, m) in groups.iter().enumerate() {
                    for &pos in *m {
                        labels[pos] = kk as i64;
                    }
                }
                if params.plex_attach_unlabelled && labels.contains(&NO_PLEX) {
                    let observed: Vec<Vec<bool>> = rows
                        .iter()
                        .map(|&i| {
                            (0..g_n)
                                .map(|g| data.values[i * g_n + g].is_finite())
                                .collect()
                        })
                        .collect();
                    let unl = labels.iter().filter(|&&l| l == NO_PLEX).count();
                    let att = attach_to_nearest_plex(&observed, &mut labels, params.plex_cut);
                    info.push(format!(
                        "{name}: {att}/{unl} profiles without a plex id attached to the nearest plex by missingness"
                    ));
                }
                labels
            }
            PlexMode::Inferred => {
                if rows.len() < params.plex_min_profiles {
                    continue;
                }
                let observed: Vec<Vec<bool>> = rows
                    .iter()
                    .map(|&i| {
                        (0..g_n)
                            .map(|g| data.values[i * g_n + g].is_finite())
                            .collect()
                    })
                    .collect();
                let cl = jaccard_plex_groups(&observed, params.plex_cut, params.plex_min_size);
                let assigned = cl.iter().filter(|&&c| c >= 0).count();
                let groups = cl
                    .iter()
                    .copied()
                    .max()
                    .map_or(0, |m| (m + 1).max(0) as usize);
                if (assigned as f64) < params.plex_min_assigned * rows.len() as f64
                    || groups < params.plex_min_groups
                {
                    continue;
                }
                cl
            }
        };
        let groups = labels
            .iter()
            .copied()
            .max()
            .map_or(0, |m| (m + 1).max(0) as usize);
        let base = plex_ds.len();
        plex_ds.extend(std::iter::repeat_n(s, groups));
        names.resize_with(groups, String::new);
        for (kk, nm) in names.iter_mut().enumerate() {
            if nm.is_empty() {
                *nm = kk.to_string();
            }
        }
        plex_names.extend(names);
        let mut assigned = 0;
        for (pos, &lab) in labels.iter().enumerate() {
            if lab >= 0 {
                prow[rows[pos]] = Some(base + lab as usize);
                assigned += 1;
            }
        }
        info.push(format!(
            "{name}: {groups} plexes, {assigned}/{} profiles",
            rows.len()
        ));
    }
    Ok((prow, plex_ds, plex_names, info))
}

/// Mutable model state (gene-major matrices).
struct State {
    m: Vec<f64>,
    lin: Vec<f64>,
    u: Vec<f64>,
    v: Vec<f64>,
    r_line: Vec<f64>,
    a: Vec<f64>,
    f: Vec<f64>,
    p: Vec<f64>,
    c: Vec<f64>,
    sig2_s: Vec<f64>,
    sig_b: Vec<f64>,
    tau_r2: f64,
    tau_p2: f64,
    taus2: Vec<f64>,
    f_fitted: Vec<bool>,
    /// Gene-major graph prior of `A` (`NaN` = none, the prior mean is `f`).
    prior_a: Vec<f64>,
}

impl State {
    #[inline]
    fn sig2(&self, abz: f64, s: usize) -> f64 {
        (self.sig2_s[s] * (self.sig_b[s] * abz).exp()).clamp(SIG2_RANGE.0, SIG2_RANGE.1)
    }

    /// Prior mean of `A[s,g]`: the graph prior where it exists, else `f`.
    #[inline]
    fn prior_mean(&self, s_n: usize, g: usize, s: usize) -> f64 {
        let p = self.prior_a[g * s_n + s];
        if p.is_finite() {
            p
        } else {
            self.f[g * s_n + s]
        }
    }

    #[inline]
    fn uv(&self, rank: usize, g: usize, l: usize) -> f64 {
        let ul = &self.u[l * rank..(l + 1) * rank];
        let vg = &self.v[g * rank..(g + 1) * rank];
        ul.iter().zip(vg).map(|(a, b)| a * b).sum()
    }
}

/// Cell state in the gene-major observation mask.
const UNOBSERVED: u8 = 0;
const OBSERVED: u8 = 1;
const HELD: u8 = 2;

/// Fit BRIDLE.
///
/// `design` is `genes x p` (first column the intercept, see [`build_design`]);
/// `abz` is the standardised abundance axis returned by [`build_design`].
pub fn bridle_fit(
    data: &BridleData,
    design: &[Vec<f64>],
    abz: &[f64],
    params: &BridleParams,
) -> Result<BridleResult> {
    let (lay, plex_info) = build_layout(data, params)?;
    let (n, g_n, s_n, nl, nlin, k) = (lay.n, lay.g_n, lay.s_n, lay.nl, lay.nlin, lay.k);
    if design.len() != g_n || abz.len() != g_n {
        return Err(invalid(
            "design / abundance length does not match the number of genes",
        ));
    }
    let p_dim = design.first().map_or(0, Vec::len);
    if p_dim == 0 || design.iter().any(|row| row.len() != p_dim) {
        return Err(invalid(
            "design matrix must be rectangular with >= 1 column",
        ));
    }
    if params.rank == 0 || params.folds == 0 {
        return Err(invalid("rank and folds must be >= 1"));
    }
    let rank = params.rank;
    let kp = k + 1;
    for line in &plex_info {
        tracing::info!("BRIDLE plex: {line}");
    }

    // ---- gene-major observations, monitor hold-out (row-major draw order)
    let mut rs = NumpyRandomState::new(params.seed);
    let mut y = vec![0.0_f64; g_n * n];
    let mut obs = vec![UNOBSERVED; g_n * n];
    let (mut n_obs, mut n_hold) = (0_usize, 0_usize);
    for i in 0..n {
        for g in 0..g_n {
            let x = data.values[i * g_n + g];
            let u = rs.next_f64();
            if x.is_finite() {
                y[g * n + i] = x;
                let held = u < params.holdout_frac;
                obs[g * n + i] = if held { HELD } else { OBSERVED };
                n_obs += 1;
                n_hold += usize::from(held);
            }
        }
    }
    tracing::info!(
        "BRIDLE: n={n} G={g_n} S={s_n} lines={nl} anchored datasets={} plexes={k} observed={n_obs} held={n_hold}",
        lay.anchored.iter().filter(|&&b| b).count()
    );
    // datasets that share no anchor sample with any other dataset get no
    // residual offset `r`: their correction is the feature model `f` alone
    for (s, name) in lay.studies.iter().enumerate() {
        if lay.nb[s] > 0 {
            continue;
        }
        if s == lay.ref_s {
            tracing::warn!(
                "BRIDLE: reference dataset '{name}' shares no anchor samples with the rest; \
                 no other dataset can be anchored to it"
            );
        } else {
            tracing::warn!(
                "BRIDLE: dataset '{name}' shares no anchor samples with the rest; \
                 only intrinsic-detectability correction applied"
            );
        }
    }

    // ---- graph prior of the offsets (gene-major), also their initial value
    let prior_a = if params.graph_prior {
        let batch: Vec<usize> = (0..n)
            .map(|i| {
                let s = lay.srow[i];
                if s != lay.ref_s && lay.prow[i] < k {
                    s_n + lay.prow[i]
                } else {
                    s
                }
            })
            .collect();
        let rm = graph::graph_prior(&graph::GraphInput {
            values: &data.values,
            g_n,
            s_n,
            srow: &lay.srow,
            li: &lay.li,
            batch: &batch,
            n_batch: s_n + k,
            ref_s: lay.ref_s,
            min_shared: params.graph_min_shared,
            iters: params.graph_iters,
        });
        let mut gm = vec![f64::NAN; g_n * s_n];
        for s in 0..s_n {
            for g in 0..g_n {
                gm[g * s_n + s] = rm[s * g_n + g];
            }
        }
        tracing::info!(
            "BRIDLE graph prior: {} of {} dataset x gene offsets",
            gm.iter().filter(|x| x.is_finite()).count(),
            (s_n - 1) * g_n
        );
        gm
    } else {
        vec![f64::NAN; g_n * s_n]
    };
    let a0: Vec<f64> = prior_a
        .iter()
        .map(|&p| if p.is_finite() { p } else { 0.0 })
        .collect();

    // ---- initial state (U, V from a second NumPy stream, like torch.randn)
    let mut init = NumpyRandomState::new(params.seed.wrapping_add(1));
    let u0: Vec<f64> = (0..nl * rank).map(|_| 0.01 * init.next_gauss()).collect();
    let v0: Vec<f64> = (0..g_n * rank).map(|_| 0.01 * init.next_gauss()).collect();
    let mut st = State {
        m: vec![0.0; g_n],
        lin: vec![0.0; g_n * nlin],
        u: u0,
        v: v0,
        r_line: vec![0.0; g_n * nl],
        a: a0,
        f: vec![0.0; g_n * s_n],
        p: vec![0.0; g_n * kp],
        c: vec![0.0; n],
        sig2_s: vec![0.3; s_n],
        sig_b: vec![0.0; s_n],
        tau_r2: params.tau_r * params.tau_r,
        tau_p2: params.tau_p * params.tau_p,
        taus2: vec![params.tau_s * params.tau_s; s_n],
        f_fitted: vec![false; s_n],
        prior_a,
    };
    let tau_l2 = params.tau_l * params.tau_l;
    let lam_c = 1.0 / (params.sd_c * params.sd_c);

    // noise-trend abundance bins (torch.quantile + bucketize(right=False))
    let mut sorted_abz = abz.to_vec();
    sorted_abz.sort_by(f64::total_cmp);
    let bounds: Vec<f64> = (1..NOISE_BINS)
        .map(|b| features::quantile_sorted(&sorted_abz, b as f64 / NOISE_BINS as f64))
        .collect();
    let gbin: Vec<usize> = abz
        .iter()
        .map(|&x| bounds.iter().filter(|&&b| b < x).count())
        .collect();
    let mut bin_x = vec![0.0; NOISE_BINS];
    let mut bin_n = [0_usize; NOISE_BINS];
    for g in 0..g_n {
        bin_x[gbin[g]] += abz[g];
        bin_n[gbin[g]] += 1;
    }
    for b in 0..NOISE_BINS {
        bin_x[b] /= bin_n[b].max(1) as f64;
    }
    let ds_has_anchor: Vec<bool> = (0..s_n)
        .map(|s| (0..n).any(|i| lay.srow[i] == s && lay.anchor_row[i]))
        .collect();

    let mut wl = vec![0.0_f64; g_n * nl];
    let mut zl = vec![0.0_f64; g_n * nl];
    let mut work = vec![0.0_f64; g_n * nl];
    let mut theta = vec![0.0_f64; g_n * nl];
    let mut history: Vec<SweepStats> = Vec::new();
    let mut converged = false;
    let mut prev_offset: Option<Vec<f64>> = None;

    for sweep in 0..params.sweeps {
        // ============ theta blocks on line-level aggregates
        {
            let st_ref = &st;
            wl.par_chunks_mut(nl)
                .zip(zl.par_chunks_mut(nl))
                .enumerate()
                .for_each(|(g, (wl_g, zl_g))| {
                    wl_g.fill(0.0);
                    zl_g.fill(0.0);
                    let a_g = &st_ref.a[g * s_n..(g + 1) * s_n];
                    let p_g = &st_ref.p[g * kp..(g + 1) * kp];
                    for i in 0..n {
                        if obs[g * n + i] != OBSERVED {
                            continue;
                        }
                        let s = lay.srow[i];
                        let w = 1.0 / st_ref.sig2(abz[g], s);
                        let l = lay.li[i];
                        wl_g[l] += w;
                        zl_g[l] += w * (y[g * n + i] - a_g[s] - st_ref.c[i] - p_g[lay.prow[i]]);
                    }
                    for l in 0..nl {
                        zl_g[l] = if wl_g[l] > 0.0 {
                            zl_g[l] / wl_g[l].max(1e-12)
                        } else {
                            0.0
                        };
                    }
                });
        }
        // m, Lin; then WZ = Wl * (Zl - m - Lin - R) for ALS (stored in `work`)
        {
            let (u, v, r_line) = (&st.u, &st.v, &st.r_line);
            let lin_old = &st.lin;
            let (wl, zl) = (&wl, &zl);
            let res: Vec<(f64, Vec<f64>)> = (0..g_n)
                .into_par_iter()
                .map(|g| {
                    let wl_g = &wl[g * nl..(g + 1) * nl];
                    let zl_g = &zl[g * nl..(g + 1) * nl];
                    let r_g = &r_line[g * nl..(g + 1) * nl];
                    let lin_g = &lin_old[g * nlin..(g + 1) * nlin];
                    let vg = &v[g * rank..(g + 1) * rank];
                    let uv: Vec<f64> = (0..nl)
                        .map(|l| {
                            u[l * rank..(l + 1) * rank]
                                .iter()
                                .zip(vg)
                                .map(|(a, b)| a * b)
                                .sum()
                        })
                        .collect();
                    let (mut num, mut den) = (0.0, 0.0);
                    for l in 0..nl {
                        num += wl_g[l] * (zl_g[l] - lin_g[lay.lin_of_line[l]] - uv[l] - r_g[l]);
                        den += wl_g[l];
                    }
                    let m = num / den.max(1e-12);
                    let mut lnum = vec![0.0; nlin];
                    let mut lden = vec![0.0; nlin];
                    for l in 0..nl {
                        let k_ = lay.lin_of_line[l];
                        lnum[k_] += wl_g[l] * (zl_g[l] - m - uv[l] - r_g[l]);
                        lden[k_] += wl_g[l];
                    }
                    let mut lin = vec![0.0; nlin];
                    for k_ in 0..nlin - 1 {
                        lin[k_] = lnum[k_] / (lden[k_] + 1.0 / tau_l2);
                    }
                    (m, lin)
                })
                .collect();
            for (g, (m, lin)) in res.into_iter().enumerate() {
                st.m[g] = m;
                st.lin[g * nlin..(g + 1) * nlin].copy_from_slice(&lin);
            }
        }
        {
            let (m, lin, r_line) = (&st.m, &st.lin, &st.r_line);
            let (wl, zl) = (&wl, &zl);
            work.par_chunks_mut(nl).enumerate().for_each(|(g, wz)| {
                for l in 0..nl {
                    let z2 = zl[g * nl + l]
                        - m[g]
                        - lin[g * nlin + lay.lin_of_line[l]]
                        - r_line[g * nl + l];
                    wz[l] = wl[g * nl + l] * z2;
                }
            });
        }
        als(&lay, params, &wl, &work, &mut st.u, &mut st.v);
        // R, tauR, theta
        {
            let tau_r2 = st.tau_r2;
            let (m, lin, u, v) = (&st.m, &st.lin, &st.u, &st.v);
            let (wl, zl) = (&wl, &zl);
            let parts: Vec<(f64, usize)> = st
                .r_line
                .par_chunks_mut(nl)
                .zip(theta.par_chunks_mut(nl))
                .enumerate()
                .map(|(g, (r_g, th_g))| {
                    let vg = &v[g * rank..(g + 1) * rank];
                    let (mut sum, mut cnt) = (0.0, 0_usize);
                    for l in 0..nl {
                        let uv: f64 = u[l * rank..(l + 1) * rank]
                            .iter()
                            .zip(vg)
                            .map(|(a, b)| a * b)
                            .sum();
                        let w = wl[g * nl + l];
                        let prior = m[g] + lin[g * nlin + lay.lin_of_line[l]] + uv;
                        let r = w * (zl[g * nl + l] - prior) / (w + 1.0 / tau_r2);
                        r_g[l] = r;
                        th_g[l] = prior + r;
                        if w > 0.0 {
                            sum += r * r + 1.0 / (w + 1.0 / tau_r2);
                            cnt += 1;
                        }
                    }
                    (sum, cnt)
                })
                .collect();
            let (sum, cnt) = parts.iter().fold((0.0, 0), |(s, c), (a, b)| (s + a, c + b));
            if cnt > 0 {
                st.tau_r2 = sum / cnt as f64;
            }
        }
        // ============ A = f + r
        update_offsets(&lay, params, design, abz, &y, &obs, &theta, &mut st);
        // ============ plex
        if k > 0 {
            update_plex(&lay, abz, &y, &obs, &theta, &mut st);
        }
        // ============ sample loading
        if params.sample_loading {
            let (num, den) = row_reduce(
                &lay,
                |g, i, w, st_: &State| {
                    let pred = theta[g * nl + lay.li[i]]
                        + st_.a[g * s_n + lay.srow[i]]
                        + st_.p[g * kp + lay.prow[i]];
                    w * (y[g * n + i] - pred)
                },
                abz,
                &obs,
                &st,
            );
            for i in 0..n {
                st.c[i] = num[i] / (den[i] + lam_c);
            }
        }
        // ============ noise model + monitor
        let hold_mse = update_noise(
            &lay,
            abz,
            &y,
            &obs,
            &theta,
            &wl,
            &gbin,
            &bin_x,
            &ds_has_anchor,
            &mut st,
        );
        check_finite(&st)?;
        let offset = output_offsets(&lay, &obs, &st);
        let out_change = prev_offset
            .as_ref()
            .map_or(f64::NAN, |prev| mean_abs_change(prev, &offset, &obs));
        prev_offset = Some(offset);
        history.push(SweepStats {
            sweep,
            hold_mse,
            out_change,
            tau_r: st.tau_r2.sqrt(),
            tau_p: st.tau_p2.sqrt(),
        });
        tracing::debug!(
            "BRIDLE sweep {sweep}: hold_mse={hold_mse:.4} out_change={out_change:.2e} tauR={:.3} tauP={:.3}",
            st.tau_r2.sqrt(),
            st.tau_p2.sqrt()
        );
        let h = history.len();
        let stop = match params.stop_rule {
            StopRule::OutputChange => {
                sweep + 1 >= params.min_sweeps && out_change < params.converge_tol
            }
            StopRule::MonitorMse => {
                sweep >= params.min_sweeps
                    && h > 3
                    && (history[h - 4].hold_mse - hold_mse).abs() < params.converge_tol
            }
            StopRule::Never => false,
        };
        if stop {
            converged = true;
            break;
        }
    }

    // ============ stage 2: cross-fitted output offsets
    let (mut corrected, cf) = output_values(&lay, params, abz, &y, &obs, &st);
    if params.keep_sample_loading {
        for (row, &c) in corrected.chunks_mut(g_n).zip(&st.c) {
            for v in row.iter_mut().filter(|v| v.is_finite()) {
                *v += c;
            }
        }
    }
    let scales = if params.anchor_scale {
        anchor_scale(
            data,
            &mut corrected,
            &params.reference,
            params.anchor_scale_min_anchors,
        )
    } else {
        Vec::new()
    };
    let rescales = if params.plex_rescale {
        // batches: datasets (unplexed profiles), then dataset x plex
        let mut names = lay.studies.clone();
        names.extend(
            (0..k).map(|kk| format!("{}|{}", lay.studies[lay.plex_ds[kk]], lay.plex_names[kk])),
        );
        let dataset: Vec<usize> = (0..s_n).chain(lay.plex_ds.iter().copied()).collect();
        let plexed: Vec<bool> = (0..s_n + k).map(|b| b >= s_n).collect();
        let batch: Vec<usize> = (0..n)
            .map(|i| {
                if lay.prow[i] < k {
                    s_n + lay.prow[i]
                } else {
                    lay.srow[i]
                }
            })
            .collect();
        rescale::plex_rescale(
            &mut corrected,
            g_n,
            &rescale::RescaleBatches {
                batch: &batch,
                names: &names,
                dataset: &dataset,
                plexed: &plexed,
            },
        )
    } else {
        Vec::new()
    };
    let mut theta_rows = vec![0.0; nl * g_n];
    let mut theta_obs = vec![false; nl * g_n];
    for g in 0..g_n {
        for l in 0..nl {
            theta_rows[l * g_n + g] = theta[g * nl + l];
        }
        for i in 0..n {
            if obs[g * n + i] != UNOBSERVED {
                theta_obs[lay.li[i] * g_n + g] = true;
            }
        }
    }
    let mut offsets = vec![0.0; s_n * g_n];
    let mut feature_offsets = vec![0.0; s_n * g_n];
    for g in 0..g_n {
        for s in 0..s_n {
            offsets[s * g_n + g] = st.a[g * s_n + s];
            feature_offsets[s * g_n + g] = st.f[g * s_n + s];
        }
    }
    let c_mean = st.c.iter().sum::<f64>() / n as f64;
    let c_sd =
        (st.c.iter().map(|x| (x - c_mean).powi(2)).sum::<f64>() / (n.max(2) - 1) as f64).sqrt();
    let datasets = (0..s_n)
        .map(|s| {
            let n_prof = lay.srow.iter().filter(|&&x| x == s).count();
            let plexed = (0..n)
                .filter(|&i| lay.srow[i] == s && lay.prow[i] < k)
                .count();
            DatasetReport {
                name: lay.studies[s].clone(),
                n_profiles: n_prof,
                n_anchor_samples: lay.nb[s],
                anchored: lay.anchored[s],
                cross_fitted: cf[s],
                n_plexes: lay.plex_ds.iter().filter(|&&d| d == s).count(),
                n_plexed_profiles: plexed,
                tau_s: st.taus2[s].sqrt(),
                sig2: st.sig2_s[s],
                sig_slope: st.sig_b[s],
                f_fitted: st.f_fitted[s],
                graph_prior_genes: (0..g_n)
                    .filter(|&g| st.prior_a[g * s_n + s].is_finite())
                    .count(),
            }
        })
        .collect();
    Ok(BridleResult {
        corrected,
        line_names: lay.line_names.clone(),
        theta: theta_rows,
        theta_observed: theta_obs,
        dataset_names: lay.studies.clone(),
        offsets,
        feature_offsets,
        sample_loading: st.c.clone(),
        profile_plex: lay.prow.iter().map(|&p| (p < k).then_some(p)).collect(),
        report: BridleReport {
            reference: params.reference.clone(),
            n_profiles: n,
            n_genes: g_n,
            n_lines: nl,
            n_lineages: lay.n_lineages,
            n_plexes: k,
            design_columns: p_dim,
            observed_cells: n_obs,
            holdout_cells: n_hold,
            converged,
            datasets,
            history,
            sample_loading_sd: c_sd,
            anchor_scale: scales,
            plex_rescale: rescales,
        },
    })
}

/// Two weighted-ridge ALS rounds for `U` (lines) and `V` (genes) on the fixed
/// working response `WZ = Wl * Z2` (`wz`, gene-major).
fn als(lay: &Layout, params: &BridleParams, wl: &[f64], wz: &[f64], u: &mut [f64], v: &mut [f64]) {
    let (g_n, nl, rank) = (lay.g_n, lay.nl, params.rank);
    // line-major copies for the U step
    let mut wl_t = vec![0.0; nl * g_n];
    let mut wz_t = vec![0.0; nl * g_n];
    wl_t.par_chunks_mut(g_n)
        .zip(wz_t.par_chunks_mut(g_n))
        .enumerate()
        .for_each(|(l, (a, b))| {
            for g in 0..g_n {
                a[g] = wl[g * nl + l];
                b[g] = wz[g * nl + l];
            }
        });
    for _ in 0..2 {
        {
            let v_ref: &[f64] = v;
            u.par_chunks_mut(rank).enumerate().for_each(|(l, ul)| {
                if !lay.uv_free[l] {
                    ul.fill(0.0);
                    return;
                }
                let sol = weighted_ridge_rows(
                    &wl_t[l * g_n..(l + 1) * g_n],
                    &wz_t[l * g_n..(l + 1) * g_n],
                    v_ref,
                    rank,
                    params.lam_uv,
                );
                ul.copy_from_slice(&sol);
            });
        }
        let u_ref: &[f64] = u;
        v.par_chunks_mut(rank).enumerate().for_each(|(g, vg)| {
            let sol = weighted_ridge_rows(
                &wl[g * nl..(g + 1) * nl],
                &wz[g * nl..(g + 1) * nl],
                u_ref,
                rank,
                params.lam_uv,
            );
            vg.copy_from_slice(&sol);
        });
    }
}

/// Solve `(sum_j w_j x_j x_j' + lam I) b = sum_j wz_j x_j` where `x_j` is row
/// `j` of the row-major `factors` (`rank` columns).
fn weighted_ridge_rows(w: &[f64], wz: &[f64], factors: &[f64], rank: usize, lam: f64) -> Vec<f64> {
    let mut gm = vec![0.0; rank * rank];
    let mut rhs = vec![0.0; rank];
    for (j, (&wj, &wzj)) in w.iter().zip(wz).enumerate() {
        if wj == 0.0 {
            continue;
        }
        let x = &factors[j * rank..(j + 1) * rank];
        for a in 0..rank {
            let wa = wj * x[a];
            let row = &mut gm[a * rank..(a + 1) * rank];
            for b in a..rank {
                row[b] += wa * x[b];
            }
            rhs[a] += wzj * x[a];
        }
    }
    for a in 0..rank {
        gm[a * rank + a] += lam;
        for b in 0..a {
            gm[a * rank + b] = gm[b * rank + a];
        }
    }
    linalg::solve(&gm, &rhs, rank).unwrap_or_else(|| vec![0.0; rank])
}

/// `f` (per-dataset feature ridge), `r` (EB residual for anchored datasets
/// around the prior mean: the graph prior where it exists, else `f`), `tau_s`,
/// and `A = prior + r` with `A[reference] = 0`.
#[allow(clippy::too_many_arguments)]
fn update_offsets(
    lay: &Layout,
    params: &BridleParams,
    design: &[Vec<f64>],
    abz: &[f64],
    y: &[f64],
    obs: &[u8],
    theta: &[f64],
    st: &mut State,
) {
    let (n, g_n, s_n, nl, kp) = (lay.n, lay.g_n, lay.s_n, lay.nl, lay.k + 1);
    let st_ref: &State = st;
    // sufficient statistics prec[g,s], abar[g,s]
    let stats: Vec<(Vec<f64>, Vec<f64>)> = (0..g_n)
        .into_par_iter()
        .map(|g| {
            let mut prec = vec![0.0; s_n];
            let mut num = vec![0.0; s_n];
            for i in 0..n {
                if obs[g * n + i] != OBSERVED {
                    continue;
                }
                let s = lay.srow[i];
                let w = 1.0 / st_ref.sig2(abz[g], s);
                let z = y[g * n + i]
                    - theta[g * nl + lay.li[i]]
                    - st_ref.c[i]
                    - st_ref.p[g * kp + lay.prow[i]];
                prec[s] += w;
                num[s] += w * z;
            }
            let abar = num
                .iter()
                .zip(&prec)
                .map(|(a, b)| a / b.max(1e-12))
                .collect();
            (prec, abar)
        })
        .collect();
    let wf = |g: usize, s: usize| -> f64 {
        let prec = stats[g].0[s];
        if prec > 0.0 {
            let tb = if lay.anchored[s] {
                st_ref.taus2[s]
            } else {
                0.0
            };
            1.0 / (tb + 1.0 / prec.max(1e-12))
        } else {
            0.0
        }
    };
    let p = design[0].len();
    let betas: Vec<Option<Vec<f64>>> = (0..s_n)
        .into_par_iter()
        .map(|s| {
            if s == lay.ref_s {
                return None;
            }
            let mut xtx = vec![0.0; p * p];
            let mut xty = vec![0.0; p];
            let mut ok = 0_usize;
            for (g, dg) in design.iter().enumerate() {
                let w = wf(g, s);
                if w <= 0.0 {
                    continue;
                }
                ok += 1;
                let ab = stats[g].1[s];
                for a in 0..p {
                    let wa = w * dg[a];
                    for b in a..p {
                        xtx[a * p + b] += wa * dg[b];
                    }
                    xty[a] += wa * ab;
                }
            }
            if ok < params.min_f_genes {
                return None;
            }
            for a in 0..p {
                if a > 0 {
                    xtx[a * p + a] += params.ridge_f;
                }
                for b in 0..a {
                    xtx[a * p + b] = xtx[b * p + a];
                }
            }
            linalg::solve(&xtx, &xty, p)
        })
        .collect();
    // new tau_s from r and its posterior variance (old tau_s in both)
    let taus2_old = st.taus2.clone();
    let rows: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..g_n)
        .into_par_iter()
        .map(|g| {
            let mut f_g = vec![0.0; s_n];
            let mut a_g = vec![0.0; s_n];
            let mut e_g = vec![0.0; s_n];
            for s in 0..s_n {
                if let Some(beta) = &betas[s] {
                    f_g[s] = design[g].iter().zip(beta).map(|(d, b)| d * b).sum();
                }
                let prec = stats[g].0[s];
                let pa = st_ref.prior_a[g * s_n + s];
                let pm = if pa.is_finite() { pa } else { f_g[s] };
                let mut r = 0.0;
                if prec > 0.0 && lay.anchored[s] {
                    r = (stats[g].1[s] - pm) * prec / (prec + 1.0 / taus2_old[s]);
                    e_g[s] = r * r + 1.0 / (prec + 1.0 / taus2_old[s]);
                }
                a_g[s] = if s == lay.ref_s { 0.0 } else { pm + r };
            }
            (f_g, a_g, e_g)
        })
        .collect();
    let mut sums = vec![0.0; s_n];
    let mut cnts = vec![0_usize; s_n];
    for (g, (f_g, a_g, e_g)) in rows.into_iter().enumerate() {
        st.f[g * s_n..(g + 1) * s_n].copy_from_slice(&f_g);
        st.a[g * s_n..(g + 1) * s_n].copy_from_slice(&a_g);
        for s in 0..s_n {
            if stats[g].0[s] > 0.0 {
                sums[s] += e_g[s];
                cnts[s] += 1;
            }
        }
    }
    for s in 0..s_n {
        if lay.anchored[s] && cnts[s] > 0 {
            st.taus2[s] = sums[s] / cnts[s] as f64;
        }
        st.taus2[s] = st.taus2[s].clamp(TAUS2_RANGE.0, TAUS2_RANGE.1);
        st.f_fitted[s] = betas[s].is_some();
    }
}

/// Plex effects `P[k,g]` (ridge, centred within each dataset) and `tauP`.
fn update_plex(lay: &Layout, abz: &[f64], y: &[f64], obs: &[u8], theta: &[f64], st: &mut State) {
    let (n, s_n, nl, k) = (lay.n, lay.s_n, lay.nl, lay.k);
    let kp = k + 1;
    let tau_p2 = st.tau_p2;
    let plex_datasets: Vec<usize> = lay
        .plex_ds
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let (a, c) = (&st.a, &st.c);
    let (sig2_s, sig_b) = (&st.sig2_s, &st.sig_b);
    let parts: Vec<(f64, usize)> = st
        .p
        .par_chunks_mut(kp)
        .enumerate()
        .map(|(g, p_g)| {
            let mut num = vec![0.0; kp];
            let mut den = vec![0.0; kp];
            for i in 0..n {
                let kk = lay.prow[i];
                if kk == k || obs[g * n + i] != OBSERVED {
                    continue;
                }
                let s = lay.srow[i];
                let w =
                    1.0 / (sig2_s[s] * (sig_b[s] * abz[g]).exp()).clamp(SIG2_RANGE.0, SIG2_RANGE.1);
                num[kk] += w * (y[g * n + i] - theta[g * nl + lay.li[i]] - a[g * s_n + s] - c[i]);
                den[kk] += w;
            }
            for kk in 0..k {
                p_g[kk] = num[kk] / (den[kk] + 1.0 / tau_p2);
            }
            p_g[k] = 0.0;
            for &d in &plex_datasets {
                let (mut sw, mut swp) = (0.0, 0.0);
                for kk in 0..k {
                    if lay.plex_ds[kk] == d {
                        sw += den[kk];
                        swp += den[kk] * p_g[kk];
                    }
                }
                let mean = swp / sw.max(1e-12);
                for (pk, &pd) in p_g.iter_mut().zip(&lay.plex_ds) {
                    if pd == d {
                        *pk -= mean;
                    }
                }
            }
            let (mut sum, mut cnt) = (0.0, 0_usize);
            for kk in 0..k {
                if den[kk] > 0.0 {
                    sum += p_g[kk] * p_g[kk] + 1.0 / (den[kk] + 1.0 / tau_p2);
                    cnt += 1;
                }
            }
            (sum, cnt)
        })
        .collect();
    let (sum, cnt) = parts.iter().fold((0.0, 0), |(s, c), (a, b)| (s + a, c + b));
    if cnt > 0 {
        st.tau_p2 = sum / cnt as f64;
    }
}

/// Per-profile sums over genes of `term(g, i, w)` and of the weights `w`, in
/// fixed gene blocks (deterministic order).
fn row_reduce<F>(lay: &Layout, term: F, abz: &[f64], obs: &[u8], st: &State) -> (Vec<f64>, Vec<f64>)
where
    F: Fn(usize, usize, f64, &State) -> f64 + Sync,
{
    let (n, g_n) = (lay.n, lay.g_n);
    let blocks: Vec<(Vec<f64>, Vec<f64>)> = (0..g_n.div_ceil(GENE_BLOCK))
        .into_par_iter()
        .map(|b| {
            let mut num = vec![0.0; n];
            let mut den = vec![0.0; n];
            for g in b * GENE_BLOCK..((b + 1) * GENE_BLOCK).min(g_n) {
                for i in 0..n {
                    if obs[g * n + i] != OBSERVED {
                        continue;
                    }
                    let w = 1.0 / st.sig2(abz[g], lay.srow[i]);
                    num[i] += term(g, i, w, st);
                    den[i] += w;
                }
            }
            (num, den)
        })
        .collect();
    let mut num = vec![0.0; n];
    let mut den = vec![0.0; n];
    for (bn, bd) in blocks {
        for i in 0..n {
            num[i] += bn[i];
            den[i] += bd[i];
        }
    }
    (num, den)
}

/// Per-dataset noise trend `log sig2[s,g] = a_s + b_s * abz_g` from
/// leverage-corrected residuals of anchor rows; returns the monitor MSE.
#[allow(clippy::too_many_arguments)]
fn update_noise(
    lay: &Layout,
    abz: &[f64],
    y: &[f64],
    obs: &[u8],
    theta: &[f64],
    wl: &[f64],
    gbin: &[usize],
    bin_x: &[f64],
    ds_has_anchor: &[bool],
    st: &mut State,
) -> f64 {
    let (n, g_n, s_n, nl, kp) = (lay.n, lay.g_n, lay.s_n, lay.nl, lay.k + 1);
    let st_ref: &State = st;
    let tau_r2 = st.tau_r2;
    type NoiseBlock = (Vec<f64>, Vec<f64>, f64, usize);
    let blocks: Vec<NoiseBlock> = (0..g_n.div_ceil(GENE_BLOCK))
        .into_par_iter()
        .map(|b| {
            let mut cnt = vec![0.0; s_n * NOISE_BINS];
            let mut sum = vec![0.0; s_n * NOISE_BINS];
            let (mut hs, mut hc) = (0.0, 0_usize);
            for g in b * GENE_BLOCK..((b + 1) * GENE_BLOCK).min(g_n) {
                for i in 0..n {
                    let o = obs[g * n + i];
                    if o == UNOBSERVED {
                        continue;
                    }
                    let s = lay.srow[i];
                    let l = lay.li[i];
                    let mu = theta[g * nl + l]
                        + st_ref.a[g * s_n + s]
                        + st_ref.c[i]
                        + st_ref.p[g * kp + lay.prow[i]];
                    let e2 = (y[g * n + i] - mu).powi(2);
                    if o == HELD {
                        hs += e2;
                        hc += 1;
                        continue;
                    }
                    if !lay.anchor_row[i] {
                        continue;
                    }
                    let w = 1.0 / st_ref.sig2(abz[g], s);
                    let h = (w / (wl[g * nl + l] + 1.0 / tau_r2)).min(0.95);
                    let idx = s * NOISE_BINS + gbin[g];
                    cnt[idx] += 1.0;
                    sum[idx] += e2 / (1.0 - h);
                }
            }
            (cnt, sum, hs, hc)
        })
        .collect();
    let mut cnt = vec![0.0; s_n * NOISE_BINS];
    let mut sum = vec![0.0; s_n * NOISE_BINS];
    let (mut hs, mut hc) = (0.0, 0_usize);
    for (bc, bs, h1, h2) in blocks {
        for j in 0..cnt.len() {
            cnt[j] += bc[j];
            sum[j] += bs[j];
        }
        hs += h1;
        hc += h2;
    }
    for s in 0..s_n {
        if !ds_has_anchor[s] {
            continue;
        }
        let (mut xs, mut ys, mut ws) = (Vec::new(), Vec::new(), Vec::new());
        for b in 0..NOISE_BINS {
            let c = cnt[s * NOISE_BINS + b];
            if c > 50.0 {
                xs.push(bin_x[b]);
                ys.push((sum[s * NOISE_BINS + b] / c).ln());
                ws.push(c);
            }
        }
        if xs.len() >= 3 {
            if let Some((slope, icpt)) = linalg::weighted_line(&xs, &ys, &ws) {
                if slope.is_finite() && icpt.is_finite() {
                    st.sig2_s[s] = icpt.exp();
                    st.sig_b[s] = slope.clamp(-1.0, 1.0);
                }
            }
        }
    }
    // datasets without anchor rows take the (lower) median of the others
    let lower_median = |v: &mut Vec<f64>| -> Option<f64> {
        if v.is_empty() {
            return None;
        }
        v.sort_by(f64::total_cmp);
        Some(v[(v.len() - 1) / 2])
    };
    if ds_has_anchor.iter().any(|&b| !b) {
        let mut a: Vec<f64> = (0..s_n)
            .filter(|&s| ds_has_anchor[s])
            .map(|s| st.sig2_s[s])
            .collect();
        let mut b: Vec<f64> = (0..s_n)
            .filter(|&s| ds_has_anchor[s])
            .map(|s| st.sig_b[s])
            .collect();
        if let (Some(ma), Some(mb)) = (lower_median(&mut a), lower_median(&mut b)) {
            for (s, &has) in ds_has_anchor.iter().enumerate() {
                if !has {
                    st.sig2_s[s] = ma;
                    st.sig_b[s] = mb;
                }
            }
        }
    }
    if hc > 0 {
        hs / hc as f64
    } else {
        f64::NAN
    }
}

/// Gene-major `A[s,g] + c[i] + P[k,g]` on observed cells (0 elsewhere): the
/// part of the output `v = y - A - c - P` that changes between sweeps.
fn output_offsets(lay: &Layout, obs: &[u8], st: &State) -> Vec<f64> {
    let (n, s_n, kp) = (lay.n, lay.s_n, lay.k + 1);
    let mut out = vec![0.0; lay.g_n * n];
    out.par_chunks_mut(n).enumerate().for_each(|(g, o)| {
        for i in 0..n {
            if obs[g * n + i] != UNOBSERVED {
                o[i] = st.a[g * s_n + lay.srow[i]] + st.c[i] + st.p[g * kp + lay.prow[i]];
            }
        }
    });
    out
}

/// Mean |a - b| over observed cells (fixed-size block sums, added in order).
fn mean_abs_change(a: &[f64], b: &[f64], obs: &[u8]) -> f64 {
    let parts: Vec<(f64, usize)> = obs
        .par_chunks(GENE_BLOCK)
        .zip(a.par_chunks(GENE_BLOCK).zip(b.par_chunks(GENE_BLOCK)))
        .map(|(o, (x, y))| {
            o.iter()
                .zip(x.iter().zip(y))
                .filter(|(&o, _)| o != UNOBSERVED)
                .fold((0.0, 0_usize), |(s, c), (_, (x, y))| {
                    (s + (x - y).abs(), c + 1)
                })
        })
        .collect();
    let (s, c) = parts.iter().fold((0.0, 0), |(s, c), (a, b)| (s + a, c + b));
    if c == 0 {
        f64::NAN
    } else {
        s / c as f64
    }
}

fn check_finite(st: &State) -> Result<()> {
    let blocks: [(&str, &[f64]); 8] = [
        ("m", &st.m),
        ("Lin", &st.lin),
        ("U", &st.u),
        ("V", &st.v),
        ("R", &st.r_line),
        ("A", &st.a),
        ("P", &st.p),
        ("c", &st.c),
    ];
    for (name, v) in blocks {
        if v.iter().any(|x| !x.is_finite()) {
            return Err(invalid(format!(
                "BRIDLE fit diverged: non-finite values in {name}"
            )));
        }
    }
    Ok(())
}

/// `v = y - A_out - c - P` on observed cells (row-major `n x G`), with `A_out`
/// cross-fitted for weakly anchored datasets. Returns `(values, cross_fitted)`.
fn output_values(
    lay: &Layout,
    params: &BridleParams,
    abz: &[f64],
    y: &[f64],
    obs: &[u8],
    st: &State,
) -> (Vec<f64>, Vec<bool>) {
    let (n, g_n, s_n, nlin, kp, rank) = (lay.n, lay.g_n, lay.s_n, lay.nlin, lay.k + 1, params.rank);
    let tau_r2 = st.tau_r2;
    // cross-fit plan per dataset: anchor rows, folds of their lines
    struct Plan {
        s: usize,
        rows: Vec<usize>,
        anchor_rows: Vec<usize>,
        lines: Vec<usize>,
        fold_of_line: HashMap<usize, usize>,
    }
    let mut plans = Vec::new();
    let mut cf = vec![false; s_n];
    #[allow(clippy::needless_range_loop)] // s indexes several per-dataset arrays
    for s in 0..s_n {
        if !lay.anchored[s] || lay.nb[s] >= params.cf_max_nb {
            continue;
        }
        let rows: Vec<usize> = (0..n).filter(|&i| lay.srow[i] == s).collect();
        let anchor_rows: Vec<usize> = rows
            .iter()
            .copied()
            .filter(|&i| lay.anchor_row[i])
            .collect();
        let lines: Vec<usize> = anchor_rows
            .iter()
            .map(|&i| lay.li[i])
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let perm = NumpyRandomState::new(u32::try_from(s).unwrap_or(u32::MAX)).permutation(&lines);
        let fold_of_line = perm
            .iter()
            .enumerate()
            .map(|(pos, &l)| (l, pos % params.folds))
            .collect();
        cf[s] = true;
        plans.push(Plan {
            s,
            rows,
            anchor_rows,
            lines,
            fold_of_line,
        });
    }
    let mut out_gm = vec![f64::NAN; g_n * n];
    out_gm.par_chunks_mut(n).enumerate().for_each(|(g, out)| {
        for i in 0..n {
            if obs[g * n + i] != UNOBSERVED {
                out[i] = y[g * n + i]
                    - st.a[g * s_n + lay.srow[i]]
                    - st.c[i]
                    - st.p[g * kp + lay.prow[i]];
            }
        }
        for plan in &plans {
            let s = plan.s;
            let taus2 = st.taus2[s];
            let pm_gs = st.prior_mean(s_n, g, s);
            // leave-dataset-out theta for this dataset's anchor samples
            let mut tloo: HashMap<usize, f64> = HashMap::with_capacity(plan.lines.len());
            for &l in &plan.lines {
                let prior = st.m[g] + st.lin[g * nlin + lay.lin_of_line[l]] + st.uv(rank, g, l);
                let (mut num, mut den) = (0.0, 0.0);
                for &j in &lay.line_rows[l] {
                    let sj = lay.srow[j];
                    if sj == s || obs[g * n + j] != OBSERVED {
                        continue;
                    }
                    let w = 1.0 / st.sig2(abz[g], sj);
                    num += w
                        * (y[g * n + j]
                            - st.a[g * s_n + sj]
                            - st.c[j]
                            - st.p[g * kp + lay.prow[j]]
                            - prior);
                    den += w;
                }
                tloo.insert(l, prior + num / (den + 1.0 / tau_r2));
            }
            // per-fold EB sums of the anchor rows' residual offsets
            let mut fold_num = vec![0.0; params.folds];
            let mut fold_den = vec![0.0; params.folds];
            for &i in &plan.anchor_rows {
                if obs[g * n + i] != OBSERVED {
                    continue;
                }
                let l = lay.li[i];
                let w = 1.0 / st.sig2(abz[g], s);
                let res = y[g * n + i] - tloo[&l] - st.c[i] - pm_gs - st.p[g * kp + lay.prow[i]];
                let fold = plan.fold_of_line[&l];
                fold_num[fold] += w * res;
                fold_den[fold] += w;
            }
            let tot_num: f64 = fold_num.iter().sum();
            let tot_den: f64 = fold_den.iter().sum();
            for &i in &plan.rows {
                if obs[g * n + i] == UNOBSERVED {
                    continue;
                }
                let r = if lay.anchor_row[i] {
                    let fold = plan.fold_of_line[&lay.li[i]];
                    (tot_num - fold_num[fold]) / (tot_den - fold_den[fold] + 1.0 / taus2)
                } else {
                    tot_num / (tot_den + 1.0 / taus2)
                };
                out[i] = y[g * n + i] - (pm_gs + r) - st.c[i] - st.p[g * kp + lay.prow[i]];
            }
        }
    });
    let mut out = vec![f64::NAN; n * g_n];
    for g in 0..g_n {
        for i in 0..n {
            out[i * g_n + g] = out_gm[g * n + i];
        }
    }
    (out, cf)
}

#[cfg(test)]
mod tests;
