//! `correct-batches --method bridle`: BRIDLE (Batch Removal via Intrinsic
//! Detectability and Latent Estimation) integration of a multi-dataset
//! collection (see [`mokume_stats::batch::bridle`] for the model). `--method
//! lim` is accepted as a hidden, deprecated alias.
//!
//! Input: one long-format table (`.parquet`, `.tsv` or `.csv`) with dataset,
//! anchor, gene and log2 value columns (one row per observed cell; replicates
//! already collapsed per (dataset, anchor)). The anchor column
//! (`--anchor-column`, alias `--line-column`) names the biological unit that
//! links datasets: a cell line, reference material, pooled QC or the same
//! patient across cohorts. Profiles are the (dataset, anchor) pairs, sorted by
//! name; genes are sorted by name. Missing cells stay missing: no imputation,
//! no protein filter.
//!
//! Optional inputs:
//! * `--lineage-table` (e.g. DepMap `Model.csv`, `RRID` -> `OncotreeLineage`):
//!   anchors without a match get no lineage effect.
//! * `--plex-column` (+ `--plex-table`): explicit TMT plex / mixture ids per
//!   profile, e.g. derived from the SDRF. When absent, plexes are inferred from
//!   shared missingness (Jaccard + average linkage, datasets with >= 20
//!   profiles), exactly as the prototype. `--no-plex` disables the block.
//! * `--fasta`: technical sequence features for the offset model.
//!
//! Output (`-o`, parquet when the extension is `.parquet`, else TSV/CSV):
//! `<dataset>, <anchor>, <gene>, <value>, imputed` for every observed input
//! cell, with `value = y - A - c - P` and `imputed = false`. `--theta-output`
//! writes the pooled per-anchor biology for observed anchor/gene cells,
//! `--report` a JSON fit summary.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, Float64Array, StringArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use mokume_core::{MokumeError, Result};
use mokume_stats::batch::bridle::{
    bridle_fit, build_design, reference_abundance, sequence_features, BridleData, BridleParams,
    BridleResult, PlexMode, StopRule,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::other_args::{BridleArgs, BridleStopRule};
use crate::CorrectBatchesArgs;

/// Rows per written parquet batch.
const WRITE_BATCH: usize = 1 << 20;

fn invalid(message: impl Into<String>) -> MokumeError {
    MokumeError::InvalidInput {
        message: message.into(),
    }
}

fn io_err(path: &Path, source: std::io::Error) -> MokumeError {
    MokumeError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Default BRIDLE options (as parsed by clap with no BRIDLE flags).
pub(crate) fn default_bridle_args() -> BridleArgs {
    BridleArgs {
        dataset_column: "ds".to_owned(),
        anchor_column: "cvcl".to_owned(),
        gene_column: "gene".to_owned(),
        value_column: "v".to_owned(),
        reference: None,
        lineage_table: None,
        lineage_key_column: "RRID".to_owned(),
        lineage_column: "OncotreeLineage".to_owned(),
        plex_column: None,
        plex_table: None,
        no_plex: false,
        fasta: None,
        fasta_organism: None,
        rank: 16,
        sweeps: 400,
        stop_rule: BridleStopRule::Output,
        stop_tol: None,
        min_sweeps: None,
        seed: 0,
        anchor_scale: false,
        theta_output: None,
        report: None,
    }
}

/// ComBat must not silently ignore BRIDLE-only options.
pub(crate) fn reject_bridle_only_options(bridle: &BridleArgs) -> Result<()> {
    let d = default_bridle_args();
    let set = [
        (
            "--dataset-column",
            bridle.dataset_column != d.dataset_column,
        ),
        ("--anchor-column", bridle.anchor_column != d.anchor_column),
        ("--gene-column", bridle.gene_column != d.gene_column),
        ("--value-column", bridle.value_column != d.value_column),
        ("--reference", bridle.reference.is_some()),
        ("--lineage-table", bridle.lineage_table.is_some()),
        (
            "--lineage-key-column",
            bridle.lineage_key_column != d.lineage_key_column,
        ),
        (
            "--lineage-column",
            bridle.lineage_column != d.lineage_column,
        ),
        ("--plex-column", bridle.plex_column.is_some()),
        ("--plex-table", bridle.plex_table.is_some()),
        ("--no-plex", bridle.no_plex),
        ("--fasta", bridle.fasta.is_some()),
        ("--fasta-organism", bridle.fasta_organism.is_some()),
        ("--rank", bridle.rank != d.rank),
        ("--sweeps", bridle.sweeps != d.sweeps),
        ("--stop-rule", bridle.stop_rule != d.stop_rule),
        ("--stop-tol", bridle.stop_tol.is_some()),
        ("--min-sweeps", bridle.min_sweeps.is_some()),
        ("--seed", bridle.seed != d.seed),
        ("--anchor-scale", bridle.anchor_scale),
        ("--theta-output", bridle.theta_output.is_some()),
        ("--report", bridle.report.is_some()),
    ];
    let bad: Vec<&str> = set.iter().filter(|(_, on)| *on).map(|(n, _)| *n).collect();
    if bad.is_empty() {
        Ok(())
    } else {
        Err(invalid(format!(
            "{} only apply to --method bridle",
            bad.join(", ")
        )))
    }
}

/// String interner (ids in first-seen order).
#[derive(Default)]
struct Interner {
    ids: HashMap<String, u32>,
    names: Vec<String>,
}

impl Interner {
    fn id(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.ids.get(s) {
            return id;
        }
        let id = self.names.len() as u32;
        self.ids.insert(s.to_owned(), id);
        self.names.push(s.to_owned());
        id
    }
}

/// Long table with interned keys.
#[derive(Default)]
struct LongInput {
    ds: Interner,
    line: Interner,
    gene: Interner,
    plex: Interner,
    /// (ds, line, gene, value, plex id or u32::MAX)
    rows: Vec<(u32, u32, u32, f64, u32)>,
    n_missing_value: usize,
}

const NO_ID: u32 = u32::MAX;

fn is_parquet(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("parquet"))
}

fn delimiter_for(path: &Path) -> u8 {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("csv"))
    {
        b','
    } else {
        b'\t'
    }
}

fn parse_float(raw: &str) -> Option<f64> {
    let t = raw.trim();
    if t.is_empty() {
        return None;
    }
    t.parse::<f64>().ok().filter(|v| v.is_finite())
}

fn read_long(path: &Path, bridle: &BridleArgs, plex_in_input: Option<&str>) -> Result<LongInput> {
    let mut out = LongInput::default();
    let mut cols = vec![
        bridle.dataset_column.as_str(),
        bridle.anchor_column.as_str(),
        bridle.gene_column.as_str(),
        bridle.value_column.as_str(),
    ];
    if let Some(p) = plex_in_input {
        cols.push(p);
    }
    let push = |out: &mut LongInput, fields: [Option<&str>; 4], value: Option<f64>| -> Result<()> {
        let (Some(d), Some(l), Some(g)) = (fields[0], fields[1], fields[2]) else {
            return Err(invalid("null dataset / line / gene in the BRIDLE input"));
        };
        let Some(v) = value else {
            out.n_missing_value += 1;
            return Ok(());
        };
        let plex = fields[3]
            .filter(|p| !p.trim().is_empty())
            .map_or(NO_ID, |p| out.plex.id(p.trim()));
        let row = (out.ds.id(d), out.line.id(l), out.gene.id(g), v, plex);
        out.rows.push(row);
        Ok(())
    };
    if is_parquet(path) {
        let file = File::open(path).map_err(|e| io_err(path, e))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
        let schema = Arc::clone(builder.schema());
        let idx: Vec<usize> = cols
            .iter()
            .map(|c| {
                schema
                    .index_of(c)
                    .map_err(|_| invalid(format!("column '{c}' not found in {}", path.display())))
            })
            .collect::<Result<_>>()?;
        let reader = builder
            .build()
            .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
        for batch in reader {
            let batch = batch.map_err(|e| invalid(format!("{}: {e}", path.display())))?;
            let to_str = |j: usize| -> Result<ArrayRef> {
                cast(batch.column(idx[j]), &DataType::Utf8)
                    .map_err(|e| invalid(format!("column '{}': {e}", cols[j])))
            };
            let (d, l, g) = (to_str(0)?, to_str(1)?, to_str(2)?);
            let p = if cols.len() > 4 {
                Some(to_str(4)?)
            } else {
                None
            };
            let v = cast(batch.column(idx[3]), &DataType::Float64)
                .map_err(|e| invalid(format!("column '{}': {e}", cols[3])))?;
            let as_s = |a: &ArrayRef| a.as_any().downcast_ref::<StringArray>().cloned();
            let (Some(d), Some(l), Some(g)) = (as_s(&d), as_s(&l), as_s(&g)) else {
                return Err(invalid("string cast failed"));
            };
            let p = p.as_ref().and_then(as_s);
            let Some(v) = v.as_any().downcast_ref::<Float64Array>() else {
                return Err(invalid("float cast failed"));
            };
            fn get(a: &StringArray, r: usize) -> Option<&str> {
                (!a.is_null(r)).then(|| a.value(r))
            }
            for r in 0..batch.num_rows() {
                let value = (!v.is_null(r))
                    .then(|| v.value(r))
                    .filter(|x| x.is_finite());
                let plex = p.as_ref().and_then(|a| get(a, r));
                push(&mut out, [get(&d, r), get(&l, r), get(&g, r), plex], value)?;
            }
        }
    } else {
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(delimiter_for(path))
            .from_path(path)
            .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
        let headers = reader
            .headers()
            .map_err(|e| invalid(format!("{}: {e}", path.display())))?
            .clone();
        let idx: Vec<usize> =
            cols.iter()
                .map(|c| {
                    headers.iter().position(|h| h == *c).ok_or_else(|| {
                        invalid(format!("column '{c}' not found in {}", path.display()))
                    })
                })
                .collect::<Result<_>>()?;
        for rec in reader.records() {
            let rec = rec.map_err(|e| invalid(format!("{}: {e}", path.display())))?;
            let f = |j: usize| rec.get(idx[j]);
            let plex = if cols.len() > 4 { f(4) } else { None };
            push(
                &mut out,
                [f(0), f(1), f(2), plex],
                f(3).and_then(parse_float),
            )?;
        }
    }
    Ok(out)
}

/// Read a two-or-three key table (`.csv`/`.tsv`) into `key -> value`.
fn read_map(
    path: &Path,
    key_cols: &[&str],
    value_col: &str,
) -> Result<HashMap<Vec<String>, String>> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter_for(path))
        .flexible(true)
        .from_path(path)
        .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
    let headers = reader
        .headers()
        .map_err(|e| invalid(format!("{}: {e}", path.display())))?
        .clone();
    let pos = |c: &str| {
        headers
            .iter()
            .position(|h| h == c)
            .ok_or_else(|| invalid(format!("column '{c}' not found in {}", path.display())))
    };
    let kidx: Vec<usize> = key_cols.iter().map(|c| pos(c)).collect::<Result<_>>()?;
    let vidx = pos(value_col)?;
    let mut out = HashMap::new();
    for rec in reader.records() {
        let rec = rec.map_err(|e| invalid(format!("{}: {e}", path.display())))?;
        let key: Vec<String> = kidx
            .iter()
            .map(|&i| rec.get(i).unwrap_or("").trim().to_owned())
            .collect();
        let val = rec.get(vidx).unwrap_or("").trim().to_owned();
        if key.iter().any(String::is_empty) || val.is_empty() {
            continue;
        }
        // first occurrence wins (pandas drop_duplicates)
        out.entry(key).or_insert(val);
    }
    Ok(out)
}

/// Entry point for `correct-batches --method bridle`.
pub(crate) fn run_bridle(args: &CorrectBatchesArgs) -> Result<()> {
    let bridle = &args.bridle;
    if !args.input.is_file() {
        return Err(MokumeError::MissingInput {
            path: args.input.clone(),
        });
    }
    if args.export_anndata {
        return Err(invalid(
            "--export-anndata is not supported with --method bridle",
        ));
    }
    for out in std::iter::once(&args.output)
        .chain(bridle.theta_output.iter())
        .chain(bridle.report.iter())
    {
        if out == &args.input {
            return Err(invalid(format!(
                "output {} would overwrite the input",
                out.display()
            )));
        }
    }
    if bridle.plex_table.is_some() && bridle.plex_column.is_none() {
        return Err(invalid("--plex-table needs --plex-column"));
    }
    if bridle.no_plex && bridle.plex_column.is_some() {
        return Err(invalid("--no-plex conflicts with --plex-column"));
    }
    let t0 = std::time::Instant::now();
    let plex_in_input = if bridle.plex_table.is_none() {
        bridle.plex_column.as_deref()
    } else {
        None
    };
    let long = read_long(&args.input, bridle, plex_in_input)?;
    tracing::info!(
        "BRIDLE input: {} observed cells, {} missing/non-finite values skipped",
        long.rows.len(),
        long.n_missing_value
    );

    // profiles sorted by (dataset, line) name, genes by name (pandas pivot order)
    let mut profile_keys: Vec<(u32, u32)> = long.rows.iter().map(|r| (r.0, r.1)).collect();
    profile_keys.sort_unstable();
    profile_keys.dedup();
    profile_keys.sort_by(|a, b| {
        (&long.ds.names[a.0 as usize], &long.line.names[a.1 as usize])
            .cmp(&(&long.ds.names[b.0 as usize], &long.line.names[b.1 as usize]))
    });
    let profile_pos: HashMap<(u32, u32), usize> = profile_keys
        .iter()
        .enumerate()
        .map(|(i, k)| (*k, i))
        .collect();
    let mut gene_order: Vec<u32> = (0..long.gene.names.len() as u32).collect();
    gene_order.sort_by(|a, b| long.gene.names[*a as usize].cmp(&long.gene.names[*b as usize]));
    let mut gene_pos = vec![0_usize; gene_order.len()];
    for (pos, &g) in gene_order.iter().enumerate() {
        gene_pos[g as usize] = pos;
    }
    let (n, g_n) = (profile_keys.len(), gene_order.len());
    let mut values = vec![f64::NAN; n * g_n];
    let mut profile_plex: Vec<u32> = vec![NO_ID; n];
    let mut dup = 0_usize;
    for &(d, l, g, v, p) in &long.rows {
        let i = profile_pos[&(d, l)];
        let cell = &mut values[i * g_n + gene_pos[g as usize]];
        if cell.is_nan() {
            *cell = v;
        } else {
            dup += 1;
        }
        if p != NO_ID {
            if profile_plex[i] == NO_ID {
                profile_plex[i] = p;
            } else if profile_plex[i] != p {
                return Err(invalid(format!(
                    "profile ({}, {}) has more than one plex id",
                    long.ds.names[d as usize], long.line.names[l as usize]
                )));
            }
        }
    }
    if dup > 0 {
        tracing::warn!(
            "BRIDLE input: {dup} duplicate (dataset, line, gene) rows; the first value was kept"
        );
    }
    let datasets: Vec<String> = profile_keys
        .iter()
        .map(|k| long.ds.names[k.0 as usize].clone())
        .collect();
    let lines: Vec<String> = profile_keys
        .iter()
        .map(|k| long.line.names[k.1 as usize].clone())
        .collect();
    let genes: Vec<String> = gene_order
        .iter()
        .map(|&g| long.gene.names[g as usize].clone())
        .collect();

    let lineages: Vec<Option<String>> = match &bridle.lineage_table {
        Some(path) => {
            let map = read_map(
                path,
                &[bridle.lineage_key_column.as_str()],
                &bridle.lineage_column,
            )?;
            lines
                .iter()
                .map(|l| map.get(std::slice::from_ref(l)).cloned())
                .collect()
        }
        None => vec![None; n],
    };
    let plexes: Option<Vec<Option<String>>> = match (&bridle.plex_column, &bridle.plex_table) {
        (Some(col), Some(path)) => {
            let map = read_map(
                path,
                &[
                    bridle.dataset_column.as_str(),
                    bridle.anchor_column.as_str(),
                ],
                col,
            )?;
            Some(
                (0..n)
                    .map(|i| {
                        map.get(&vec![datasets[i].clone(), lines[i].clone()])
                            .cloned()
                    })
                    .collect(),
            )
        }
        (Some(_), None) => Some(
            profile_plex
                .iter()
                .map(|&p| (p != NO_ID).then(|| long.plex.names[p as usize].clone()))
                .collect(),
        ),
        _ => None,
    };
    drop(long);

    let reference = match &bridle.reference {
        Some(r) => r.clone(),
        None => {
            let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
            for d in &datasets {
                *counts.entry(d.as_str()).or_insert(0) += 1;
            }
            let best = counts.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)));
            best.map(|(d, _)| (*d).to_owned()).unwrap_or_default()
        }
    };
    let plex_mode = if bridle.no_plex {
        PlexMode::Off
    } else if plexes.is_some() {
        PlexMode::Explicit
    } else {
        PlexMode::Inferred
    };
    let n_lineage = lineages.iter().filter(|l| l.is_some()).count();
    tracing::info!(
        "BRIDLE: {n} profiles x {g_n} genes, reference={reference}, {n_lineage} profiles with lineage, plex mode={plex_mode:?}"
    );
    let data = BridleData {
        datasets,
        lines,
        lineages,
        plexes,
        genes,
        values,
    };

    let features = match &bridle.fasta {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| io_err(path, e))?;
            let f = sequence_features(&text, &data.genes, bridle.fasta_organism.as_deref());
            tracing::info!(
                "BRIDLE features: {} of {g_n} genes without a FASTA sequence",
                f.n_missing
            );
            Some(f)
        }
        None => None,
    };
    let abundance = reference_abundance(&data, &reference);
    let (design, design_names, abz) = build_design(&abundance, features.as_ref());
    let (stop_rule, min_sweeps, converge_tol) = stop_settings(bridle);
    let params = BridleParams {
        reference,
        rank: bridle.rank,
        sweeps: bridle.sweeps,
        seed: bridle.seed,
        plex_mode,
        anchor_scale: bridle.anchor_scale,
        stop_rule,
        min_sweeps,
        converge_tol,
        ..BridleParams::default()
    };
    let res = bridle_fit(&data, &design, &abz, &params)?;
    tracing::info!(
        "BRIDLE fit: {} sweeps (converged={}) in {:.1}s",
        res.report.history.len(),
        res.report.converged,
        t0.elapsed().as_secs_f64()
    );

    write_values(&args.output, bridle, &data, &res)?;
    if let Some(path) = &bridle.theta_output {
        write_theta(path, bridle, &data, &res)?;
    }
    if let Some(path) = &bridle.report {
        write_report(path, args, &design_names, &res, t0.elapsed().as_secs_f64())?;
    }
    Ok(())
}

/// Effective stop rule, minimum sweeps and tolerance (rule-specific defaults).
fn stop_settings(bridle: &BridleArgs) -> (StopRule, usize, f64) {
    let defaults = match bridle.stop_rule {
        BridleStopRule::Monitor => BridleParams::default().monitor_stop(),
        _ => BridleParams::default(),
    };
    let rule = match bridle.stop_rule {
        BridleStopRule::Output => StopRule::OutputChange,
        BridleStopRule::Monitor => StopRule::MonitorMse,
        BridleStopRule::None => StopRule::Never,
    };
    (
        rule,
        bridle.min_sweeps.unwrap_or(defaults.min_sweeps),
        bridle.stop_tol.unwrap_or(defaults.converge_tol),
    )
}

/// Column-oriented output table.
struct OutTable {
    names: Vec<String>,
    strings: Vec<Vec<String>>,
    values: Vec<f64>,
}

fn write_table(path: &Path, t: &OutTable) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| io_err(parent, e))?;
    }
    let n = t.values.len();
    if is_parquet(path) {
        let mut fields: Vec<Field> = t.names[..t.strings.len()]
            .iter()
            .map(|c| Field::new(c.as_str(), DataType::Utf8, false))
            .collect();
        fields.push(Field::new(
            t.names[t.strings.len()].as_str(),
            DataType::Float64,
            false,
        ));
        fields.push(Field::new("imputed", DataType::Boolean, false));
        let schema = Arc::new(Schema::new(fields));
        let file = File::create(path).map_err(|e| io_err(path, e))?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props))
            .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
        let mut start = 0;
        while start < n {
            let end = (start + WRITE_BATCH).min(n);
            let mut cols: Vec<ArrayRef> = t
                .strings
                .iter()
                .map(|c| Arc::new(StringArray::from_iter_values(c[start..end].iter())) as ArrayRef)
                .collect();
            cols.push(Arc::new(Float64Array::from(t.values[start..end].to_vec())));
            cols.push(Arc::new(BooleanArray::from(vec![false; end - start])));
            let batch = RecordBatch::try_new(Arc::clone(&schema), cols)
                .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
            writer
                .write(&batch)
                .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
            start = end;
        }
        writer
            .close()
            .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
    } else {
        let file = File::create(path).map_err(|e| io_err(path, e))?;
        let mut w = csv::WriterBuilder::new()
            .delimiter(delimiter_for(path))
            .from_writer(BufWriter::new(file));
        let mut header: Vec<&str> = t.names.iter().map(String::as_str).collect();
        header.push("imputed");
        w.write_record(&header)
            .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
        for r in 0..n {
            let v = t.values[r].to_string();
            let mut rec: Vec<&str> = t.strings.iter().map(|c| c[r].as_str()).collect();
            rec.push(&v);
            rec.push("False");
            w.write_record(&rec)
                .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
        }
        w.flush().map_err(|e| io_err(path, e))?;
    }
    Ok(())
}

fn write_values(
    path: &Path,
    bridle: &BridleArgs,
    data: &BridleData,
    res: &BridleResult,
) -> Result<()> {
    let g_n = data.genes.len();
    let mut t = OutTable {
        names: vec![
            bridle.dataset_column.clone(),
            bridle.anchor_column.clone(),
            bridle.gene_column.clone(),
            bridle.value_column.clone(),
        ],
        strings: vec![Vec::new(), Vec::new(), Vec::new()],
        values: Vec::new(),
    };
    for i in 0..data.n_profiles() {
        for g in 0..g_n {
            let v = res.corrected[i * g_n + g];
            if data.values[i * g_n + g].is_finite() && v.is_finite() {
                t.strings[0].push(data.datasets[i].clone());
                t.strings[1].push(data.lines[i].clone());
                t.strings[2].push(data.genes[g].clone());
                t.values.push(v);
            }
        }
    }
    tracing::info!(
        "BRIDLE: writing {} corrected cells to {}",
        t.values.len(),
        path.display()
    );
    write_table(path, &t)
}

fn write_theta(
    path: &Path,
    bridle: &BridleArgs,
    data: &BridleData,
    res: &BridleResult,
) -> Result<()> {
    let g_n = data.genes.len();
    let mut t = OutTable {
        names: vec![
            bridle.anchor_column.clone(),
            bridle.gene_column.clone(),
            bridle.value_column.clone(),
        ],
        strings: vec![Vec::new(), Vec::new()],
        values: Vec::new(),
    };
    for (l, name) in res.line_names.iter().enumerate() {
        for g in 0..g_n {
            if res.theta_observed[l * g_n + g] {
                t.strings[0].push(name.clone());
                t.strings[1].push(data.genes[g].clone());
                t.values.push(res.theta[l * g_n + g]);
            }
        }
    }
    write_table(path, &t)
}

fn write_report(
    path: &Path,
    args: &CorrectBatchesArgs,
    design_names: &[String],
    res: &BridleResult,
    runtime_s: f64,
) -> Result<()> {
    let r = &res.report;
    let datasets: Vec<serde_json::Value> = r
        .datasets
        .iter()
        .map(|d| {
            serde_json::json!({
                "name": d.name, "n_profiles": d.n_profiles, "n_anchor_samples": d.n_anchor_samples,
                "anchored": d.anchored, "cross_fitted": d.cross_fitted, "n_plexes": d.n_plexes,
                "n_plexed_profiles": d.n_plexed_profiles, "tau_s": d.tau_s, "sig2": d.sig2,
                "sig_slope": d.sig_slope, "f_fitted": d.f_fitted,
            })
        })
        .collect();
    let anchor_scale: serde_json::Map<String, serde_json::Value> = r
        .anchor_scale
        .iter()
        .map(|a| {
            (
                a.name.clone(),
                serde_json::json!({"b": a.b, "applied": a.applied, "n_shared_anchors": a.n_shared, "n_genes": a.n_genes}),
            )
        })
        .collect();
    let history: Vec<serde_json::Value> = r
        .history
        .iter()
        .map(|h| serde_json::json!({"sweep": h.sweep, "hold_mse": h.hold_mse, "out_change": h.out_change, "tau_r": h.tau_r, "tau_p": h.tau_p}))
        .collect();
    let bridle = &args.bridle;
    let stop = stop_settings(bridle);
    let mut doc = serde_json::json!({
        "method": "bridle-linear",
        "mokume_version": env!("CARGO_PKG_VERSION"),
        "input": args.input.display().to_string(),
        "params": {
            "reference": r.reference, "rank": bridle.rank, "sweeps": bridle.sweeps, "seed": bridle.seed,
            "stop_rule": format!("{:?}", stop.0), "min_sweeps": stop.1, "stop_tol": stop.2,
            "fasta": bridle.fasta.as_ref().map(|p| p.display().to_string()),
            "fasta_organism": bridle.fasta_organism,
            "lineage_table": bridle.lineage_table.as_ref().map(|p| p.display().to_string()),
            "plex_column": bridle.plex_column, "plex_table": bridle.plex_table.as_ref().map(|p| p.display().to_string()),
            "no_plex": bridle.no_plex, "anchor_scale": bridle.anchor_scale,
        },
        "n_profiles": r.n_profiles, "n_genes": r.n_genes, "n_lines": r.n_lines,
        "n_lineages": r.n_lineages, "n_plexes": r.n_plexes, "design_columns": design_names,
        "observed_cells": r.observed_cells, "holdout_cells": r.holdout_cells,
        "converged": r.converged, "sample_loading_sd": r.sample_loading_sd,
        "runtime_s": runtime_s, "datasets": datasets, "history": history,
    });
    if bridle.anchor_scale {
        doc["anchor_scale"] = serde_json::Value::Object(anchor_scale);
    }
    let text = serde_json::to_string_pretty(&doc).map_err(|e| invalid(e.to_string()))?;
    std::fs::write(path, text).map_err(|e| io_err(path, e))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::other_args::CorrectBatchesMethod;

    type TestResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

    fn temp_dir(tag: &str) -> TestResult<PathBuf> {
        let dir = std::env::temp_dir().join(format!("mokume-bridle-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    fn args(input: PathBuf, output: PathBuf) -> CorrectBatchesArgs {
        CorrectBatchesArgs {
            method: CorrectBatchesMethod::Bridle,
            input,
            pattern: "*pibaq.tsv".to_owned(),
            comment: "#".to_owned(),
            sep: "\t".to_owned(),
            output,
            sample_id_column: "SampleID".to_owned(),
            protein_id_column: "ProteinName".to_owned(),
            pibaq_raw_column: "PiBAQ".to_owned(),
            pibaq_corrected_column: "PiBAQBec".to_owned(),
            export_anndata: false,
            bridle: BridleArgs {
                rank: 2,
                sweeps: 10,
                ..default_bridle_args()
            },
        }
    }

    /// Two datasets sharing 4 lines (B = REF + 1) plus a plex column; 30 genes,
    /// a few missing cells.
    fn write_toy(dir: &Path) -> TestResult<PathBuf> {
        let path = dir.join("long.tsv");
        let mut s = String::from("ds\tcvcl\tgene\tv\tplex\n");
        for (d, off) in [("REF", 0.0), ("B", 1.0)] {
            for l in 0..4 {
                for g in 0..30 {
                    if (l + g) % 11 == 3 && d == "B" {
                        continue; // missing
                    }
                    let v = 20.0 + (g as f64) * 0.1 + (l as f64) * 0.05 * ((g % 3) as f64) + off;
                    s.push_str(&format!(
                        "{d}\tL{l}\tG{g:02}\t{v}\t{}\n",
                        if d == "B" {
                            format!("m{}", l % 2)
                        } else {
                            String::new()
                        }
                    ));
                }
            }
        }
        std::fs::write(&path, s)?;
        Ok(path)
    }

    #[test]
    fn bridle_cli_writes_observed_cells_only() -> TestResult<()> {
        let dir = temp_dir("cli")?;
        let input = write_toy(&dir)?;
        let out = dir.join("values.parquet");
        let mut a = args(input.clone(), out.clone());
        a.bridle.theta_output = Some(dir.join("theta.tsv"));
        a.bridle.report = Some(dir.join("fit.json"));
        a.bridle.lineage_table = None;
        run_bridle(&a)?;
        let file = File::open(&out)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
        let mut rows = 0;
        for batch in reader {
            let batch = batch?;
            assert_eq!(batch.schema().field(0).name(), "ds");
            assert_eq!(batch.schema().field(4).name(), "imputed");
            rows += batch.num_rows();
        }
        let n_input = std::fs::read_to_string(&input)?.lines().count() - 1;
        assert_eq!(rows, n_input, "one output row per observed input cell");
        let report: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("fit.json"))?)?;
        assert_eq!(report["params"]["reference"], "B"); // tie on profiles -> alphabetical
        assert!(
            std::fs::read_to_string(dir.join("theta.tsv"))?
                .lines()
                .count()
                > 1
        );
        Ok(())
    }

    #[test]
    fn bridle_cli_reads_explicit_plexes_from_table() -> TestResult<()> {
        let dir = temp_dir("plex")?;
        let input = write_toy(&dir)?;
        let table = dir.join("plex.tsv");
        std::fs::write(
            &table,
            "ds\tcvcl\tmix\nB\tL0\tx\nB\tL1\tx\nB\tL2\ty\nB\tL3\ty\n",
        )?;
        let mut a = args(input, dir.join("v.tsv"));
        a.bridle.reference = Some("REF".to_owned());
        a.bridle.plex_column = Some("mix".to_owned());
        a.bridle.plex_table = Some(table);
        a.bridle.report = Some(dir.join("fit.json"));
        run_bridle(&a)?;
        let report: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("fit.json"))?)?;
        assert_eq!(report["n_plexes"], 2);
        Ok(())
    }

    #[test]
    fn bridle_cli_stop_rule_defaults_and_overrides() -> TestResult<()> {
        let d = default_bridle_args();
        assert_eq!((d.sweeps, d.stop_rule), (400, BridleStopRule::Output));
        assert_eq!(stop_settings(&d), (StopRule::OutputChange, 200, 1e-5));
        let monitor = BridleArgs {
            stop_rule: BridleStopRule::Monitor,
            ..default_bridle_args()
        };
        assert_eq!(stop_settings(&monitor), (StopRule::MonitorMse, 8, 2e-4));
        let dir = temp_dir("stop")?;
        let input = write_toy(&dir)?;
        let mut a = args(input, dir.join("v.tsv"));
        a.bridle.report = Some(dir.join("fit.json"));
        a.bridle.min_sweeps = Some(3);
        a.bridle.stop_tol = Some(1e9);
        run_bridle(&a)?;
        let report: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("fit.json"))?)?;
        assert_eq!(report["params"]["stop_rule"], "OutputChange");
        assert_eq!(report["converged"], true);
        assert_eq!(report["history"].as_array().map(Vec::len), Some(3));
        assert!(report["history"][2]["out_change"].is_number());
        Ok(())
    }

    #[test]
    fn bridle_cli_anchor_scale_is_reported() -> TestResult<()> {
        let dir = temp_dir("scale")?;
        let input = write_toy(&dir)?;
        let mut a = args(input.clone(), dir.join("v.tsv"));
        a.bridle.reference = Some("REF".to_owned());
        a.bridle.report = Some(dir.join("fit.json"));
        run_bridle(&a)?;
        let off: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("fit.json"))?)?;
        assert!(off.get("anchor_scale").is_none());
        let off_values = std::fs::read_to_string(dir.join("v.tsv"))?;
        a.bridle.anchor_scale = true;
        run_bridle(&a)?;
        let on: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("fit.json"))?)?;
        assert_eq!(on["params"]["anchor_scale"], true);
        // B shares 4 anchors with REF (< 20): b = 1, not applied, same output
        assert_eq!(on["anchor_scale"]["B"]["b"], 1.0);
        assert_eq!(on["anchor_scale"]["B"]["applied"], false);
        assert_eq!(on["anchor_scale"]["B"]["n_shared_anchors"], 4);
        assert_eq!(std::fs::read_to_string(dir.join("v.tsv"))?, off_values);
        Ok(())
    }

    #[test]
    fn bridle_only_options_are_rejected_for_combat() {
        let mut bridle = default_bridle_args();
        assert!(reject_bridle_only_options(&bridle).is_ok());
        bridle.fasta = Some(PathBuf::from("x.fasta"));
        bridle.rank = 4;
        let Err(e) = reject_bridle_only_options(&bridle) else {
            panic!("combat accepted BRIDLE options");
        };
        assert!(e.to_string().contains("--fasta, --rank"));
    }

    #[test]
    fn bridle_cli_rejects_output_over_input() -> TestResult<()> {
        let dir = temp_dir("clash")?;
        let input = write_toy(&dir)?;
        let a = args(input.clone(), input);
        assert!(run_bridle(&a).is_err());
        Ok(())
    }

    fn parse_correct_batches(extra: &[&str]) -> CorrectBatchesArgs {
        use clap::Parser;
        let argv = ["mokume", "correct-batches", "-i", "in", "-o", "out"];
        let cli = crate::Cli::parse_from(argv.iter().chain(extra.iter()));
        let crate::Commands::CorrectBatches(a) = cli.command else {
            panic!("expected the correct-batches subcommand");
        };
        *a
    }

    #[test]
    fn deprecated_lim_and_line_column_aliases_still_parse() {
        let a = parse_correct_batches(&["--method", "bridle", "--anchor-column", "sample"]);
        assert_eq!(a.method, CorrectBatchesMethod::Bridle);
        assert_eq!(a.bridle.anchor_column, "sample");
        let a = parse_correct_batches(&["--method", "lim", "--line-column", "cvcl2"]);
        assert_eq!(a.method, CorrectBatchesMethod::Lim);
        assert_eq!(a.bridle.anchor_column, "cvcl2");
    }

    #[test]
    fn help_shows_bridle_and_hides_deprecated_aliases() {
        use clap::CommandFactory;
        let mut cmd = crate::Cli::command();
        let Some(sub) = cmd.find_subcommand_mut("correct-batches") else {
            panic!("missing correct-batches subcommand");
        };
        let help = sub.render_long_help().to_string();
        assert!(help.contains("bridle"), "{help}");
        assert!(help.contains("--anchor-column"), "{help}");
        assert!(!help.contains("--line-column"), "{help}");
        assert!(!help.contains("lim:") && !help.contains("[possible values: combat, bridle, lim]"));
    }
}
