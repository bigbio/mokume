use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use csv::{ReaderBuilder, StringRecord, Trim};
use mokume_core::{MokumeError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SdrfRecord {
    pub sample_accession: String,
    pub run_accession: Option<String>,
    pub data_file: String,
    pub file_uri: Option<String>,
    pub label: Option<String>,
    pub fraction: Option<String>,
    pub biological_replicate: Option<u32>,
    pub technical_replicate: Option<u32>,
    pub condition: Option<String>,
    /// Raw value of `characteristics[pooled sample]` (SDRF-Proteomics spec). The
    /// ratio reference detection treats `pooled` (or an `SN=...` member list) as a
    /// reference channel, mirroring Python's `detect_pooled_from_sdrf`.
    pub pooled_sample: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct SdrfTable {
    records: Vec<SdrfRecord>,
    // Both indexes hold positions in `records`; they are built once in `from_records`.
    by_run: HashMap<String, Vec<usize>>,
    by_run_label: HashMap<(String, String), usize>,
}

impl SdrfTable {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(MokumeError::MissingInput { path });
        }

        let file = File::open(&path).map_err(|source| MokumeError::Io {
            path: path.clone(),
            source,
        })?;
        Self::from_reader(file)
    }

    pub fn from_reader(reader: impl Read) -> Result<Self> {
        let mut reader = ReaderBuilder::new()
            .delimiter(b'\t')
            .flexible(true)
            .trim(Trim::All)
            .from_reader(reader);
        let headers = reader
            .headers()
            .map_err(|source| invalid_input(format!("failed to read SDRF header: {source}")))?
            .clone();
        let columns = SdrfColumns::from_headers(&headers)?;

        let mut records = Vec::new();
        for record in reader.records() {
            let record = record
                .map_err(|source| invalid_input(format!("failed to read SDRF row: {source}")))?;
            records.push(columns.to_record(&record)?);
        }

        Self::from_records(records)
    }

    pub fn from_records(records: Vec<SdrfRecord>) -> Result<Self> {
        let record_count = records.len();
        let mut table = Self {
            records,
            by_run: HashMap::with_capacity(record_count),
            by_run_label: HashMap::with_capacity(record_count),
        };
        table.rebuild_index()?;
        Ok(table)
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn records(&self) -> &[SdrfRecord] {
        &self.records
    }

    /// Resolve an MSstats run/reference alias to its authoritative SDRF data file.
    /// Ambiguous aliases are intentionally treated as unresolved.
    pub fn resolve_run_file(&self, alias: &str) -> Option<String> {
        let indices = self.by_run.get(&normalize_file_key(alias))?;
        let mut data_file: Option<&str> = None;
        for index in indices {
            let candidate = self.records[*index].data_file.as_str();
            if data_file.is_some_and(|known| known != candidate) {
                return None;
            }
            data_file = Some(candidate);
        }
        data_file.map(ToOwned::to_owned)
    }

    pub fn lookup(&self, run_file_name: &str, label: Option<&str>) -> Result<&SdrfRecord> {
        let run_key = normalize_file_key(run_file_name);
        if let Some(label) = label {
            let label_key = normalize_label_key(label);
            if let Some(index) = self.by_run_label.get(&(run_key.clone(), label_key)) {
                return Ok(&self.records[*index]);
            }
            return match self.by_run.get(&run_key) {
                Some(indices) => Err(self.unmatched_label(run_file_name, label, indices)),
                None => Err(invalid_input(format!(
                    "QPX run `{run_file_name}` and label `{label}` have no exact SDRF match; normalized run has no SDRF records"
                ))),
            };
        }

        match self.by_run.get(&run_key).map(Vec::as_slice) {
            None => Err(invalid_input(format!(
                "QPX run `{run_file_name}` has no SDRF match"
            ))),
            Some([index]) => {
                let record = &self.records[*index];
                let label_key = record.label.as_deref().map(normalize_label_key);
                if label_key
                    .as_deref()
                    .is_some_and(|label| !label.is_empty() && label != "label free sample")
                {
                    return Err(invalid_input(format!(
                        "QPX run `{run_file_name}` has no reporter label; {} declares a labeled channel",
                        self.record_context(&[*index])
                    )));
                }
                Ok(record)
            }
            Some(indices) => Err(invalid_input(format!(
                "SDRF run `{run_file_name}` is ambiguous across {}",
                self.record_context(indices)
            ))),
        }
    }

    fn rebuild_index(&mut self) -> Result<()> {
        self.by_run.clear();
        self.by_run_label.clear();

        // `comment[data file]` is the row's identity: two rows claiming the
        // same file under the same label really are a duplicate mapping, and
        // nothing downstream could pick between them. Index those first so an
        // alias can never take the slot belonging to a real data file.
        for (index, record) in self.records.iter().enumerate() {
            for key in record.run_keys() {
                self.by_run.entry(key).or_default().push(index);
            }
            let Some(label) = record.label.as_deref() else {
                continue;
            };
            let key = normalize_file_key(&record.data_file);
            if key.is_empty() {
                continue;
            }
            let label_key = normalize_label_key(label);
            if let Some(previous) = self
                .by_run_label
                .insert((key.clone(), label_key.clone()), index)
            {
                let records = self.record_context(&[previous, index]);
                return Err(invalid_input(format!(
                    "duplicate SDRF mapping for normalized run `{key}` and label `{label_key}` at {records}"
                )));
            }
        }

        // `comment[file uri]` and `assay name` are extra names the same row can
        // be found under, not identities. Real SDRFs reuse an assay name across
        // distinct runs -- this repository's own PXD063291 fixture repeats
        // `run 2`, `run 4` and `run 6` over six different raw files -- so a
        // collision here is ordinary data, not a mapping error, and rejecting
        // it would refuse SDRFs whose data-file join is perfectly unambiguous.
        // Keep the first row, which is what the index did before duplicates
        // became an error.
        for (index, record) in self.records.iter().enumerate() {
            let Some(label) = record.label.as_deref() else {
                continue;
            };
            let label_key = normalize_label_key(label);
            let data_file_key = normalize_file_key(&record.data_file);
            for key in record.run_keys() {
                if key == data_file_key {
                    continue;
                }
                self.by_run_label
                    .entry((key, label_key.clone()))
                    .or_insert(index);
            }
        }
        Ok(())
    }

    fn unmatched_label(&self, run_file_name: &str, label: &str, indices: &[usize]) -> MokumeError {
        invalid_input(format!(
            "QPX run `{run_file_name}` and label `{label}` have no exact SDRF match; the run occurs in {}",
            self.record_context(indices)
        ))
    }

    fn record_context(&self, indices: &[usize]) -> String {
        indices
            .iter()
            .map(|index| {
                let record = &self.records[*index];
                format!(
                    "record {} (`{}`, label `{}`)",
                    index + 1,
                    record.data_file,
                    record.label.as_deref().unwrap_or("")
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Raw SDRF rows for covariate / explicit-batch-column extraction, mirroring
/// Python's `load_sdrf` (`pd.read_csv(sep="\t")` then lowercasing every header).
/// The typed [`SdrfTable`] only keeps the spec columns it needs for run/label
/// lookup, but batch correction reads arbitrary `characteristics[*]` /
/// `factor value[*]` / custom batch columns by name, so it parses the file
/// through this verbatim view instead. Headers are lowercased and trimmed to
/// match Python's column addressing; cell values are kept case-sensitive
/// (factorize encodes by value identity).
#[derive(Debug, Clone, Default)]
pub struct SdrfRawTable {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl SdrfRawTable {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(MokumeError::MissingInput { path });
        }
        let file = File::open(&path).map_err(|source| MokumeError::Io {
            path: path.clone(),
            source,
        })?;
        Self::from_reader(file)
    }

    pub fn from_reader(reader: impl Read) -> Result<Self> {
        let mut reader = ReaderBuilder::new()
            .delimiter(b'\t')
            .flexible(true)
            .trim(Trim::All)
            .from_reader(reader);
        let headers = reader
            .headers()
            .map_err(|source| invalid_input(format!("failed to read SDRF header: {source}")))?
            .iter()
            .map(normalize_header)
            .collect();
        let mut rows = Vec::new();
        for record in reader.records() {
            let record = record
                .map_err(|source| invalid_input(format!("failed to read SDRF row: {source}")))?;
            rows.push(record.iter().map(ToOwned::to_owned).collect());
        }
        Ok(Self { headers, rows })
    }

    pub fn headers(&self) -> &[String] {
        &self.headers
    }

    /// Index of the column whose lowercased header equals `name` (which the
    /// caller lowercases), or `None` when absent.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.headers.iter().position(|header| header == name)
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Cell at `row`/`col`, or `""` when the (flexible) row is short, matching
    /// how a missing trailing SDRF cell reads as empty.
    pub fn cell(&self, row: usize, col: usize) -> &str {
        self.rows
            .get(row)
            .and_then(|cells| cells.get(col))
            .map_or("", String::as_str)
    }
}

impl SdrfRecord {
    fn run_keys(&self) -> Vec<String> {
        let mut keys = Vec::with_capacity(3);
        push_key(&mut keys, &self.data_file);
        if let Some(file_uri) = self.file_uri.as_deref() {
            push_key(&mut keys, file_uri);
        }
        if let Some(run_accession) = self.run_accession.as_deref() {
            push_key(&mut keys, run_accession);
        }
        keys
    }
}

#[derive(Debug, Clone)]
struct SdrfColumns {
    source_name: Option<usize>,
    assay_name: Option<usize>,
    data_file: Option<usize>,
    file_uri: Option<usize>,
    label: Option<usize>,
    fraction: Option<usize>,
    biological_replicate: Option<usize>,
    technical_replicate: Option<usize>,
    condition: Option<usize>,
    pooled_sample: Option<usize>,
}

impl SdrfColumns {
    fn from_headers(headers: &StringRecord) -> Result<Self> {
        let data_file = header_index(
            headers,
            &["comment[data file]", "comment[raw file]", "data file"],
        );
        let file_uri = header_index(
            headers,
            &["comment[file uri]", "comment[associated file uri]"],
        );
        if data_file.is_none() && file_uri.is_none() {
            return Err(invalid_input(
                "SDRF must contain `comment[data file]` or `comment[file uri]`",
            ));
        }

        Ok(Self {
            source_name: header_index(headers, &["source name", "sample accession"]),
            assay_name: header_index(headers, &["assay name", "run accession"]),
            data_file,
            file_uri,
            label: header_index(headers, &["comment[label]", "label"]),
            fraction: header_index(headers, &["comment[fraction identifier]", "fraction"]),
            biological_replicate: header_index(
                headers,
                &[
                    "characteristics[biological replicate]",
                    "comment[biological replicate]",
                    "biological replicate",
                ],
            ),
            technical_replicate: header_index(
                headers,
                &[
                    "comment[technical replicate]",
                    "technical replicate",
                    "technical_replica",
                ],
            ),
            condition: first_factor_value(headers).or_else(|| {
                header_index(
                    headers,
                    &[
                        "factor value[cell line]",
                        "characteristics[organism part]",
                        "characteristics[cell line]",
                        "characteristics[disease]",
                    ],
                )
            }),
            pooled_sample: header_index(headers, &["characteristics[pooled sample]"]),
        })
    }

    fn to_record(&self, record: &StringRecord) -> Result<SdrfRecord> {
        let file_uri = optional_cell(record, self.file_uri).map(ToOwned::to_owned);
        let data_file = optional_cell(record, self.data_file)
            .map(ToOwned::to_owned)
            .or_else(|| file_uri.as_deref().map(basename).map(ToOwned::to_owned))
            .ok_or_else(|| invalid_input("SDRF row has no data file"))?;
        let sample_accession = optional_cell(record, self.source_name)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| data_file.clone());

        Ok(SdrfRecord {
            sample_accession,
            run_accession: optional_cell(record, self.assay_name).map(ToOwned::to_owned),
            data_file,
            file_uri,
            label: optional_cell(record, self.label).map(extract_label_value),
            fraction: optional_cell(record, self.fraction).map(ToOwned::to_owned),
            biological_replicate: optional_cell(record, self.biological_replicate)
                .and_then(parse_positive_u32),
            technical_replicate: optional_cell(record, self.technical_replicate)
                .and_then(parse_positive_u32),
            condition: optional_cell(record, self.condition).map(ToOwned::to_owned),
            pooled_sample: optional_cell(record, self.pooled_sample).map(ToOwned::to_owned),
        })
    }
}

fn push_key(keys: &mut Vec<String>, value: &str) {
    let key = normalize_file_key(value);
    if !key.is_empty() && !keys.iter().any(|known| known == &key) {
        keys.push(key);
    }
}

/// Memoizes [`SdrfTable::lookup`] by the raw QPX run and label. A QPX stream
/// repeats a few runs across millions of rows, so each distinct pair is
/// normalized and resolved once. Failed lookups are not cached, so they keep
/// returning `lookup`'s error.
#[derive(Debug)]
pub struct SdrfLookupCache<'a> {
    table: &'a SdrfTable,
    resolved: HashMap<String, Vec<(Option<String>, &'a SdrfRecord)>>,
}

impl<'a> SdrfLookupCache<'a> {
    pub fn new(table: &'a SdrfTable) -> Self {
        Self {
            table,
            resolved: HashMap::new(),
        }
    }

    pub fn lookup(&mut self, run_file_name: &str, label: Option<&str>) -> Result<&'a SdrfRecord> {
        let cached = self.resolved.get(run_file_name).and_then(|labels| {
            labels
                .iter()
                .find(|(known, _)| known.as_deref() == label)
                .map(|(_, record)| *record)
        });
        if let Some(record) = cached {
            return Ok(record);
        }
        let record = self.table.lookup(run_file_name, label)?;
        self.resolved
            .entry(run_file_name.to_owned())
            .or_default()
            .push((label.map(str::to_owned), record));
        Ok(record)
    }
}

pub fn normalize_file_key(value: &str) -> String {
    let key = basename(value)
        .trim()
        .trim_start_matches("file://")
        .to_ascii_lowercase();
    // Strip a trailing run extension so the SDRF `comment[data file]` key matches
    // the QPX `run_file_name`, which is stored extension-less. Without this the
    // per-feature SDRF lookup misses and the Condition column falls back to the
    // run filename. `mokume-pipeline` de.rs keys run-level DE columns the same way.
    let last_extension = [".raw", ".mzml", ".wiff", ".d"]
        .into_iter()
        .flat_map(|extension| {
            key.match_indices(extension)
                .map(move |(index, _)| (index, extension.len()))
        })
        .filter_map(|(index, length)| {
            let suffix = &key[index + length..];
            (suffix.is_empty()
                || suffix.starts_with('.')
                || suffix.starts_with('_')
                || suffix.starts_with(char::is_whitespace))
            .then_some(index)
        })
        .max();
    if let Some(index) = last_extension {
        return key[..index].to_owned();
    }
    key.strip_suffix(".scan").unwrap_or(&key).to_owned()
}

pub fn normalize_label_key(value: &str) -> String {
    let key = extract_label_value(value).trim().to_ascii_lowercase();
    match key.as_str() {
        "lfq" | "label-free" | "label free sample" | "raw" => "label free sample".to_owned(),
        _ => key,
    }
}

fn extract_label_value(value: &str) -> String {
    let trimmed = value.trim();
    for part in trimmed.split(';') {
        let part = part.trim();
        if let Some(value) = part.strip_prefix("NT=") {
            return value.trim().to_owned();
        }
    }
    trimmed.to_owned()
}

fn basename(value: &str) -> &str {
    let trimmed = value.trim();
    let without_forward = trimmed
        .rsplit_once('/')
        .map_or(trimmed, |(_, basename)| basename);
    without_forward
        .rsplit_once('\\')
        .map_or(without_forward, |(_, basename)| basename)
}

fn header_index(headers: &StringRecord, candidates: &[&str]) -> Option<usize> {
    headers.iter().enumerate().find_map(|(index, header)| {
        let header = normalize_header(header);
        candidates
            .iter()
            .any(|candidate| header == normalize_header(candidate))
            .then_some(index)
    })
}

fn first_factor_value(headers: &StringRecord) -> Option<usize> {
    headers.iter().enumerate().find_map(|(index, header)| {
        normalize_header(header)
            .starts_with("factor value[")
            .then_some(index)
    })
}

fn normalize_header(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

fn optional_cell(record: &StringRecord, index: Option<usize>) -> Option<&str> {
    index
        .and_then(|index| record.get(index))
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn parse_positive_u32(value: &str) -> Option<u32> {
    value.trim().parse::<u32>().ok()
}

fn invalid_input(message: impl Into<String>) -> MokumeError {
    MokumeError::InvalidInput {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sdrf_and_uses_label_aware_lookup() -> Result<()> {
        let input = concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tcomment[fraction identifier]\tcomment[technical replicate]\tfactor value[cell line]\n",
            "sample-1\trun 1\trun1.raw\tAC=MS:1002038;NT=label free sample\t1\t2\tHeLa\n",
            "sample-2\trun 2\trun2.raw\tTMT126\t1\t1\tK562\n",
        );

        let table = SdrfTable::from_reader(input.as_bytes())?;
        let label_free = table.lookup("/tmp/run1.raw", Some("label free sample"))?;
        let tmt = table.lookup("run2.raw", Some("TMT126"))?;

        assert_eq!(table.len(), 2);
        assert_eq!(label_free.sample_accession, "sample-1");
        assert_eq!(label_free.condition.as_deref(), Some("HeLa"));
        assert_eq!(label_free.technical_replicate, Some(2));
        assert_eq!(tmt.sample_accession, "sample-2");
        Ok(())
    }

    #[test]
    fn raw_table_lowercases_headers_and_addresses_cells_by_name() -> Result<()> {
        let input = concat!(
            "Source Name\tcharacteristics[sex]\tbatch\n",
            "S1\tmale\tb1\n",
            "S2\tfemale\tb2\n",
        );
        let raw = SdrfRawTable::from_reader(input.as_bytes())?;

        // Headers are lowercased (Python's `load_sdrf`); cells stay verbatim.
        assert_eq!(
            raw.headers(),
            ["source name", "characteristics[sex]", "batch"]
        );
        assert_eq!(raw.column_index("source name"), Some(0));
        assert_eq!(raw.column_index("Source Name"), None);
        assert_eq!(raw.column_index("missing"), None);
        assert_eq!(raw.row_count(), 2);

        let sex = raw
            .column_index("characteristics[sex]")
            .ok_or_else(|| invalid_input("missing sex column"))?;
        assert_eq!(raw.cell(0, sex), "male");
        assert_eq!(raw.cell(1, sex), "female");
        // Out-of-range row/column reads as empty (short flexible rows).
        assert_eq!(raw.cell(9, sex), "");
        Ok(())
    }

    #[test]
    fn parses_pooled_sample_column() -> Result<()> {
        let input = concat!(
            "source name\tcomment[data file]\tcharacteristics[pooled sample]\n",
            "p1_1\tp1_ref.raw\tpooled\n",
            "p1_2\tp1_a.raw\tnot pooled\n",
        );

        let table = SdrfTable::from_reader(input.as_bytes())?;
        let pooled = table.lookup("p1_ref.raw", None)?;
        let other = table.lookup("p1_a.raw", None)?;

        assert_eq!(pooled.pooled_sample.as_deref(), Some("pooled"));
        assert_eq!(other.pooled_sample.as_deref(), Some("not pooled"));
        Ok(())
    }

    #[test]
    fn normalize_file_key_strips_run_extensions() {
        // The QPX `run_file_name` is stored extension-less while the SDRF
        // `comment[data file]` carries a run extension; both must normalize to
        // the same key so the per-feature lookup matches.
        for name in [
            "Run_Condition_A_01.raw",
            "Run_Condition_A_01.mzML",
            "Run_Condition_A_01.d",
            "Run_Condition_A_01.wiff",
            "/data/Run_Condition_A_01.raw",
            "Run_Condition_A_01.mzML_controllerType=0 controllerNumber=1 scan=42",
            "Run_Condition_A_01",
        ] {
            assert_eq!(
                normalize_file_key(name),
                "run_condition_a_01",
                "key for {name}"
            );
        }
        for label in [
            "LFQ",
            "label-free",
            "AC=MS:1002038;NT=label free sample",
            "raw",
        ] {
            assert_eq!(normalize_label_key(label), "label free sample");
        }
        assert_eq!(normalize_file_key("foo.raw.bar.raw"), "foo.raw.bar");
        assert_eq!(normalize_label_key(" TMT127N "), "tmt127n");
    }

    #[test]
    fn lookup_matches_extensionless_run_key() -> Result<()> {
        // Regression: SDRF data files carry `.raw`, but the QPX run key is
        // extension-less. Without extension stripping the lookup misses and the
        // Condition column falls back to the run filename.
        let input = concat!(
            "source name\tcomment[data file]\tfactor value[group]\n",
            "A1\tRun_Condition_A_01.raw\tA\n",
            "B1\tRun_Condition_B_01.raw\tB\n",
        );
        let table = SdrfTable::from_reader(input.as_bytes())?;
        let a = table.lookup("Run_Condition_A_01", None)?;
        assert_eq!(a.condition.as_deref(), Some("A"));
        Ok(())
    }

    #[test]
    fn rejects_duplicate_normalized_run_label_keys() -> Result<()> {
        for rows in [
            [
                "sample-a\trun.raw\tTMT126\n",
                "sample-b\tRUN.mzML\ttmt126\n",
            ],
            [
                "sample-b\tRUN.mzML\ttmt126\n",
                "sample-a\trun.raw\tTMT126\n",
            ],
        ] {
            let input = format!(
                "source name\tcomment[data file]\tcomment[label]\n{}{}",
                rows[0], rows[1]
            );

            let error = SdrfTable::from_reader(input.as_bytes())
                .err()
                .ok_or_else(|| invalid_input("duplicate mapping unexpectedly succeeded"))?;
            let message = error.to_string();

            assert!(message.contains("duplicate SDRF mapping"));
            assert!(message.contains("run"));
            assert!(message.contains("tmt126"));
            assert!(message.contains("record 1"));
            assert!(message.contains("record 2"));
            assert!(message.contains("run.raw"));
            assert!(message.contains("RUN.mzML"));
        }
        Ok(())
    }

    #[test]
    fn accepts_reused_assay_names_across_distinct_data_files() -> Result<()> {
        // Real PRIDE SDRFs reuse an `assay name` across runs; this repository's
        // own PXD063291 fixture repeats `run 2`, `run 4` and `run 6` over six
        // different raw files, all label free. The data-file join is completely
        // unambiguous there, so the table has to load.
        let input = concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\n",
            "sample-a\trun 2\tSet12_A2.raw\tAC=MS:1002038;NT=label free sample\n",
            "sample-b\trun 2\tSet12_B1.raw\tAC=MS:1002038;NT=label free sample\n",
        );

        let table = SdrfTable::from_reader(input.as_bytes())?;

        // Each row is still reachable through its own data file.
        assert_eq!(
            table.lookup("Set12_A2.raw", None)?.sample_accession,
            "sample-a"
        );
        assert_eq!(table.lookup("Set12_B1", None)?.sample_accession, "sample-b");
        Ok(())
    }

    #[test]
    fn reused_assay_name_is_ambiguous_only_when_looked_up_by_that_name() -> Result<()> {
        // Keeping the alias out of the duplicate check must not turn a genuinely
        // ambiguous lookup into a silent first-wins answer: asking for the
        // shared assay name itself still has no single right answer.
        let input = concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\n",
            "sample-a\trun 2\tSet12_A2.raw\tAC=MS:1002038;NT=label free sample\n",
            "sample-b\trun 2\tSet12_B1.raw\tAC=MS:1002038;NT=label free sample\n",
        );

        let table = SdrfTable::from_reader(input.as_bytes())?;
        let message = table
            .lookup("run 2", None)
            .err()
            .ok_or_else(|| invalid_input("ambiguous assay-name lookup unexpectedly succeeded"))?
            .to_string();

        assert!(message.contains("ambiguous"), "{message}");
        Ok(())
    }

    #[test]
    fn rejects_unmatched_or_ambiguous_channel_lookup() -> Result<()> {
        let input = concat!(
            "source name\tcomment[data file]\tcomment[label]\n",
            "sample-126\tplex.raw\tTMT126\n",
            "sample-127n\tplex.raw\tTMT127N\n",
        );
        let table = SdrfTable::from_reader(input.as_bytes())?;

        let unmatched = table
            .lookup("PLEX.mzML", Some("TMT129"))
            .err()
            .ok_or_else(|| invalid_input("unknown channel unexpectedly matched"))?
            .to_string();
        let ambiguous = table
            .lookup("plex", None)
            .err()
            .ok_or_else(|| invalid_input("run-only lookup unexpectedly matched"))?
            .to_string();

        assert!(unmatched.contains("QPX run `PLEX.mzML` and label `TMT129`"));
        assert!(unmatched.contains("record 1 (`plex.raw`, label `TMT126`)"));
        assert!(unmatched.contains("record 2 (`plex.raw`, label `TMT127N`)"));
        assert!(ambiguous.contains("SDRF run `plex` is ambiguous"));
        Ok(())
    }

    #[test]
    fn rejects_missing_reporter_label_but_accepts_label_free_run() -> Result<()> {
        let tmt = SdrfTable::from_reader(
            concat!(
                "source name\tcomment[data file]\tcomment[label]\n",
                "sample-126\tplex.raw\tTMT126\n",
            )
            .as_bytes(),
        )?;
        let lfq = SdrfTable::from_reader(
            concat!(
                "source name\tcomment[data file]\tcomment[label]\n",
                "sample-lfq\trun.raw\tLabel free sample\n",
            )
            .as_bytes(),
        )?;

        let error = tmt
            .lookup("plex", None)
            .err()
            .ok_or_else(|| invalid_input("missing reporter label unexpectedly matched"))?;

        assert!(error.to_string().contains("has no reporter label"));
        assert_eq!(lfq.lookup("run", None)?.sample_accession, "sample-lfq");
        assert_eq!(
            lfq.lookup("run", Some("raw"))?.sample_accession,
            "sample-lfq"
        );
        Ok(())
    }
}
