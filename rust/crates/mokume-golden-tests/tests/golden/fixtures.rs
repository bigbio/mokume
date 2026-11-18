use super::*;

// ===========================================================================
// features2peptides (standalone command) golden tests vs the Python oracle.
//
// The synthetic feature set mirrors `scratchpad/gen_features.py`, which is the
// exact input fed to:
//   conda run -n Bigbio python -m mokume.mokume_cli features2peptides \
//       -p feat.parquet --skip_normalization -o pep.csv
// (run WITHOUT --sdrf; see the divergence note below). The peptide matrix:
//   P1/APEPTIDECK/run1 = 200, P1/PEPTIDEAK/run1 = 100 (z2) + 150 (z3) = 250,
//   P1/APEPTIDECK/run2 = 400, P1/PEPTIDEAK/run2 = 300.
//   P2/ONEPEPTK is dropped: a single canonical peptide fails min_unique=2.
//
// Divergence (flagged): Python's `peptide_normalization` computes its sample
// list from the *pre-SDRF-enrichment* view, so when an SDRF join would remap
// `sample_accession`, the per-sample filter matches nothing and the output is
// empty -- a latent Python quirk. The faithful, non-empty deterministic path
// is therefore run *without* --sdrf, where both engines key on run_file_name.
// On that path every column matches cell-for-cell, `Condition` included:
// Python's QPX view defaults it to run_file_name and the Rust peptide export
// now applies the same fallback (`sample_condition`, formerly the literal
// "Empty"). Verified on real PXD003539 (no/SDRF-miss): Condition mismatches
// drop from 100% to 0 while NormIntensity/BioReplicate/row set stay aligned.
pub(crate) fn synthetic_peptide_rows() -> Vec<QpxRow<'static>> {
    vec![
        QpxRow::new_with_charge("PEPTIDEAK", "run1", 100.0, 2, &["P1"]),
        QpxRow::new_with_charge("PEPTIDEAK", "run1", 150.0, 3, &["P1"]),
        QpxRow::new_with_charge("APEPTIDECK", "run1", 200.0, 2, &["P1"]),
        QpxRow::new_with_charge("PEPTIDEAK", "run2", 300.0, 2, &["P1"]),
        QpxRow::new_with_charge("APEPTIDECK", "run2", 400.0, 2, &["P1"]),
        QpxRow::new_with_charge("ONEPEPTK", "run1", 999.0, 2, &["P2"]),
    ]
}

pub(crate) fn default_peptides_config(
    parquet: PathBuf,
    output: PathBuf,
) -> FeatureToPeptidesConfig {
    FeatureToPeptidesConfig {
        input: InputConfig {
            parquet: Some(parquet),
            msstats: None,
            psm: None,
            sdrf: None,
            run: None,
            fasta: None,
            transferred_features: TransferredFeatures::Keep,
            transferred_weight: None,
        },
        output,
        filtering: FilterConfig {
            min_aa: 7,
            min_unique_peptides: 2,
            remove_contaminants: false,
        },
        remove_ids: None,
        remove_low_frequency_peptides: false,
        keep_shared_peptides: false,
        skip_normalization: true,
        run_normalization: "median".to_owned(),
        sample_normalization: "globalmedian".to_owned(),
        log2: false,
        save_parquet: false,
        aggregation_level: AggregationLevel::Sample,
        filter_pipeline: None,
        irs: None,
    }
}

pub(crate) fn write_synthetic_qpx(path: &Path) -> Result<(), Box<dyn Error>> {
    let sequence = Arc::new(StringArray::from(vec![
        "PEPTIDEAK",
        "APEPTIDECK",
        "PEPTIDEAK",
        "ALYAAEK",
        "SHORT",
        "CONTAMPEPK",
        "DECOYPEPK",
        "THIDPEAK",
        "ATHIDPECK",
        "PEPTIDEAK",
        "ASHAEDPEPK",
        "GAAAPEAK",
        "AGAAAPECK",
    ])) as ArrayRef;
    let peptidoform = Arc::new(StringArray::from(vec![
        "PEPTIDEAK",
        "APEPTIDECK",
        "PEPTIDEAK",
        "ALYAAEK",
        "SHORT",
        "CONTAMPEPK",
        "DECOYPEPK",
        "THIDPEAK",
        "ATHIDPECK",
        "PEPTIDEAK",
        "ASHAEDPEPK",
        "GAAAPEAK",
        "AGAAAPECK",
    ])) as ArrayRef;
    let charge = Arc::new(Int16Array::from(vec![
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    ])) as ArrayRef;
    let run_file_name = Arc::new(StringArray::from(vec![
        "run1.raw", "run1.raw", "run2.raw", "run1.raw", "run1.raw", "run1.raw", "run1.raw",
        "run1.raw", "run1.raw", "run1.raw", "run1.raw", "run1.raw", "run1.raw",
    ])) as ArrayRef;
    let is_decoy = Arc::new(BooleanArray::from(vec![
        false, false, false, false, false, false, true, false, false, false, false, false, false,
    ])) as ArrayRef;
    let unique = Arc::new(BooleanArray::from(vec![
        true, true, true, true, true, true, true, true, true, true, false, true, true,
    ])) as ArrayRef;
    let intensities = synthetic_intensities()?;
    let proteins = synthetic_proteins()?;
    let intensities_field = Field::new("intensities", intensities.data_type().clone(), true);
    let proteins_field = Field::new("pg_accessions", proteins.data_type().clone(), true);

    let schema = Arc::new(Schema::new(vec![
        Field::new("sequence", DataType::Utf8, false),
        Field::new("peptidoform", DataType::Utf8, false),
        Field::new("charge", DataType::Int16, false),
        Field::new("run_file_name", DataType::Utf8, false),
        Field::new("is_decoy", DataType::Boolean, false),
        Field::new("unique", DataType::Boolean, false),
        intensities_field,
        proteins_field,
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            sequence,
            peptidoform,
            charge,
            run_file_name,
            is_decoy,
            unique,
            intensities,
            proteins,
        ],
    )?;

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

pub(crate) fn synthetic_intensities() -> Result<ArrayRef, Box<dyn Error>> {
    let fields = Fields::from(vec![
        Field::new("label", DataType::Utf8, false),
        Field::new("intensity", DataType::Float32, false),
    ]);
    let struct_builder = StructBuilder::new(
        fields,
        vec![
            Box::new(StringBuilder::new()),
            Box::new(Float32Builder::new()),
        ],
    );
    let mut builder = ListBuilder::new(struct_builder);
    for intensity in [
        100.0_f32, 200.0, 300.0, 500.0, 999.0, 800.0, 50.0, 700.0, 900.0, 150.0, 10_000.0, 10.0,
        20.0,
    ] {
        append_string_field(builder.values(), 0, "label free sample")?;
        append_f32_field(builder.values(), 1, intensity)?;
        builder.values().append(true);
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

pub(crate) fn synthetic_proteins() -> Result<ArrayRef, Box<dyn Error>> {
    let fields = Fields::from(vec![
        Field::new("accession", DataType::Utf8, false),
        Field::new("start", DataType::Int32, true),
        Field::new("end", DataType::Int32, true),
        Field::new("pre", DataType::Utf8, true),
        Field::new("post", DataType::Utf8, true),
    ]);
    let struct_builder = StructBuilder::new(
        fields,
        vec![
            Box::new(StringBuilder::new()),
            Box::new(Int32Builder::new()),
            Box::new(Int32Builder::new()),
            Box::new(StringBuilder::new()),
            Box::new(StringBuilder::new()),
        ],
    );
    let mut builder = ListBuilder::new(struct_builder);
    let rows = vec![
        (vec!["P1"], 1, 8),
        (vec!["P1"], 9, 16),
        (vec!["P1"], 1, 8),
        (vec!["P2"], 1, 8),
        (vec!["P1"], 17, 21),
        (vec!["sp|CONTAMINANT_P00761|TRYP_PIG"], 1, 9),
        (vec!["sp|DECOY_P11111|REV_HUMAN"], 1, 8),
        (vec!["P3"], 1, 8),
        (vec!["P3"], 9, 16),
        (vec!["P1"], 1, 8),
        (vec!["P1"], 22, 30),
        (vec!["sp|P4A|GROUP_A", "tr|P4B|GROUP_B"], 1, 8),
        (vec!["sp|P4A|GROUP_A", "tr|P4B|GROUP_B"], 9, 16),
    ];

    for (accessions, start, end) in rows {
        for accession in accessions {
            append_string_field(builder.values(), 0, accession)?;
            append_i32_field(builder.values(), 1, start)?;
            append_i32_field(builder.values(), 2, end)?;
            append_string_field(builder.values(), 3, "K")?;
            append_string_field(builder.values(), 4, "R")?;
            builder.values().append(true);
        }
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

pub(crate) fn append_string_field(
    builder: &mut StructBuilder,
    index: usize,
    value: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(builder) = builder.field_builder::<StringBuilder>(index) else {
        return Err(format!("missing string field builder {index}").into());
    };
    builder.append_value(value);
    Ok(())
}

pub(crate) fn append_f32_field(
    builder: &mut StructBuilder,
    index: usize,
    value: f32,
) -> Result<(), Box<dyn Error>> {
    let Some(builder) = builder.field_builder::<Float32Builder>(index) else {
        return Err(format!("missing f32 field builder {index}").into());
    };
    builder.append_value(value);
    Ok(())
}

pub(crate) fn append_i32_field(
    builder: &mut StructBuilder,
    index: usize,
    value: i32,
) -> Result<(), Box<dyn Error>> {
    let Some(builder) = builder.field_builder::<Int32Builder>(index) else {
        return Err(format!("missing i32 field builder {index}").into());
    };
    builder.append_value(value);
    Ok(())
}

pub(crate) fn write_synthetic_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tfactor value[cell line]\n",
            "sample-1\trun 1\trun1.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "sample-2\trun 2\trun2.raw\tAC=MS:1002038;NT=label free sample\tB\n",
        ),
    )?;
    Ok(())
}

/// Rewrites a QPX file with an `id_run_file_name` column, one value per row.
pub(crate) fn add_id_run_column(
    path: &Path,
    id_runs: &[Option<&str>],
) -> Result<(), Box<dyn Error>> {
    let batch = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?
        .build()?
        .next()
        .ok_or("empty QPX fixture")??;
    let mut fields = batch.schema().fields().to_vec();
    fields.push(Arc::new(Field::new(
        "id_run_file_name",
        DataType::Utf8,
        true,
    )));
    let mut columns = batch.columns().to_vec();
    columns.push(Arc::new(StringArray::from(id_runs.to_vec())) as ArrayRef);
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let mut writer = ArrowWriter::try_new(File::create(path)?, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

pub(crate) fn write_maxlfq_order_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tfactor value[cell line]\n",
            "Sample 1\trun 1\trun1.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "Sample 2\trun 2\trun2.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "Sample 3\trun 3\trun3.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "Sample 4\trun 4\trun4.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "Sample 5\trun 5\trun5.raw\tAC=MS:1002038;NT=label free sample\tA\n",
        ),
    )?;
    Ok(())
}

pub(crate) fn write_hier_order_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    // Names sort [hsample-a..d]; the parquet emits run-c first, so registration
    // order is [c, a, d, b]. The data-file -> source-name mapping is what makes
    // the two orders disagree.
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tfactor value[cell line]\n",
            "hsample-a\trun a\trun_a.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "hsample-b\trun b\trun_b.raw\tAC=MS:1002038;NT=label free sample\tB\n",
            "hsample-c\trun c\trun_c.raw\tAC=MS:1002038;NT=label free sample\tC\n",
            "hsample-d\trun d\trun_d.raw\tAC=MS:1002038;NT=label free sample\tD\n",
        ),
    )?;
    Ok(())
}

pub(crate) fn write_condition_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tfactor value[cell line]\n",
            "condA_1\tcond a1\tcond_a1.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "condA_2\tcond a2\tcond_a2.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "condB_1\tcond b1\tcond_b1.raw\tAC=MS:1002038;NT=label free sample\tB\n",
            "condB_2\tcond b2\tcond_b2.raw\tAC=MS:1002038;NT=label free sample\tB\n",
        ),
    )?;
    Ok(())
}

pub(crate) fn write_run_replicate_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tcomment[technical replicate]\tfactor value[cell line]\n",
            "sample-run\trun a\trun_a.raw\tAC=MS:1002038;NT=label free sample\t1\tA\n",
            "sample-run\trun b\trun_b.raw\tAC=MS:1002038;NT=label free sample\t2\tA\n",
        ),
    )?;
    Ok(())
}

pub(crate) fn write_irs_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tfactor value[cell line]\tcomment[fraction identifier]\n",
            "plexA_1\tplex a ref\tplex_a_ref.raw\tAC=MS:1002038;NT=label free sample\treference\tplexA\n",
            "plexA_2\tplex a case\tplex_a_case.raw\tAC=MS:1002038;NT=label free sample\tcase\tplexA\n",
            "plexB_1\tplex b ref\tplex_b_ref.raw\tAC=MS:1002038;NT=label free sample\treference\tplexB\n",
            "plexB_2\tplex b case\tplex_b_case.raw\tAC=MS:1002038;NT=label free sample\tcase\tplexB\n",
        ),
    )?;
    Ok(())
}

pub(crate) fn write_imputation_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tfactor value[cell line]\n",
            "sample-1\timp 1\timp1.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "sample-2\timp 2\timp2.raw\tAC=MS:1002038;NT=label free sample\tA\n",
            "sample-3\timp 3\timp3.raw\tAC=MS:1002038;NT=label free sample\tA\n",
        ),
    )?;
    Ok(())
}

pub(crate) fn write_synthetic_fasta(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            ">P1\n",
            "PEPTIDEAKAPEPTIDECKASHAEDPEPK\n",
            ">P2\n",
            "ALYAAEK\n",
            ">P3\n",
            "THIDPEAKATHIDPECK\n",
            ">sp|P4A|GROUP_A\n",
            "GAAAPEAKAGAAAPECK\n",
            ">tr|P4B|GROUP_B\n",
            "GAAAPEAKAGAAAPECK\n",
        ),
    )?;
    Ok(())
}

/// Legacy proportional piBAQ, which these Python-oracle fixtures were written for.
pub(crate) fn legacy_pibaq_config() -> PibaqConfig {
    PibaqConfig {
        shared_mode: PibaqSharedMode::Proportional,
        ..PibaqConfig::default()
    }
}

pub(crate) fn test_pibaq_digest(entries: &[(&str, &[&str])]) -> PibaqDigest {
    let accession_peptides = entries
        .iter()
        .map(|(accession, peptides)| {
            (
                (*accession).to_owned(),
                peptides
                    .iter()
                    .map(|peptide| (*peptide).to_owned())
                    .collect::<HashSet<_>>(),
            )
        })
        .collect::<HashMap<_, _>>();
    PibaqDigest {
        accession_peptides,
        provenance: PibaqDigestProvenance {
            pyopenms_version: "test".to_owned(),
            enzyme: "Trypsin".to_owned(),
            catalog_hash: "test".to_owned(),
            min_aa: 7,
            max_aa: 30,
            missed_cleavages: 0,
        },
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct QpxRow<'a> {
    pub(crate) sequence: &'a str,
    pub(crate) peptidoform: &'a str,
    pub(crate) run_file_name: &'a str,
    pub(crate) intensity: f32,
    pub(crate) charge: i16,
    pub(crate) accessions: &'a [&'a str],
    /// Maps to the parquet `unique` column. `false` marks a shared peptide,
    /// which Python's `SQLFilterBuilder` keeps in the median pre-pass only when
    /// `--keep-shared-peptides` is set (`require_unique = not keep_shared`).
    pub(crate) unique: bool,
    pub(crate) peptide_qvalue: Option<f64>,
    pub(crate) protein_qvalue: Option<f64>,
    pub(crate) score: Option<(&'a str, f64, bool)>,
}

impl<'a> QpxRow<'a> {
    pub(crate) const fn new(
        sequence: &'a str,
        run_file_name: &'a str,
        intensity: f32,
        accessions: &'a [&'a str],
    ) -> Self {
        Self {
            sequence,
            peptidoform: sequence,
            run_file_name,
            intensity,
            charge: 2,
            accessions,
            unique: true,
            peptide_qvalue: None,
            protein_qvalue: None,
            score: None,
        }
    }

    pub(crate) const fn new_with_charge(
        sequence: &'a str,
        run_file_name: &'a str,
        intensity: f32,
        charge: i16,
        accessions: &'a [&'a str],
    ) -> Self {
        Self {
            sequence,
            peptidoform: sequence,
            run_file_name,
            intensity,
            charge,
            accessions,
            unique: true,
            peptide_qvalue: None,
            protein_qvalue: None,
            score: None,
        }
    }

    pub(crate) const fn new_shared(
        sequence: &'a str,
        run_file_name: &'a str,
        intensity: f32,
        accessions: &'a [&'a str],
    ) -> Self {
        Self {
            sequence,
            peptidoform: sequence,
            run_file_name,
            intensity,
            charge: 2,
            accessions,
            unique: false,
            peptide_qvalue: None,
            protein_qvalue: None,
            score: None,
        }
    }

    pub(crate) const fn with_qvalues(mut self, peptide: Option<f64>, protein: Option<f64>) -> Self {
        self.peptide_qvalue = peptide;
        self.protein_qvalue = protein;
        self
    }

    pub(crate) const fn with_score(
        mut self,
        name: &'a str,
        value: f64,
        higher_better: bool,
    ) -> Self {
        self.score = Some((name, value, higher_better));
        self
    }

    pub(crate) const fn with_peptidoform(mut self, peptidoform: &'a str) -> Self {
        self.peptidoform = peptidoform;
        self
    }
}

pub(crate) fn write_qpx_rows(path: &Path, rows: &[QpxRow<'_>]) -> Result<(), Box<dyn Error>> {
    let sequence = Arc::new(StringArray::from(
        rows.iter().map(|row| row.sequence).collect::<Vec<_>>(),
    )) as ArrayRef;
    let peptidoform = Arc::new(StringArray::from(
        rows.iter().map(|row| row.peptidoform).collect::<Vec<_>>(),
    )) as ArrayRef;
    let charge = Arc::new(Int16Array::from(
        rows.iter().map(|row| row.charge).collect::<Vec<_>>(),
    )) as ArrayRef;
    let run_file_name = Arc::new(StringArray::from(
        rows.iter().map(|row| row.run_file_name).collect::<Vec<_>>(),
    )) as ArrayRef;
    let is_decoy = Arc::new(BooleanArray::from(vec![false; rows.len()])) as ArrayRef;
    let unique = Arc::new(BooleanArray::from(
        rows.iter().map(|row| row.unique).collect::<Vec<_>>(),
    )) as ArrayRef;
    let peptide_qvalue = Arc::new(Float64Array::from(
        rows.iter()
            .map(|row| row.peptide_qvalue)
            .collect::<Vec<_>>(),
    )) as ArrayRef;
    let protein_qvalue = Arc::new(Float64Array::from(
        rows.iter()
            .map(|row| row.protein_qvalue)
            .collect::<Vec<_>>(),
    )) as ArrayRef;
    let additional_scores = qpx_row_scores(rows)?;
    let intensities = qpx_row_intensities(rows)?;
    let proteins = qpx_row_proteins(rows)?;
    let intensities_field = Field::new("intensities", intensities.data_type().clone(), true);
    let proteins_field = Field::new("pg_accessions", proteins.data_type().clone(), true);
    let scores_field = Field::new(
        "additional_scores",
        additional_scores.data_type().clone(),
        true,
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("sequence", DataType::Utf8, false),
        Field::new("peptidoform", DataType::Utf8, false),
        Field::new("charge", DataType::Int16, false),
        Field::new("run_file_name", DataType::Utf8, false),
        Field::new("is_decoy", DataType::Boolean, false),
        Field::new("unique", DataType::Boolean, false),
        Field::new("peptide_qvalue", DataType::Float64, true),
        Field::new("pg_global_qvalue", DataType::Float64, true),
        scores_field,
        intensities_field,
        proteins_field,
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            sequence,
            peptidoform,
            charge,
            run_file_name,
            is_decoy,
            unique,
            peptide_qvalue,
            protein_qvalue,
            additional_scores,
            intensities,
            proteins,
        ],
    )?;

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

pub(crate) fn qpx_row_scores(rows: &[QpxRow<'_>]) -> Result<ArrayRef, Box<dyn Error>> {
    let fields = Fields::from(vec![
        Field::new("score_name", DataType::Utf8, false),
        Field::new("score_value", DataType::Float64, false),
        Field::new("higher_better", DataType::Boolean, false),
    ]);
    let struct_builder = StructBuilder::new(
        fields,
        vec![
            Box::new(StringBuilder::new()),
            Box::new(Float64Builder::new()),
            Box::new(BooleanBuilder::new()),
        ],
    );
    let mut builder = ListBuilder::new(struct_builder);
    for row in rows {
        if let Some((name, value, higher_better)) = row.score {
            append_string_field(builder.values(), 0, name)?;
            builder
                .values()
                .field_builder::<Float64Builder>(1)
                .expect("missing score_value builder")
                .append_value(value);
            builder
                .values()
                .field_builder::<BooleanBuilder>(2)
                .expect("missing higher_better builder")
                .append_value(higher_better);
            builder.values().append(true);
        }
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

pub(crate) fn qpx_row_intensities(rows: &[QpxRow<'_>]) -> Result<ArrayRef, Box<dyn Error>> {
    let fields = Fields::from(vec![
        Field::new("label", DataType::Utf8, false),
        Field::new("intensity", DataType::Float32, false),
    ]);
    let struct_builder = StructBuilder::new(
        fields,
        vec![
            Box::new(StringBuilder::new()),
            Box::new(Float32Builder::new()),
        ],
    );
    let mut builder = ListBuilder::new(struct_builder);
    for row in rows {
        append_string_field(builder.values(), 0, "label free sample")?;
        append_f32_field(builder.values(), 1, row.intensity)?;
        builder.values().append(true);
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

pub(crate) fn qpx_row_proteins(rows: &[QpxRow<'_>]) -> Result<ArrayRef, Box<dyn Error>> {
    let fields = Fields::from(vec![
        Field::new("accession", DataType::Utf8, false),
        Field::new("start", DataType::Int32, true),
        Field::new("end", DataType::Int32, true),
        Field::new("pre", DataType::Utf8, true),
        Field::new("post", DataType::Utf8, true),
    ]);
    let struct_builder = StructBuilder::new(
        fields,
        vec![
            Box::new(StringBuilder::new()),
            Box::new(Int32Builder::new()),
            Box::new(Int32Builder::new()),
            Box::new(StringBuilder::new()),
            Box::new(StringBuilder::new()),
        ],
    );
    let mut builder = ListBuilder::new(struct_builder);
    for row in rows {
        for accession in row.accessions {
            append_string_field(builder.values(), 0, accession)?;
            append_i32_field(builder.values(), 1, 1)?;
            append_i32_field(builder.values(), 2, row.sequence.len() as i32)?;
            append_string_field(builder.values(), 3, "K")?;
            append_string_field(builder.values(), 4, "R")?;
            builder.values().append(true);
        }
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

/// One synthetic TMT feature: a peptide in a run carrying several channels, each
/// an `(label, intensity)` pair in the `intensities[]` list. The QPX flattener
/// emits one `QpxFeatureRecord` per channel, so the IRS reference-channel filter
/// (`label == irs_channel`) sees the per-channel `label` directly.
pub(crate) struct TmtRow<'a> {
    pub(crate) sequence: &'a str,
    pub(crate) run_file_name: &'a str,
    pub(crate) channels: &'a [(&'a str, f32)],
    pub(crate) accessions: &'a [&'a str],
}

/// Writer for multi-channel TMT QPX parquets (channel-aware sibling of
/// [`write_qpx_rows`], whose rows are single label-free intensities). The schema
/// matches the new-QPX layout the Python `Feature` loader expects.
pub(crate) fn write_qpx_tmt_rows(path: &Path, rows: &[TmtRow<'_>]) -> Result<(), Box<dyn Error>> {
    let sequence = Arc::new(StringArray::from(
        rows.iter().map(|row| row.sequence).collect::<Vec<_>>(),
    )) as ArrayRef;
    let peptidoform = Arc::new(StringArray::from(
        rows.iter().map(|row| row.sequence).collect::<Vec<_>>(),
    )) as ArrayRef;
    let charge = Arc::new(Int16Array::from(vec![2_i16; rows.len()])) as ArrayRef;
    let run_file_name = Arc::new(StringArray::from(
        rows.iter().map(|row| row.run_file_name).collect::<Vec<_>>(),
    )) as ArrayRef;
    let is_decoy = Arc::new(BooleanArray::from(vec![false; rows.len()])) as ArrayRef;
    let unique = Arc::new(BooleanArray::from(vec![true; rows.len()])) as ArrayRef;

    let intensity_fields = Fields::from(vec![
        Field::new("label", DataType::Utf8, false),
        Field::new("intensity", DataType::Float32, false),
    ]);
    let intensity_struct = StructBuilder::new(
        intensity_fields,
        vec![
            Box::new(StringBuilder::new()),
            Box::new(Float32Builder::new()),
        ],
    );
    let mut intensity_builder = ListBuilder::new(intensity_struct);
    for row in rows {
        for (label, value) in row.channels {
            append_string_field(intensity_builder.values(), 0, label)?;
            append_f32_field(intensity_builder.values(), 1, *value)?;
            intensity_builder.values().append(true);
        }
        intensity_builder.append(true);
    }
    let intensities = Arc::new(intensity_builder.finish()) as ArrayRef;

    let proteins = qpx_tmt_proteins(rows)?;
    let intensities_field = Field::new("intensities", intensities.data_type().clone(), true);
    let proteins_field = Field::new("pg_accessions", proteins.data_type().clone(), true);
    let schema = Arc::new(Schema::new(vec![
        Field::new("sequence", DataType::Utf8, false),
        Field::new("peptidoform", DataType::Utf8, false),
        Field::new("charge", DataType::Int16, false),
        Field::new("run_file_name", DataType::Utf8, false),
        Field::new("is_decoy", DataType::Boolean, false),
        Field::new("unique", DataType::Boolean, false),
        intensities_field,
        proteins_field,
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            sequence,
            peptidoform,
            charge,
            run_file_name,
            is_decoy,
            unique,
            intensities,
            proteins,
        ],
    )?;
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

pub(crate) fn qpx_tmt_proteins(rows: &[TmtRow<'_>]) -> Result<ArrayRef, Box<dyn Error>> {
    let fields = Fields::from(vec![
        Field::new("accession", DataType::Utf8, false),
        Field::new("start", DataType::Int32, true),
        Field::new("end", DataType::Int32, true),
        Field::new("pre", DataType::Utf8, true),
        Field::new("post", DataType::Utf8, true),
    ]);
    let struct_builder = StructBuilder::new(
        fields,
        vec![
            Box::new(StringBuilder::new()),
            Box::new(Int32Builder::new()),
            Box::new(Int32Builder::new()),
            Box::new(StringBuilder::new()),
            Box::new(StringBuilder::new()),
        ],
    );
    let mut builder = ListBuilder::new(struct_builder);
    for row in rows {
        for accession in row.accessions {
            append_string_field(builder.values(), 0, accession)?;
            append_i32_field(builder.values(), 1, 1)?;
            append_i32_field(builder.values(), 2, row.sequence.len() as i32)?;
            append_string_field(builder.values(), 3, "K")?;
            append_string_field(builder.values(), 4, "R")?;
            builder.values().append(true);
        }
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

pub(crate) fn write_ratio_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tfactor value[cell line]\n",
            "plexA_1\tref run\tref.raw\tAC=MS:1002038;NT=label free sample\treference\n",
            "plexA_2\tcase run\tcase.raw\tAC=MS:1002038;NT=label free sample\tcase\n",
        ),
    )?;
    Ok(())
}

pub(crate) fn write_ratio_multiplex_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            "source name\tassay name\tcomment[data file]\tcomment[label]\tcharacteristics[pooled sample]\tfactor value[cell line]\n",
            "p1_1\tp1 ref\tp1_ref.raw\tAC=MS:1002038;NT=label free sample\tpooled\treference\n",
            "p1_2\tp1 a\tp1_a.raw\tAC=MS:1002038;NT=label free sample\tnot pooled\tcase\n",
            "p1_3\tp1 b\tp1_b.raw\tAC=MS:1002038;NT=label free sample\tnot pooled\tcase\n",
            "p2_1\tp2 ref\tp2_ref.raw\tAC=MS:1002038;NT=label free sample\tpooled\treference\n",
            "p2_2\tp2 a\tp2_a.raw\tAC=MS:1002038;NT=label free sample\tnot pooled\tcase\n",
            "p2_3\tp2 b\tp2_b.raw\tAC=MS:1002038;NT=label free sample\tnot pooled\tcase\n",
        ),
    )?;
    Ok(())
}

pub(crate) fn write_family_fasta(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::write(
        path,
        concat!(
            ">P5A\n",
            "ACDEFGKLMNSTYKQASTVWK\n",
            ">P5B\n",
            "ACDEFGKLMNSTYKGHILMVK\n",
        ),
    )?;
    Ok(())
}

/// Per-protein intended per-sample protein-level Sum (linear), columns
/// A1,A2,A3,B1,B2,B3. Mirrors scratchpad/gen_deqms_parquet.py.
pub(crate) const DEQMS_TARGETS: &[(&str, [f32; 6])] = &[
    ("PROT01", [1000.0, 1149.0, 870.0, 4000.0, 4290.0, 3732.0]),
    (
        "PROT02",
        [32000.0, 34000.0, 30000.0, 8000.0, 9200.0, 6800.0],
    ),
    ("PROT03", [256.0, 270.0, 244.0, 257.0, 248.0, 256.0]),
    ("PROT04", [5000.0, 5200.0, 4800.0, 10000.0, 10400.0, 9600.0]),
    ("PROT05", [320.0, 322.0, 318.0, 640.0, 644.0, 636.0]),
    ("PROT06", [512.0, 724.0, 362.0, 480.0, 660.0, 350.0]),
    ("PROT07", [2048.0, 2100.0, 1990.0, 1024.0, 1050.0, 995.0]),
    ("PROT08", [800.0, 820.0, 780.0, 1600.0, 1640.0, 1560.0]),
    ("PROT09", [128.0, 130.0, 126.0, 256.0, 260.0, 252.0]),
    ("PROT10", [6000.0, 6300.0, 5700.0, 3000.0, 3150.0, 2850.0]),
    ("PROT11", [1500.0, 1520.0, 1480.0, 1490.0, 1510.0, 1505.0]),
    ("PROT12", [900.0, 950.0, 850.0, 1800.0, 1900.0, 1700.0]),
];

/// Per-protein number of unique peptide sequences, each seen in every sample (the
/// DEqMS count is this plus 1). Index aligns with `DEQMS_TARGETS`.
pub(crate) const DEQMS_COUNTS: [usize; 12] = [1, 2, 3, 5, 8, 13, 4, 7, 11, 6, 9, 15];

pub(crate) const DEQMS_SAMPLE_RUNS: [&str; 6] =
    ["A1.raw", "A2.raw", "A3.raw", "B1.raw", "B2.raw", "B3.raw"];

/// Write the 12-protein deqms feature parquet. Each protein gets `count` distinct
/// peptide sequences: one anchor carrying `target - (count-1)` and `count-1`
/// extras carrying 1.0, so the Sum rollup per (protein, sample) recovers the
/// integer target. The single `pg_accessions` entry equals the protein name, so
/// the parsed ProteinName == the anchor_protein grouping Python counts over.
pub(crate) fn write_deqms_count_features(path: &Path) -> Result<(), Box<dyn Error>> {
    // Owned storage that outlives the borrowed `QpxRow` slice.
    let mut sequences: Vec<String> = Vec::new();
    let mut accession_storage: Vec<Vec<String>> = Vec::new();
    // (sequence_index, accession_storage_index, run_index, intensity)
    let mut plan: Vec<(usize, usize, usize, f32)> = Vec::new();

    for (protein_index, (protein, targets)) in DEQMS_TARGETS.iter().enumerate() {
        let count = DEQMS_COUNTS[protein_index];
        let accession_index = accession_storage.len();
        accession_storage.push(vec![(*protein).to_owned()]);
        let first_sequence = sequences.len();
        for sequence_number in 0..count {
            sequences.push(format!("{protein}PEPSEQ{sequence_number:03}K"));
        }
        let extras = count - 1;
        for (run_index, target) in targets.iter().enumerate() {
            // Anchor sequence carries target - extras; each extra carries 1.0.
            let anchor_intensity = target - extras as f32;
            plan.push((first_sequence, accession_index, run_index, anchor_intensity));
            for extra in 1..count {
                plan.push((first_sequence + extra, accession_index, run_index, 1.0));
            }
        }
    }

    let accession_refs: Vec<Vec<&str>> = accession_storage
        .iter()
        .map(|group| group.iter().map(String::as_str).collect())
        .collect();
    let rows = plan
        .iter()
        .map(|&(sequence_index, accession_index, run_index, intensity)| {
            QpxRow::new(
                sequences[sequence_index].as_str(),
                DEQMS_SAMPLE_RUNS[run_index],
                intensity,
                accession_refs[accession_index].as_slice(),
            )
        })
        .collect::<Vec<_>>();
    write_qpx_rows(path, &rows)
}

/// Write the 3-vs-3 SDRF for the deqms count fixture (A1..A3 -> groupA, B1..B3 ->
/// groupB), one source name / data file per sample.
pub(crate) fn write_deqms_count_sdrf(path: &Path) -> Result<(), Box<dyn Error>> {
    let mut tsv = String::from(
        "source name\tassay name\tcomment[data file]\tcomment[label]\tfactor value[phenotype]\n",
    );
    for (index, run) in DEQMS_SAMPLE_RUNS.iter().enumerate() {
        let sample = run.trim_end_matches(".raw");
        let condition = if sample.starts_with('A') {
            "groupA"
        } else {
            "groupB"
        };
        tsv.push_str(&format!(
            "{sample}\trun {index}\t{run}\tAC=MS:1002038;NT=label free sample\t{condition}\n"
        ));
    }
    std::fs::write(path, tsv)?;
    Ok(())
}
