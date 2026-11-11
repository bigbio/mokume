use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::Path;

use mokume_core::{MokumeError, Result};
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder};
use parquet::arrow::ProjectionMask;
use rayon::prelude::*;

use crate::qpx::list_string_values;

/// Preferred observed member order for each unordered protein group.
#[derive(Default, Clone)]
pub struct ProteinGroupOrders(HashMap<Vec<String>, Vec<String>>);

impl ProteinGroupOrders {
    pub fn from_path(path: &Path) -> Result<Self> {
        let metadata = ArrowReaderMetadata::load(&open_file(path)?, Default::default())
            .map_err(|error| invalid_input(error.to_string()))?;
        let roots = metadata
            .schema()
            .fields()
            .iter()
            .enumerate()
            .filter_map(|(index, field)| {
                ["pg_accessions", "protein_accessions", "proteins"]
                    .contains(&field.name().as_str())
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        if roots.is_empty() {
            return Ok(Self::default());
        }
        let projection = ProjectionMask::roots(metadata.parquet_schema(), roots);
        let observed = (0..metadata.metadata().num_row_groups())
            .into_par_iter()
            .map(|index| ObservedOrders::from_row_group(path, &metadata, projection.clone(), index))
            .try_reduce(ObservedOrders::default, |left, right| Ok(left.merge(right)))?;
        Ok(observed.finish())
    }

    pub fn is_identity(&self) -> bool {
        self.0.is_empty()
    }

    pub fn resolve(&self, accessions: &[String]) -> Vec<String> {
        self.0
            .get(accessions)
            .map_or_else(|| accessions.to_vec(), Clone::clone)
    }
}

#[derive(Default)]
struct ObservedOrders {
    raw_lists: HashMap<Vec<String>, Vec<String>>,
    orders: HashMap<Vec<String>, (String, Vec<String>)>,
}

impl ObservedOrders {
    fn from_row_group(
        path: &Path,
        metadata: &ArrowReaderMetadata,
        projection: ProjectionMask,
        row_group: usize,
    ) -> Result<Self> {
        let reader =
            ParquetRecordBatchReaderBuilder::new_with_metadata(open_file(path)?, metadata.clone())
                .with_row_groups(vec![row_group])
                .with_projection(projection)
                .build()
                .map_err(|error| invalid_input(error.to_string()))?;
        let mut groups = Self::default();
        for batch in reader {
            let batch = batch.map_err(|error| invalid_input(error.to_string()))?;
            let column = batch.column(0);
            for row in 0..batch.num_rows() {
                groups.observe(list_string_values(
                    column.as_ref(),
                    row,
                    "pg_accessions",
                    &["accession", "protein_accession"],
                )?);
            }
        }
        Ok(groups)
    }

    fn observe(&mut self, accessions: Vec<String>) {
        if self.raw_lists.contains_key(&accessions) {
            return;
        }
        let members = distinct_members(&accessions);
        let key = member_set(&members);
        let name = members
            .iter()
            .map(|value| accession_name(value))
            .collect::<Vec<_>>()
            .join(";");
        self.insert_min(key.clone(), (name, members));
        self.raw_lists.insert(accessions, key);
    }

    fn insert_min(&mut self, key: Vec<String>, candidate: (String, Vec<String>)) {
        let known = self.orders.entry(key).or_insert_with(|| candidate.clone());
        if candidate < *known {
            *known = candidate;
        }
    }

    fn merge(mut self, other: Self) -> Self {
        self.raw_lists.extend(other.raw_lists);
        for (key, candidate) in other.orders {
            self.insert_min(key, candidate);
        }
        self
    }

    fn finish(self) -> ProteinGroupOrders {
        if self
            .raw_lists
            .iter()
            .all(|(raw, key)| *raw == self.orders[key].1)
        {
            return ProteinGroupOrders::default();
        }
        ProteinGroupOrders(
            self.raw_lists
                .into_iter()
                .map(|(raw, key)| (raw, self.orders[&key].1.clone()))
                .collect(),
        )
    }
}

fn open_file(path: &Path) -> Result<File> {
    File::open(path).map_err(|source| MokumeError::Io {
        path: path.to_owned(),
        source,
    })
}

fn distinct_members(accessions: &[String]) -> Vec<String> {
    let mut seen = BTreeMap::new();
    accessions
        .iter()
        .filter(|accession| {
            let name = accession_name(accession);
            !name.is_empty() && seen.insert(name, ()).is_none()
        })
        .cloned()
        .collect()
}

fn member_set(accessions: &[String]) -> Vec<String> {
    let mut members = accessions
        .iter()
        .map(|value| accession_name(value).to_owned())
        .collect::<Vec<_>>();
    members.sort();
    members
}

fn accession_name(accession: &str) -> &str {
    let accession = accession.trim();
    let fields = accession.split('|').collect::<Vec<_>>();
    if fields.len() == 3 {
        fields[1]
    } else {
        accession
    }
}

fn invalid_input(message: String) -> MokumeError {
    MokumeError::InvalidInput { message }
}

#[cfg(test)]
mod tests {
    use super::{ObservedOrders, ProteinGroupOrders};
    use arrow::array::{Array, ArrayRef, ListBuilder, StringBuilder};
    use arrow::datatypes::{Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::error::Error;
    use std::fs::File;
    use std::path::Path;
    use std::sync::Arc;

    fn write_group_rows(path: &Path, rows: &[Vec<String>]) -> Result<(), Box<dyn Error>> {
        let mut values = ListBuilder::new(StringBuilder::new());
        for row in rows {
            for value in row {
                values.values().append_value(value);
            }
            values.append(true);
        }
        let values = values.finish();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "pg_accessions",
            values.data_type().clone(),
            true,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(values) as ArrayRef])?;
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(1))
            .build();
        let mut writer = ArrowWriter::try_new(File::create(path)?, schema, Some(properties))?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    #[test]
    fn parallel_row_groups_merge_observed_orders_independently_of_file_order(
    ) -> Result<(), Box<dyn Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("groups.parquet");
        let mut rows = [
            vec!["P2", "P1", "P2"],
            vec!["P1", "P2"],
            vec!["P4", "P3"],
            vec!["P2", "P1"],
        ]
        .map(|row| row.into_iter().map(str::to_owned).collect::<Vec<_>>());
        for _ in 0..2 {
            write_group_rows(&path, &rows)?;
            let groups = ProteinGroupOrders::from_path(&path)?;
            assert!(!groups.is_identity());
            assert_eq!(
                groups.resolve(&["P2", "P1"].map(str::to_owned)),
                vec!["P1", "P2"]
            );
            assert_eq!(
                groups.resolve(&["P4", "P3"].map(str::to_owned)),
                vec!["P4", "P3"]
            );
            rows.reverse();
        }
        Ok(())
    }

    #[test]
    fn preferred_order_is_observed_deduplicated_and_independent_of_read_order() {
        let a = vec!["P2".to_owned(), "P1".to_owned(), "P2".to_owned()];
        let b = vec!["P1".to_owned(), "P2".to_owned()];
        let mut groups = ObservedOrders::default();
        groups.observe(a.clone());
        assert_eq!(groups.orders[&b].1, vec!["P2", "P1"]);
        groups.observe(b.clone());
        assert_eq!(groups.finish().resolve(&a), b);
        let mut reversed = ObservedOrders::default();
        reversed.observe(b.clone());
        reversed.observe(a.clone());
        assert_eq!(reversed.finish().resolve(&a), b);
    }

    #[test]
    fn unchanged_lists_use_identity_path_across_row_groups() -> Result<(), Box<dyn Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("groups.parquet");
        let rows = [vec!["P2", "P1"], vec!["P3"], vec!["P2", "P1"], vec![]]
            .map(|row| row.into_iter().map(str::to_owned).collect::<Vec<_>>());
        write_group_rows(&path, &rows)?;
        let groups = ProteinGroupOrders::from_path(&path)?;
        assert!(groups.is_identity());
        assert_eq!(groups.resolve(&rows[0]), rows[0]);
        Ok(())
    }

    #[test]
    fn repeated_members_require_lookup_even_without_order_conflicts() -> Result<(), Box<dyn Error>>
    {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("groups.parquet");
        let rows = [vec!["P1", "P1"], vec!["P3", "P2"], vec!["P1", "P1"], vec![]]
            .map(|row| row.into_iter().map(str::to_owned).collect::<Vec<_>>());
        write_group_rows(&path, &rows)?;
        let groups = ProteinGroupOrders::from_path(&path)?;
        assert!(!groups.is_identity());
        assert_eq!(groups.resolve(&rows[0]), vec!["P1"]);
        assert_eq!(groups.resolve(&rows[1]), rows[1]);
        assert!(groups.resolve(&rows[3]).is_empty());
        Ok(())
    }
}
