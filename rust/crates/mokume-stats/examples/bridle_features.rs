//! Dump BRIDLE's technical sequence features for a gene list (parity checks
//! against the prototype's `feats.py`).
//!
//! `cargo run --release -p mokume-stats --example bridle_features -- \
//!     proteome.fasta genes.txt HUMAN features.tsv`
//!
//! `genes.txt` holds one gene per line; the organism filter may be `-` (none).

use std::error::Error;
use std::fmt::Write as _;

use mokume_stats::batch::bridle::sequence_features;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: bridle_features FASTA GENES.txt ORGANISM|- OUT.tsv".into());
    }
    let fasta = std::fs::read_to_string(&args[1])?;
    let genes: Vec<String> = std::fs::read_to_string(&args[2])?
        .lines()
        .map(str::trim)
        .filter(|g| !g.is_empty())
        .map(str::to_owned)
        .collect();
    let organism = (args[3] != "-").then_some(args[3].as_str());
    let f = sequence_features(&fasta, &genes, organism);
    let mut out = format!("gene\t{}\n", f.names.join("\t"));
    for (g, row) in genes.iter().zip(&f.values) {
        out.push_str(g);
        for v in row {
            write!(out, "\t{v}")?;
        }
        out.push('\n');
    }
    std::fs::write(&args[4], out)?;
    eprintln!("{} genes, {} without sequence", genes.len(), f.n_missing);
    Ok(())
}
