#[cfg(target_os = "linux")]
use std::fs::read_to_string;

#[cfg(not(target_os = "linux"))]
use memory_stats::memory_stats;

use mokume_core::{MokumeError, Result, RuntimeConfig, parse_memory_to_bytes};
use mokume_io::DEFAULT_QPX_BATCH_SIZE;

const MIN_QPX_BATCH_SIZE: usize = 256;
const STREAM_BUDGET_DIVISOR: u64 = 4;
const ESTIMATED_BYTES_PER_QPX_ROW: u64 = 8 * 1024;

/// Execution plan derived from the optional `--memory` soft resident-memory budget.
///
/// With no budget, the established high-throughput reader settings are kept.
/// With a budget, QPX read-ahead is disabled and the Arrow batch size receives
/// at most one quarter of the requested bytes. RSS is checked between decoded
/// batches so dataset-sized aggregation state cannot grow past the budget
/// indefinitely without a deterministic error.
#[derive(Debug, Clone)]
pub(crate) struct MemoryPlan {
    limit_bytes: Option<u64>,
    qpx_batch_size: usize,
    qpx_window: usize,
    channel_capacity: usize,
}

impl MemoryPlan {
    pub(crate) fn from_runtime(runtime: &RuntimeConfig) -> Result<Self> {
        match runtime.memory.as_deref() {
            Some(value) => {
                let limit_bytes = parse_memory_to_bytes(value)?;
                Ok(Self::with_limit(limit_bytes))
            }
            None => Ok(Self::unlimited()),
        }
    }

    pub(crate) const fn unlimited() -> Self {
        Self {
            limit_bytes: None,
            qpx_batch_size: DEFAULT_QPX_BATCH_SIZE,
            qpx_window: 2,
            channel_capacity: 1,
        }
    }

    fn with_limit(limit_bytes: u64) -> Self {
        let stream_bytes = limit_bytes / STREAM_BUDGET_DIVISOR;
        let planned_rows = stream_bytes / ESTIMATED_BYTES_PER_QPX_ROW;
        let qpx_batch_size = usize::try_from(planned_rows)
            .unwrap_or(usize::MAX)
            .clamp(MIN_QPX_BATCH_SIZE, DEFAULT_QPX_BATCH_SIZE);
        Self {
            limit_bytes: Some(limit_bytes),
            qpx_batch_size,
            qpx_window: 1,
            channel_capacity: 0,
        }
    }

    pub(crate) const fn limit_bytes(&self) -> Option<u64> {
        self.limit_bytes
    }

    pub(crate) const fn qpx_batch_size(&self) -> usize {
        self.qpx_batch_size
    }

    pub(crate) const fn qpx_window(&self) -> usize {
        self.qpx_window
    }

    pub(crate) const fn channel_capacity(&self) -> usize {
        self.channel_capacity
    }

    pub(crate) fn check(&self, stage: &str) -> Result<()> {
        let Some(limit_bytes) = self.limit_bytes else {
            return Ok(());
        };
        let rss_bytes = current_rss_bytes().ok_or_else(|| MokumeError::InvalidInput {
            message: "--memory could not read current process resident memory".to_owned(),
        })?;
        if rss_bytes <= limit_bytes {
            return Ok(());
        }
        Err(MokumeError::InvalidInput {
            message: format!(
                "--memory budget exceeded during {stage}: process resident memory is {} MiB, limit is {} MiB",
                bytes_to_mib(rss_bytes),
                bytes_to_mib(limit_bytes),
            ),
        })
    }
}

fn bytes_to_mib(bytes: u64) -> u64 {
    bytes.div_ceil(1024_u64.pow(2))
}

#[cfg(target_os = "linux")]
fn current_rss_bytes() -> Option<u64> {
    let status = read_to_string("/proc/self/status").ok()?;
    parse_linux_rss_bytes(&status)
}

#[cfg(not(target_os = "linux"))]
fn current_rss_bytes() -> Option<u64> {
    u64::try_from(memory_stats()?.physical_mem).ok()
}

#[cfg(target_os = "linux")]
fn parse_linux_rss_bytes(status: &str) -> Option<u64> {
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    kib.checked_mul(1024)
}

#[cfg(test)]
mod tests {
    use mokume_core::RuntimeConfig;

    use super::MemoryPlan;
    #[cfg(target_os = "linux")]
    use super::parse_linux_rss_bytes;

    #[test]
    fn budget_changes_qpx_read_ahead_and_batch_size() {
        let result = MemoryPlan::from_runtime(&RuntimeConfig {
            memory: Some("1GB".to_owned()),
            threads: Some(24),
        });
        let Ok(plan) = result else {
            panic!("valid memory plan was rejected");
        };

        assert_eq!(plan.qpx_batch_size(), 32_768);
        assert_eq!(plan.qpx_window(), 1);
        assert_eq!(plan.channel_capacity(), 0);
    }

    #[cfg(any(
        target_os = "linux",
        target_os = "windows",
        target_os = "macos",
        target_os = "android",
        target_os = "ios",
        target_os = "freebsd"
    ))]
    #[test]
    fn reads_current_process_resident_memory() {
        assert!(super::current_rss_bytes().is_some_and(|bytes| bytes > 0));
    }

    #[cfg(any(
        target_os = "linux",
        target_os = "windows",
        target_os = "macos",
        target_os = "android",
        target_os = "ios",
        target_os = "freebsd"
    ))]
    #[test]
    fn reports_an_exceeded_resident_memory_budget() {
        let error = MemoryPlan::with_limit(1).check("test checkpoint");
        let Err(error) = error else {
            panic!("one-byte process memory budget was not rejected");
        };

        assert!(
            error.to_string().contains(
                "--memory budget exceeded during test checkpoint: process resident memory"
            )
        );
    }

    #[test]
    fn unlimited_plan_preserves_existing_stream_settings() {
        let plan = MemoryPlan::unlimited();

        assert_eq!(plan.qpx_batch_size(), 65_536);
        assert_eq!(plan.qpx_window(), 2);
        assert_eq!(plan.channel_capacity(), 1);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_linux_rss_from_proc_status() {
        let status = "Name:\tmokume\nVmRSS:\t  12345 kB\nThreads:\t2\n";

        assert_eq!(parse_linux_rss_bytes(status), Some(12_641_280));
    }
}
