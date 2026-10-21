"""Tests for :class:`mokume.pipeline.config.RuntimeConfig`.

Covers ``__post_init__`` validation, the DuckDB memory-string parser, and the
``effective_workers`` heuristic used to size fork-based parallel sections.
"""

from __future__ import annotations

import pytest

from mokume.pipeline.config import RuntimeConfig, _parse_memory_string_to_gib


class TestParseMemoryString:
    @pytest.mark.parametrize(
        ("value", "expected_gib"),
        [
            ("80GB", 80.0),
            ("80gb", 80.0),
            ("16384MB", 16.0),
            ("1.5TB", 1536.0),
            ("1073741824B", 1.0),
            ("1073741824", 1.0),  # bare number = bytes (matches __post_init__ regex)
        ],
    )
    def test_parses_common_units(self, value: str, expected_gib: float) -> None:
        assert _parse_memory_string_to_gib(value) == pytest.approx(expected_gib)

    def test_rejects_garbage(self) -> None:
        with pytest.raises(ValueError, match="cannot parse memory string"):
            _parse_memory_string_to_gib("eighty gigs")


class TestRuntimeConfigValidation:
    def test_defaults_allow_both_none(self) -> None:
        rc = RuntimeConfig()
        assert rc.duckdb_memory is None
        assert rc.duckdb_threads is None

    @pytest.mark.parametrize("memory", ["80GB", "16384MB", "1.5TB"])
    def test_valid_duckdb_memory(self, memory: str) -> None:
        assert RuntimeConfig(duckdb_memory=memory).duckdb_memory == memory

    def test_rejects_bad_memory_format(self) -> None:
        with pytest.raises(ValueError, match="DuckDB memory string"):
            RuntimeConfig(duckdb_memory="lots")

    def test_rejects_non_positive_threads(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            RuntimeConfig(duckdb_threads=0)
        with pytest.raises(ValueError, match=">= 1"):
            RuntimeConfig(duckdb_threads=-1)


class TestEffectiveWorkers:
    """All tests pass an explicit ``system_total_gb`` so they do not depend on
    the host's actual RAM. The heuristic is
    ``floor((system_total - duckdb_memory - os_reserve) / per_worker_gb)``
    capped at ``duckdb_threads``.
    """

    def test_returns_default_when_nothing_set(self) -> None:
        assert RuntimeConfig().effective_workers(default=24) == 24
        assert RuntimeConfig().effective_workers() is None

    def test_uses_duckdb_threads_when_memory_absent(self) -> None:
        assert RuntimeConfig(duckdb_threads=8).effective_workers(default=24) == 8

    def test_uses_memory_budget_when_threads_absent(self) -> None:
        # system=125, parent=80, os_reserve=5 -> usable=40, /10 = 4 workers
        assert (
            RuntimeConfig(duckdb_memory="80GB").effective_workers(system_total_gb=125.0)
            == 4
        )

    def test_takes_min_of_threads_and_memory_budget(self) -> None:
        # 80 GB cap on 125 GB host allows 4 workers; duckdb_threads=2 wins.
        assert (
            RuntimeConfig(duckdb_memory="80GB", duckdb_threads=2).effective_workers(
                system_total_gb=125.0
            )
            == 2
        )
        # 80 GB cap on 125 GB host allows 4 workers; duckdb_threads=16
        # looser, so 4 wins.
        assert (
            RuntimeConfig(duckdb_memory="80GB", duckdb_threads=16).effective_workers(
                system_total_gb=125.0
            )
            == 4
        )

    def test_pxd030304_scenario(self) -> None:
        # The exact scenario that OOM-killed PXD030304: 125 GB host,
        # --duckdb-memory 80GB --duckdb-threads 16 should yield 4 workers,
        # not 7.
        rc = RuntimeConfig(duckdb_memory="80GB", duckdb_threads=16)
        assert rc.effective_workers(system_total_gb=125.0) == 4

    def test_clamps_to_one_when_parent_eats_everything(self) -> None:
        # parent >= system: nothing left, but we floor to at least 1 worker
        # so the pipeline still makes progress (the worker will share pages
        # with the parent before any COW write).
        assert (
            RuntimeConfig(duckdb_memory="120GB").effective_workers(
                system_total_gb=125.0
            )
            == 1
        )
        assert (
            RuntimeConfig(duckdb_memory="125GB").effective_workers(
                system_total_gb=125.0
            )
            == 1
        )

    def test_honours_overrides(self) -> None:
        # Smaller per_worker_gb -> more workers fit.
        rc = RuntimeConfig(duckdb_memory="40GB")
        assert (
            rc.effective_workers(
                system_total_gb=125.0, per_worker_gb=5.0, os_reserve_gb=0.0
            )
            == 17  # (125 - 40) / 5 = 17
        )

    def test_memory_dominates_when_threads_loose(self) -> None:
        # 256 GB host, 40 GB parent -> (256-40-5)/10 = 21 workers;
        # duckdb_threads=16 wins because it is tighter.
        assert (
            RuntimeConfig(duckdb_memory="40GB", duckdb_threads=16).effective_workers(
                system_total_gb=256.0
            )
            == 16
        )

    def test_fallback_when_psutil_missing(self) -> None:
        # When system_total_gb cannot be probed (no psutil and no override),
        # we degrade to a conservative duckdb_memory-based budget rather
        # than silently returning cpu_count workers.
        import sys
        from unittest import mock

        rc = RuntimeConfig(duckdb_memory="80GB")
        with mock.patch.dict(sys.modules, {"psutil": None}):
            workers = rc.effective_workers()
        # 80 - 5 = 75, /10 = 7 -> conservative-but-non-default fallback.
        assert workers == 7
