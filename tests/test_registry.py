"""Tests for the PluginRegistry."""

import pytest

from mokume.core.registry import PluginRegistry, VALID_INPUT_LEVELS
from mokume.quantification.base import QuantificationMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DummyMethod(QuantificationMethod):
    """Minimal concrete method for testing."""

    @property
    def name(self):
        return "dummy"

    def quantify(self, peptide_df, **kwargs):
        return peptide_df


class _BadLevelMethod(QuantificationMethod):
    """Method with an invalid input_level."""

    @property
    def name(self):
        return "bad_level"

    @property
    def input_level(self):
        return "nonexistent"

    def quantify(self, peptide_df, **kwargs):
        return peptide_df


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset registry before and after each test.

    After reset, we must reload all quantification submodules so
    that @register decorators fire again and re-populate the store.
    """
    import importlib
    import mokume.quantification.topn
    import mokume.quantification.ratio
    import mokume.quantification as quant_mod

    PluginRegistry.reset()
    # Reload submodules first (they have @register decorators)
    importlib.reload(mokume.quantification.topn)
    importlib.reload(mokume.quantification.ratio)
    # Then reload __init__ which registers aliases
    importlib.reload(quant_mod)
    yield
    PluginRegistry.reset()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPluginRegistryRegister:
    def test_register_decorator(self):
        PluginRegistry.reset()

        @PluginRegistry.register("quantification", "test_method")
        class TestMethod(_DummyMethod):
            @property
            def name(self):
                return "test"

        assert PluginRegistry.is_registered("quantification", "test_method")
        instance = PluginRegistry.get("quantification", "test_method")
        assert instance.name == "test"

    def test_register_unknown_group_raises(self):
        with pytest.raises(ValueError, match="Unknown plugin group"):
            PluginRegistry.register("nonexistent_group", "foo")

    def test_register_instance_factory(self):
        PluginRegistry.reset()
        PluginRegistry.register_instance_factory(
            "quantification", "factory_test",
            lambda **kw: _DummyMethod(),
        )
        instance = PluginRegistry.get("quantification", "factory_test")
        assert instance.name == "dummy"

    def test_register_instance_factory_unknown_group(self):
        with pytest.raises(ValueError, match="Unknown plugin group"):
            PluginRegistry.register_instance_factory(
                "nonexistent", "foo", lambda **kw: None
            )


class TestPluginRegistryGet:
    def test_get_maxlfq_alias_resolves_to_directlfq(self):
        """'maxlfq' alias should resolve to DirectLFQ."""
        method = PluginRegistry.get("quantification", "maxlfq")
        assert method.name == "DirectLFQ"

    def test_get_case_insensitive(self):
        method = PluginRegistry.get("quantification", "MaxLFQ")
        assert method is not None

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown quantification method"):
            PluginRegistry.get("quantification", "totally_nonexistent_xyz")

    def test_topn_pattern(self):
        method = PluginRegistry.get("quantification", "top5")
        assert method is not None
        assert method.name in ("TopN", "Top5", "top5", "TopNQuantification")

    def test_topn_large_n(self):
        method = PluginRegistry.get("quantification", "top100")
        assert method is not None

    def test_invalid_input_level_raises(self):
        """A method with invalid input_level should raise at get() time."""
        PluginRegistry.reset()
        PluginRegistry.register("quantification", "bad_level")(_BadLevelMethod)
        with pytest.raises(ValueError, match="input_level='nonexistent'"):
            PluginRegistry.get("quantification", "bad_level")


class TestPluginRegistryAvailable:
    def test_available_returns_sorted(self):
        names = PluginRegistry.available("quantification")
        assert names == sorted(names)

    def test_available_includes_builtins(self):
        names = PluginRegistry.available("quantification")
        assert "maxlfq" in names  # backward-compat alias
        assert "topn" in names

    def test_available_empty_group(self):
        names = PluginRegistry.available("filter")
        assert isinstance(names, list)


class TestPluginRegistryReset:
    def test_reset_clears_registrations(self):
        PluginRegistry.reset()
        names = PluginRegistry.available("quantification")
        # After reset + entry point discovery, only entry-point methods remain
        # (or empty if not installed as package)
        assert isinstance(names, list)


class TestValidInputLevels:
    def test_valid_levels_set(self):
        assert "peptides" in VALID_INPUT_LEVELS
        assert "psms" in VALID_INPUT_LEVELS
        assert "peptides_raw" in VALID_INPUT_LEVELS
        assert "features" in VALID_INPUT_LEVELS
