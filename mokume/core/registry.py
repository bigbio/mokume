"""
Plugin registry for the mokume package.

This module provides a central registry for all mokume extension types.
Built-in methods register via decorators at import time. Third-party
packages register via Python entry points, discovered on first access.

Extension groups:
    - quantification: Protein quantification algorithms
    - normalization.feature: Feature-level normalization methods
    - normalization.sample: Sample/peptide-level normalization methods
    - harmonization: Batch effect correction methods
    - imputation: Missing value imputation methods
    - filter: Quality control filters

Example — registering a built-in method::

    from mokume.core.registry import PluginRegistry

    @PluginRegistry.register("quantification", "directlfq")
    class DirectLFQQuantification(QuantificationMethod):
        ...

Example — third-party registration via pyproject.toml::

    [project.entry-points."mokume.quantification"]
    spectral_counting = "my_package:SpectralCountingMethod"
"""

import importlib.metadata
import logging
import re
from typing import Any, Dict, List, Optional, Set, Type

logger = logging.getLogger(__name__)

# Sentinel for TopN pattern matching
_TOPN_PATTERN = re.compile(r"^top(\d+)$")

# Valid input_level values for quantification methods.
# Must match keys in FLOW_DISPATCH (runner.py) and QpxDataset._VALID_LEVELS.
VALID_INPUT_LEVELS: Set[str] = {"peptides", "psms", "peptides_raw", "features"}


class PluginRegistry:
    """Central registry for all mokume extension types.

    Manages registration and discovery of plugins across five extension
    groups. Supports both decorator-based registration (built-in) and
    entry-point discovery (third-party packages).
    """

    _stores: Dict[str, Dict[str, Any]] = {
        "quantification": {},
        "normalization.feature": {},
        "normalization.sample": {},
        "harmonization": {},
        "imputation": {},
        "filter": {},
    }

    _discovered: bool = False

    @classmethod
    def register(cls, group: str, name: str):
        """Decorator to register a plugin class.

        Parameters
        ----------
        group : str
            Extension group (e.g., "quantification", "normalization.feature").
        name : str
            Name to register the plugin under (e.g., "maxlfq").

        Returns
        -------
        Callable
            Decorator that registers the class and returns it unchanged.

        Raises
        ------
        ValueError
            If the group is not recognized.

        Examples
        --------
        >>> @PluginRegistry.register("quantification", "my_method")
        ... class MyMethod(QuantificationMethod):
        ...     ...
        """
        if group not in cls._stores:
            raise ValueError(
                f"Unknown plugin group: '{group}'. "
                f"Available groups: {list(cls._stores.keys())}"
            )

        def decorator(klass: Type) -> Type:
            cls._stores[group][name.lower()] = klass
            return klass

        return decorator

    @classmethod
    def register_instance_factory(cls, group: str, name: str, factory):
        """Register a callable factory for a plugin.

        Useful for registering aliases like top3/top5 that create
        instances with specific parameters.

        Parameters
        ----------
        group : str
            Extension group.
        name : str
            Name to register under.
        factory : callable
            A callable that accepts **kwargs and returns a plugin instance.
        """
        if group not in cls._stores:
            raise ValueError(
                f"Unknown plugin group: '{group}'. "
                f"Available groups: {list(cls._stores.keys())}"
            )
        cls._stores[group][name.lower()] = factory

    @classmethod
    def get(cls, group: str, name: str, **kwargs: Any) -> Any:
        """Get a plugin instance by group and name.

        Handles special patterns like topN (top3, top5, top10) by
        parsing the numeric suffix.

        Parameters
        ----------
        group : str
            Extension group.
        name : str
            Plugin name.
        **kwargs
            Arguments passed to the plugin constructor.

        Returns
        -------
        Any
            An instance of the requested plugin.

        Raises
        ------
        ValueError
            If the plugin is not found.
        """
        cls._ensure_discovered()
        name_lower = name.lower()

        # Check direct match first
        entry = cls._stores.get(group, {}).get(name_lower)
        instance = None
        if entry is not None:
            if isinstance(entry, type):
                instance = entry(**kwargs)
            else:
                # It's a factory callable
                instance = entry(**kwargs)
        else:
            # Handle topN pattern: top3, top5, top10, etc.
            if group == "quantification":
                match = _TOPN_PATTERN.match(name_lower)
                if match:
                    topn_cls = cls._stores.get(group, {}).get("topn")
                    if topn_cls is not None:
                        n = int(match.group(1))
                        instance = topn_cls(n=n, **kwargs)

        if instance is None:
            available = cls.available(group)
            raise ValueError(
                f"Unknown {group} method: '{name}'. "
                f"Available: {available}"
            )

        # Validate input_level for quantification methods
        if group == "quantification" and hasattr(instance, "input_level"):
            level = instance.input_level
            if level not in VALID_INPUT_LEVELS:
                raise ValueError(
                    f"Quantification method '{name}' declares "
                    f"input_level='{level}', which is not valid. "
                    f"Must be one of: {sorted(VALID_INPUT_LEVELS)}"
                )

        return instance

    @classmethod
    def get_class(cls, group: str, name: str) -> Optional[Type]:
        """Get the registered class (not an instance) for a plugin.

        Parameters
        ----------
        group : str
            Extension group.
        name : str
            Plugin name.

        Returns
        -------
        Type or None
            The registered class, or None if not found.
        """
        cls._ensure_discovered()
        return cls._stores.get(group, {}).get(name.lower())

    @classmethod
    def available(cls, group: str) -> List[str]:
        """List registered plugin names for a group.

        Parameters
        ----------
        group : str
            Extension group.

        Returns
        -------
        list[str]
            Sorted list of available plugin names.
        """
        cls._ensure_discovered()
        return sorted(cls._stores.get(group, {}).keys())

    @classmethod
    def is_registered(cls, group: str, name: str) -> bool:
        """Check if a plugin is registered.

        Parameters
        ----------
        group : str
            Extension group.
        name : str
            Plugin name.

        Returns
        -------
        bool
        """
        cls._ensure_discovered()
        return name.lower() in cls._stores.get(group, {})

    @classmethod
    def _ensure_discovered(cls):
        """Discover entry-point plugins once on first access."""
        if cls._discovered:
            return
        cls._discovered = True

        for group in cls._stores:
            ep_group = f"mokume.{group}"
            try:
                # Python 3.12+: entry_points(group=...) returns a SelectableGroups
                # Python 3.9-3.11: entry_points() returns a dict
                # Python 3.12+: entry_points(group=...) returns matching entries
                # Python 3.9-3.11: entry_points() returns SelectableGroups
                try:
                    group_eps = importlib.metadata.entry_points(group=ep_group)
                except TypeError:
                    # Fallback for older Python versions
                    eps = importlib.metadata.entry_points()
                    if isinstance(eps, dict):
                        group_eps = eps.get(ep_group, [])
                    else:
                        group_eps = [
                            ep for ep in eps if ep.group == ep_group
                        ]

                for ep in group_eps:
                    try:
                        klass = ep.load()
                        cls._stores[group][ep.name.lower()] = klass
                        logger.debug(
                            "Discovered plugin: %s.%s -> %s",
                            group, ep.name, klass,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to load plugin '%s' from group '%s': %s",
                            ep.name, ep_group, exc,
                        )
            except Exception as exc:
                logger.debug(
                    "Entry point discovery failed for group '%s': %s",
                    ep_group, exc,
                )

    @classmethod
    def reset(cls):
        """Reset the registry. Mainly useful for testing."""
        for group in cls._stores:
            cls._stores[group].clear()
        cls._discovered = False
