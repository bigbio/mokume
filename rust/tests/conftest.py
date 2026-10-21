"""Shared pytest configuration for the Rust-backed wheel."""

import os

# Headless matplotlib backend so the periphery plotting tests run without a
# display (must be set before matplotlib is first imported).
os.environ.setdefault("MPLBACKEND", "Agg")
