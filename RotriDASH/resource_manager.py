#type: ignore
"""
Resource manager for adaptive quality and memory management.

Detects system resources (RAM, CPU) and returns a quality tier that
controls PDF export quality, caching limits, and garbage collection
behaviour.  Imported by report_pdf.py and plotting.py.
"""

import gc
import os
from dataclasses import dataclass

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class QualityTier:
    tier: str                # "low", "medium", "high"
    skip_toc_pass1: bool     # skip expensive 2-pass PDF TOC build
    aggressive_gc: bool      # gc.collect() after each graph export
    max_cache_entries: int   # limit @st.cache_data entries
    kaleido_scale: float     # PNG export scale factor
    kaleido_width: int       # PNG export width in pixels
    kaleido_height: int      # PNG export height in pixels


def detect_quality() -> QualityTier:
    """Detect system resources and return an appropriate quality tier.

    Tiers
    -----
    low    : <= 4 GB RAM  (e.g. t3.medium, Railway free)
    medium : <= 8 GB RAM  (e.g. t3.large)
    high   : >  8 GB RAM
    """
    total_mb = _total_ram_mb()

    if total_mb <= 4096:
        return QualityTier(
            tier="low",
            skip_toc_pass1=True,
            aggressive_gc=True,
            max_cache_entries=10,
            kaleido_scale=0.75,
            kaleido_width=900,
            kaleido_height=450,
        )
    if total_mb <= 8192:
        return QualityTier(
            tier="medium",
            skip_toc_pass1=False,
            aggressive_gc=False,
            max_cache_entries=30,
            kaleido_scale=1.0,
            kaleido_width=1200,
            kaleido_height=600,
        )
    return QualityTier(
        tier="high",
        skip_toc_pass1=False,
        aggressive_gc=False,
        max_cache_entries=50,
        kaleido_scale=1.0,
        kaleido_width=1200,
        kaleido_height=720,
    )


def force_gc():
    """Run a full garbage-collection cycle."""
    gc.collect()


def _total_ram_mb() -> float:
    """Return total physical RAM in MB, with fallback heuristics."""
    if psutil is not None:
        return psutil.virtual_memory().total / (1024 * 1024)

    # Fallback: try cgroup memory limit (Docker / Railway)
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
            limit = int(f.read().strip())
            if limit < 2**62:
                return limit / (1024 * 1024)
    except (FileNotFoundError, ValueError, PermissionError):
        pass

    # Fallback: try cgroup v2
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            val = f.read().strip()
            if val != "max":
                return int(val) / (1024 * 1024)
    except (FileNotFoundError, ValueError, PermissionError):
        pass

    # Last resort: assume low-RAM to be safe
    return 4096.0
