"""Helpers for organization-level logo storage and lookup.

Logos are stored as PNGs under an ``org_logos`` folder next to this file,
using the pattern ``{organization_id}.png``. Any member of an organization
can overwrite the logo by uploading a new image.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional

import streamlit as st


_LOGOS_DIR = Path(__file__).resolve().parent / "org_logos"


def save_org_logo(org_id: str, file_bytes: bytes) -> Optional[str]:
    """Save an uploaded logo image for the given organization.

    The image is converted to PNG and written to ``org_logos/{org_id}.png``.
    Returns the absolute path on success, or None on failure.
    """
    if not org_id:
        st.error("Missing organization id; cannot save logo.")
        return None

    try:
        from PIL import Image
    except Exception:
        st.error("Pillow is required to process logo images. Please install the 'Pillow' package.")
        return None

    try:
        _LOGOS_DIR.mkdir(parents=True, exist_ok=True)
        img = Image.open(BytesIO(file_bytes))
        img = img.convert("RGBA")
        target = _LOGOS_DIR / f"{org_id}.png"
        img.save(target, format="PNG")
        return str(target)
    except Exception as e:
        st.error(f"Failed to save logo: {e}")
        return None


def get_org_logo_path(org_id: str) -> Optional[str]:
    """Return the absolute path to the organization's logo PNG, if it exists."""
    if not org_id:
        return None
    path = _LOGOS_DIR / f"{org_id}.png"
    if path.is_file():
        return str(path)
    return None

