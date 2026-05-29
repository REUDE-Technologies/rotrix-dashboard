"""
Helpers for saving and loading reusable report setting profiles for the
Multi report tab.

Each profile currently stores:
- A name and optional description
- Throttle aggregation settings (single-file throttle controls)
- Graph configurations

Storage: Local Postgres (SQLAlchemy) or Supabase ``report_templates`` table,
with local JSON fallback.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import streamlit as st

logger = logging.getLogger(__name__)


class ThrottleAggregationSettings(TypedDict, total=False):
    start_throttle: Optional[float]
    end_throttle: Optional[float]
    throttle_interval: Optional[float]
    ramp_mode: Optional[str]


@dataclass
class MultiReportProfile:
    name: str
    description: str = ""
    plot_data_source: str = "Sorted performance table"
    throttle_aggregation: ThrottleAggregationSettings | None = None
    saved_graphs: List[Dict[str, Any]] | None = None
    created_at: Optional[str] = None
    created_by: Optional[str] = None


def _get_settings_path() -> Path:
    """Return the JSON file path used as local fallback for profiles."""
    base_dir = Path(__file__).resolve().parent
    return base_dir / "multi_report_settings.json"


def _use_local_auth() -> bool:
    """Check if local PG backend is active."""
    from auth import USE_LOCAL_AUTH
    return USE_LOCAL_AUTH


def _get_db():
    """Return a SQLAlchemy session."""
    from models import SessionLocal
    return SessionLocal()


def _get_supabase_client():
    """Return the current Supabase client or None."""
    try:
        from auth import get_supabase
        return get_supabase()
    except Exception:
        return None


def _get_org_id() -> Optional[str]:
    """Return the current user's organization_id from session state."""
    return st.session_state.get("organization_id")


# ---------------------------------------------------------------------------
# Database-backed load / save
# ---------------------------------------------------------------------------

def _load_profiles_from_db() -> Optional[List[Dict[str, Any]]]:
    """Fetch profiles from DB. Returns None on failure."""
    if _use_local_auth():
        return _load_profiles_from_pg()
    return _load_profiles_from_supabase()


def _load_profiles_from_pg() -> Optional[List[Dict[str, Any]]]:
    """Fetch profiles from local Postgres via SQLAlchemy."""
    try:
        from models import ReportTemplate
        from sqlalchemy import select
        db = _get_db()
        try:
            org_id = _get_org_id()
            stmt = select(ReportTemplate).order_by(ReportTemplate.created_at.asc())
            if org_id:
                stmt = stmt.where(ReportTemplate.organization_id == org_id)
            rows = db.execute(stmt).scalars().all()
            profiles: List[Dict[str, Any]] = []
            for row in rows:
                row_dict = {
                    "name": row.name,
                    "description": row.description,
                    "plot_data_source": row.plot_data_source,
                    "throttle_aggregation": row.throttle_aggregation,
                    "saved_graphs": row.saved_graphs,
                    "created_by": row.created_by,
                    "created_at": str(row.created_at) if row.created_at else "",
                }
                profiles.append(_db_row_to_profile(row_dict))
            return profiles
        finally:
            db.close()
    except Exception as exc:
        logger.warning("Failed to load profiles from PG: %s", exc)
        return None


def _load_profiles_from_supabase() -> Optional[List[Dict[str, Any]]]:
    """Fetch profiles from Supabase. Returns None on failure."""
    try:
        supabase = _get_supabase_client()
        if supabase is None:
            return None
        org_id = _get_org_id()
        query = supabase.table("report_templates").select("*").order("created_at", desc=False)
        if org_id:
            query = query.eq("organization_id", org_id)
        result = query.execute()
        rows = result.data or []
        profiles: List[Dict[str, Any]] = []
        for row in rows:
            profiles.append(_db_row_to_profile(row))
        return profiles
    except Exception as exc:
        logger.warning("Failed to load profiles from Supabase: %s", exc)
        return None


def _save_profiles_to_db(profiles: List[Dict[str, Any]]) -> bool:
    """Replace all org profiles in DB. Returns True on success."""
    if _use_local_auth():
        return _save_profiles_to_pg(profiles)
    return _save_profiles_to_supabase(profiles)


def _save_profiles_to_pg(profiles: List[Dict[str, Any]]) -> bool:
    """Replace all org profiles in local Postgres via SQLAlchemy."""
    try:
        from models import ReportTemplate
        db = _get_db()
        try:
            org_id = _get_org_id()
            # Delete existing profiles for this org
            if org_id:
                db.query(ReportTemplate).filter(
                    ReportTemplate.organization_id == org_id
                ).delete()
            else:
                db.query(ReportTemplate).filter(
                    ReportTemplate.organization_id.is_(None)
                ).delete()

            # Insert new profiles
            for p in profiles:
                template = ReportTemplate(
                    organization_id=org_id,
                    name=p.get("name", "Untitled"),
                    description=p.get("description", ""),
                    plot_data_source=p.get("plot_data_source", "Sorted performance table"),
                    throttle_aggregation=p.get("throttle_aggregation") or {},
                    saved_graphs=p.get("saved_graphs") or [],
                    created_by=p.get("created_by", "\u2014"),
                )
                db.add(template)
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
    except Exception as exc:
        logger.warning("Failed to save profiles to PG: %s", exc)
        return False


def _save_profiles_to_supabase(profiles: List[Dict[str, Any]]) -> bool:
    """Replace all org profiles in Supabase. Returns True on success."""
    try:
        supabase = _get_supabase_client()
        if supabase is None:
            return False
        org_id = _get_org_id()

        if org_id:
            supabase.table("report_templates").delete().eq("organization_id", org_id).execute()
        else:
            supabase.table("report_templates").delete().is_("organization_id", "null").execute()

        for p in profiles:
            row = _profile_to_db_row(p, org_id)
            supabase.table("report_templates").insert(row).execute()

        return True
    except Exception as exc:
        logger.warning("Failed to save profiles to Supabase: %s", exc)
        return False


def _db_row_to_profile(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Supabase row into the internal profile dict format."""
    throttle_raw = row.get("throttle_aggregation") or {}
    if isinstance(throttle_raw, str):
        try:
            throttle_raw = json.loads(throttle_raw)
        except Exception:
            throttle_raw = {}

    saved_graphs_raw = row.get("saved_graphs") or []
    if isinstance(saved_graphs_raw, str):
        try:
            saved_graphs_raw = json.loads(saved_graphs_raw)
        except Exception:
            saved_graphs_raw = []

    created_at = row.get("created_at") or ""
    if isinstance(created_at, str) and len(created_at) > 16:
        created_at = created_at[:16].replace("T", " ")

    return asdict(MultiReportProfile(
        name=str(row.get("name") or "").strip() or "Untitled",
        description=str(row.get("description") or ""),
        plot_data_source=str(row.get("plot_data_source") or "Sorted performance table"),
        throttle_aggregation=throttle_raw if isinstance(throttle_raw, dict) else {},
        saved_graphs=saved_graphs_raw if isinstance(saved_graphs_raw, list) else None,
        created_at=created_at,
        created_by=str(row.get("created_by") or "—"),
    ))


def _profile_to_db_row(profile: Dict[str, Any], org_id: Optional[str]) -> Dict[str, Any]:
    """Convert an internal profile dict into a Supabase insert row."""
    return {
        "organization_id": org_id,
        "name": profile.get("name", "Untitled"),
        "description": profile.get("description", ""),
        "plot_data_source": profile.get("plot_data_source", "Sorted performance table"),
        "throttle_aggregation": profile.get("throttle_aggregation") or {},
        "saved_graphs": profile.get("saved_graphs") or [],
        "created_by": profile.get("created_by", "—"),
    }


# ---------------------------------------------------------------------------
# Local JSON fallback
# ---------------------------------------------------------------------------

def _load_profiles_from_json() -> List[Dict[str, Any]]:
    """Load profiles from local JSON file (fallback)."""
    settings_path = _get_settings_path()
    if not settings_path.exists():
        return []
    try:
        with settings_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, list):
        return []

    profiles: List[Dict[str, Any]] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip()
        if not name:
            continue
        description = str(raw.get("description", ""))
        plot_data_source = str(raw.get("plot_data_source", "Sorted performance table"))
        if plot_data_source not in ("Raw data", "Sorted performance table"):
            plot_data_source = "Sorted performance table"

        throttle_raw = raw.get("throttle_aggregation") or {}
        throttle: ThrottleAggregationSettings = ThrottleAggregationSettings()
        if isinstance(throttle_raw, dict):
            for key in ("start_throttle", "end_throttle", "throttle_interval"):
                val = throttle_raw.get(key)
                if isinstance(val, (int, float)):
                    throttle[key] = float(val)
            ramp_mode = throttle_raw.get("ramp_mode")
            if isinstance(ramp_mode, str):
                throttle["ramp_mode"] = ramp_mode

        saved_graphs_raw = raw.get("saved_graphs")
        saved_graphs: List[Dict[str, Any]] = []
        if isinstance(saved_graphs_raw, list):
            for g in saved_graphs_raw:
                if isinstance(g, dict):
                    saved_graphs.append(dict(g))

        created_at = raw.get("created_at") if isinstance(raw.get("created_at"), str) else None
        created_by = raw.get("created_by") if isinstance(raw.get("created_by"), str) else None

        profiles.append(asdict(MultiReportProfile(
            name=name,
            description=description,
            plot_data_source=plot_data_source,
            throttle_aggregation=throttle,
            saved_graphs=saved_graphs if saved_graphs else None,
            created_at=created_at,
            created_by=created_by,
        )))
    return profiles


def _save_profiles_to_json(profiles: List[Dict[str, Any]]) -> None:
    """Persist profiles to local JSON file (fallback)."""
    settings_path = _get_settings_path()
    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with settings_path.open("w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API (DB-first, JSON fallback)
# ---------------------------------------------------------------------------

def load_profiles() -> List[Dict[str, Any]]:
    """Load profiles from Supabase; fall back to local JSON on failure."""
    db_profiles = _load_profiles_from_db()
    if db_profiles is not None:
        return db_profiles
    return _load_profiles_from_json()


def save_profiles(profiles: List[Dict[str, Any]]) -> None:
    """Save profiles to Supabase; fall back to local JSON on failure."""
    if _save_profiles_to_db(profiles):
        return
    _save_profiles_to_json(profiles)


def profile_from_session_state(name: str, description: str = "") -> Dict[str, Any]:
    """Snapshot current relevant settings into a profile dict.

    Currently captures:
    - Throttle aggregation single-file controls
    """

    # For multi-report profiles we always use the Sorted Performance Table
    # as the data source. The Plot tab still allows Raw/Sorted selection,
    # but profiles no longer carry a separate data-source choice.
    plot_source = "Sorted performance table"

    # Capture throttle aggregation controls. Prefer the persistent keys that
    # survive across Streamlit reruns (set by the Data tab whenever the
    # throttle widgets are rendered). Fall back to widget keys, then defaults.
    # NOTE: explicit `is not None` checks because 0.0 is a valid value.
    _thr_min_raw = st.session_state.get("_throttle_cfg_start")
    if _thr_min_raw is None:
        _thr_min_raw = st.session_state.get("single_file_throttle_min_input")
    if _thr_min_raw is None:
        _thr_min_raw = 0.0

    _thr_max_raw = st.session_state.get("_throttle_cfg_end")
    if _thr_max_raw is None:
        _thr_max_raw = st.session_state.get("single_file_throttle_max_input")
    if _thr_max_raw is None:
        _thr_max_raw = 100.0

    _thr_int_raw = st.session_state.get("_throttle_cfg_interval")
    if _thr_int_raw is None:
        _thr_int_raw = st.session_state.get("single_file_throttle_interval_input")
    if _thr_int_raw is None:
        _thr_int_raw = 5.0

    _ramp_mode = st.session_state.get("_throttle_cfg_ramp_mode")
    if not _ramp_mode:
        _ramp_mode = _get_str_state("single_file_ramp_mode_select")
    if not _ramp_mode:
        _ramp_mode = "ramp_up"

    throttle: ThrottleAggregationSettings = ThrottleAggregationSettings(
        start_throttle=float(_thr_min_raw) if isinstance(_thr_min_raw, (int, float)) else None,
        end_throttle=float(_thr_max_raw) if isinstance(_thr_max_raw, (int, float)) else None,
        throttle_interval=float(_thr_int_raw) if isinstance(_thr_int_raw, (int, float)) else None,
        ramp_mode=_ramp_mode,
    )

    # Build graph configurations from the current Plot tab state so that ALL
    # graphs (1, 2, 3, 4, 5, …) are captured without data loss.
    saved_graphs: List[Dict[str, Any]] = []

    # Prefer explicit graph parameter dicts from session_state (multi_param_graph_{idx})
    graph_configs: List[Dict[str, Any]] = []
    for key, value in st.session_state.items():
        if not key.startswith("multi_param_graph_"):
            continue
        if not isinstance(value, dict):
            continue
        try:
            idx_str = key.rsplit("_", 1)[-1]
            idx = int(idx_str)
        except (ValueError, TypeError):
            continue
        graph_configs.append(
            {
                "graph_number": idx,
                "x_axis": value.get("x_axis"),
                "left_y_axes": value.get("left_y_axes") or [],
                "right_y_axes": value.get("right_y_axes") or [],
                "smoothing_enabled": value.get("smoothing_enabled", False),
                "smoothing_method": value.get("smoothing_method", "savgol"),
                "smoothing_window": value.get("smoothing_window", 5),
            }
        )

    # Sort by graph number so they appear in a stable order
    graph_configs.sort(key=lambda g: g.get("graph_number", 0))

    if graph_configs:
        saved_graphs.extend(graph_configs)
    else:
        # Fallback for older sessions: capture whatever is in multi_param_saved_graphs
        raw_graphs = st.session_state.get("multi_param_saved_graphs")
        if isinstance(raw_graphs, list):
            for g in raw_graphs:
                if isinstance(g, dict):
                    graph_copy = {k: v for k, v in g.items() if k != "fig"}
                    saved_graphs.append(graph_copy)

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    # Prefer the author name captured from the detail info page.
    created_by = (st.session_state.get("author_name") or "").strip()
    if not created_by:
        # Fallback to a neutral placeholder when author details are missing.
        created_by = "—"

    profile = MultiReportProfile(
        name=name.strip(),
        description=description.strip(),
        plot_data_source=plot_source,
        throttle_aggregation=throttle,
        saved_graphs=saved_graphs if saved_graphs else None,
        created_at=created_at,
        created_by=created_by,
    )
    return asdict(profile)


def apply_profile_to_session_state(profile: Dict[str, Any]) -> None:
    """Apply a saved profile back onto Streamlit session state."""

    # When applying a profile, always drive reports from the
    # Sorted Performance Table in the multi-report flow.
    st.session_state["multi_param_plot_data_source"] = "Sorted performance table"

    throttle = profile.get("throttle_aggregation") or {}
    if isinstance(throttle, dict):
        _set_if_not_none("single_file_throttle_min_input", throttle.get("start_throttle"))
        _set_if_not_none("single_file_throttle_max_input", throttle.get("end_throttle"))
        _set_if_not_none(
            "single_file_throttle_interval_input", throttle.get("throttle_interval")
        )
        ramp_mode = throttle.get("ramp_mode")
        if isinstance(ramp_mode, str):
            st.session_state["single_file_ramp_mode_select"] = ramp_mode

        if throttle.get("start_throttle") is not None:
            st.session_state["_throttle_cfg_start"] = float(throttle["start_throttle"])
        if throttle.get("end_throttle") is not None:
            st.session_state["_throttle_cfg_end"] = float(throttle["end_throttle"])
        if throttle.get("throttle_interval") is not None:
            st.session_state["_throttle_cfg_interval"] = float(throttle["throttle_interval"])
        if isinstance(ramp_mode, str):
            st.session_state["_throttle_cfg_ramp_mode"] = ramp_mode

    saved_graphs_raw = profile.get("saved_graphs")
    if isinstance(saved_graphs_raw, list):
        restored_for_saved_list: List[Dict[str, Any]] = []

        # Detect whether graphs carry explicit graph_number metadata (new style)
        has_explicit_numbers = any(
            isinstance(g, dict) and isinstance(g.get("graph_number"), int)
            for g in saved_graphs_raw
        )

        if has_explicit_numbers:
            # New-style profiles: each graph dict has graph_number pointing to
            # multi_param_graph_{graph_number}.
            for g in saved_graphs_raw:
                if not isinstance(g, dict):
                    continue
                graph_number = g.get("graph_number")
                if not isinstance(graph_number, int):
                    continue
                params = {
                    "x_axis": g.get("x_axis"),
                    "left_y_axes": g.get("left_y_axes") or [],
                    "right_y_axes": g.get("right_y_axes") or [],
                    "smoothing_enabled": g.get("smoothing_enabled", False),
                    "smoothing_method": g.get("smoothing_method", "savgol"),
                    "smoothing_window": g.get("smoothing_window", 5),
                }
                st.session_state[f"multi_param_graph_{graph_number}"] = params

                # For compatibility with existing Plot tab logic that uses
                # multi_param_saved_graphs for Graph 2 and Graph 4+, maintain
                # a parallel list excluding Graph 1 and 3.
                if graph_number >= 2 and graph_number != 3:
                    tmp = dict(params)
                    tmp["graph_number"] = graph_number
                    tmp["fig"] = None
                    restored_for_saved_list.append(tmp)
        else:
            # Old-style profiles: interpret list position according to the
            # historical mapping:
            #   index 0 -> Graph 2
            #   index 1 -> Graph 4
            #   index 2 -> Graph 5
            #   index 3 -> Graph 6, etc.
            for idx, g in enumerate(saved_graphs_raw):
                if not isinstance(g, dict):
                    continue
                if idx == 0:
                    graph_number = 2
                else:
                    graph_number = idx + 3
                params = {
                    "x_axis": g.get("x_axis"),
                    "left_y_axes": g.get("left_y_axes") or [],
                    "right_y_axes": g.get("right_y_axes") or [],
                    "smoothing_enabled": g.get("smoothing_enabled", False),
                    "smoothing_method": g.get("smoothing_method", "savgol"),
                    "smoothing_window": g.get("smoothing_window", 5),
                }
                st.session_state[f"multi_param_graph_{graph_number}"] = params

                tmp = dict(params)
                tmp["fig"] = None
                restored_for_saved_list.append(tmp)

        st.session_state["multi_param_saved_graphs"] = restored_for_saved_list
    else:
        st.session_state["multi_param_saved_graphs"] = []


def _get_float_state(key: str) -> Optional[float]:
    val = st.session_state.get(key)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _get_str_state(key: str) -> Optional[str]:
    val = st.session_state.get(key)
    if isinstance(val, str):
        return val
    return None


def _set_if_not_none(key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (int, float)):
        st.session_state[key] = float(value)
    else:
        st.session_state[key] = value

