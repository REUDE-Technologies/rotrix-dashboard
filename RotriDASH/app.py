#type: ignore
"""
RotriDash — motor assessment dashboard (Streamlit application entry point).

Run with:
    streamlit run multi_param_app/app.py
"""
import os
try:
    from dotenv import load_dotenv
    try:
        load_dotenv()
    except OSError:
        # Errno 24: too many open files — avoid crashing; env may already be set
        pass
except ImportError:
    pass
import sys
import io
import re
import gc
import copy
import math
import time
import html
import base64
import hashlib
import tempfile
import zipfile
import traceback
import logging
from io import BytesIO
from datetime import datetime, timedelta, timezone

import streamlit as st
from streamlit.components.v1 import html as st_html_component
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Module imports  (all heavy logic lives in dedicated modules)
# ---------------------------------------------------------------------------
from config import (
    REUDE_LOGO_PATH,
    SHOW_ANALYSIS_TYPE_SELECTOR,
    SHOW_CALCULATORS_BUTTON,
    _get_reude_logo_b64,
    TOPIC_ASSESSMENT_PAIRS, ASSESSMENT_Y_AXIS_MAP,
    COLUMN_DISPLAY_NAMES,
    SORTED_TABLE_PDF_COLUMN_SHORT_NAMES,
    SORTED_TABLE_REPORT_DROP_COLUMNS,
    get_display_name, get_axis_title,
    _drop_sorted_table_report_columns,
    get_short_param_label,
)
from github_loader import UploadedGitHubFile, process_url
from session_state import init_session_state, cleanup_stale_session_data
from auth import (
    require_auth, is_authenticated, is_approved, get_profile_status, sync_profile_status_from_db,
    check_role, logout,
    render_login_panel, render_profile_setup,
    render_pending_approval_screen, render_rejected_screen,
    try_restore_session_from_browser,
)
from usage_tracking import auto_track_login, auto_track_file_upload, auto_track_report, auto_track_plot
from report_quota import (
    DEFAULT_DAILY_REPORT_QUOTA,
    get_remaining_quota,
    check_quota,
    get_user_quota,
    get_daily_report_count,
)
from layout_components import render_footer
import admin_dashboard
import admin_org_dashboard
from storage import (
    list_recent_files,
    list_reports_for_user,
    list_reports_for_org,
    get_download_url,
    upload_report,
    download_file,
)
from data_loader import (
    clean_file_info_text, parse_file_info_to_table,
    fix_duplicate_columns, convert_to_numeric_safe, filter_info_rows,
    format_seconds_to_mmss, mmss_to_seconds, seconds_to_mmss,
    get_tick_spacing, get_timestamp_ticks,
    ensure_seconds_column, add_hhmmss_seconds_column,
    convert_timestamps_to_seconds,
    load_csv, load_ulog, load_data,
    find_column_by_pattern, compute_basic_file_insights, extract_test_type_from_info,
)
from data_processing import (
    add_top_param_labels,
    get_numeric_columns, is_column_empty, get_non_empty_columns,
    detect_abnormalities,
    resample_to_common_time,
    process_throttle_aggregation, detect_throttle_regimes_from_raw,
    filter_df_by_ramp_mode,
    sanitize_table_for_streamlit,
)
from plotting import (
    store_graph_for_report, invalidate_report_after_data_change,
    ensure_report_performance_graphs_for_current_file,
    ensure_summary_graphs_for_current_file,
    ensure_report_graphs_for_current_config,
    build_throttle_dwell_bar_figure,
    build_stacked_thrust_area_figure,
    _prepare_fig_for_export, _fig_to_image_bytes, _fig_to_base64,
)
from report_pdf import build_report_pdf
from org_logo import save_org_logo, get_org_logo_path
from multi_report_settings import (
    load_profiles,
    save_profiles,
    profile_from_session_state,
    apply_profile_to_session_state,
)


def _load_sorted_report_profiles():
    """Load report profiles in the same order used by the template list UI."""
    profiles = load_profiles()
    try:
        def _profile_sort_key(profile):
            ts = (profile.get("created_at") or "").strip()
            if ts and "T" not in ts and " " in ts:
                ts_val = ts.replace(" ", "T")
            else:
                ts_val = ts
            try:
                return datetime.fromisoformat(ts_val)
            except Exception:
                return datetime.min

        return sorted(profiles, key=_profile_sort_key, reverse=True)
    except Exception:
        return profiles


def _auto_apply_profile_for_test_type(detected_test_type: str):
    """
    Auto-select/apply a report template when its name matches detected test type.
    Example: detected "UAT001" -> profile name containing "UAT001".
    """
    if not isinstance(detected_test_type, str) or not detected_test_type.strip():
        return None

    profiles = _load_sorted_report_profiles()
    if not profiles:
        return None

    def _canon(text: str) -> str:
        return "".join(ch.lower() for ch in str(text) if ch.isalnum())

    token = detected_test_type.strip()
    token_c = _canon(token)
    match_idx = None

    for i, profile in enumerate(profiles):
        name_c = _canon(profile.get("name", ""))
        if name_c == token_c:
            match_idx = i
            break

    if match_idx is None:
        for i, profile in enumerate(profiles):
            name_c = _canon(profile.get("name", ""))
            if token_c and (token_c in name_c or name_c in token_c):
                match_idx = i
                break

    if match_idx is None:
        return None

    st.session_state["report_profile_selected_idx"] = match_idx
    try:
        apply_profile_to_session_state(profiles[match_idx])
    except Exception:
        return None

    st.session_state["detected_test_type_profile"] = token
    return profiles[match_idx]


def _load_file_and_build_report_data(fname, uploaded_files, throttle_override=None):
    """Load one file by name from uploaded_files; return (raw_df, sorted_df) for report generation.

    Throttle settings are taken from the selected template when `throttle_override`
    is provided (dict with start_throttle, end_throttle, throttle_interval, ramp_mode);
    otherwise they fall back to the current Streamlit session state.
    Returns (None, None) on failure.
    """
    try:
        file = [f for f in uploaded_files if f.name == fname][0]
    except IndexError:
        return None, None
    file.seek(0)
    content = file.read()
    file.seek(0)
    file_ext = os.path.splitext(fname)[-1].lower()
    df = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        try:
            if isinstance(content, str):
                tmp_file.write(content.encode("utf-8"))
            else:
                tmp_file.write(content)
            tmp_file.flush()
            if file_ext == ".ulg":
                dfs_dict, _ = load_ulog(tmp_file.name)
                if not dfs_dict:
                    return None, None
                topic_keys = list(dfs_dict.keys())
                default_topic = st.session_state.get("multi_param_ulog_topic") or topic_keys[0]
                df = dfs_dict.get(default_topic, dfs_dict[topic_keys[0]]).copy()
            else:
                df, _ = load_data(tmp_file.name, file_ext, key_suffix="_multi_param")
            if df is None or df.empty:
                return None, None

            # Detect RotriX test type from file info + filename and auto-apply
            # a matching report template (if available).
            if file_ext in (".csv", ".xlsx"):
                _info_text = (st.session_state.get("report_file_info_text") or "").strip()
                _detected_test_type = extract_test_type_from_info(_info_text, filename=fname)
                if _detected_test_type:
                    st.session_state["detected_test_type"] = _detected_test_type
                    _matched_profile = _auto_apply_profile_for_test_type(_detected_test_type)
                    if isinstance(_matched_profile, dict):
                        _matched_throttle = _matched_profile.get("throttle_aggregation") or {}
                        if isinstance(_matched_throttle, dict) and _matched_throttle:
                            throttle_override = _matched_throttle
        finally:
            try:
                os.unlink(tmp_file.name)
            except Exception:
                pass
    df = ensure_seconds_column(df)
    if "Index" not in df.columns:
        df.insert(0, "Index", range(1, len(df) + 1))
    throttle_col = find_column_by_pattern(df, [
        "Throttle - %", "Throttle Input (%)", "Throttle (%)", "Throttle", "throttle",
    ])
    if throttle_col is None:
        for col in df.columns:
            col_lower = str(col).lower()
            if "servo" in col_lower or "esc" in col_lower:
                series = pd.to_numeric(df[col], errors="coerce")
                if series.notna().sum() == 0:
                    continue
                if series.between(800, 2200).mean() >= 0.5:
                    pwm_series = (series - 1000.0) / 1000.0 * 100.0
                    df["Throttle - %"] = pwm_series.clip(0.0, 100.0)
                    throttle_col = "Throttle - %"
                    break
    if throttle_col is None:
        return None, None
    current_col = find_column_by_pattern(df, ["Cur - A", "Current (A)", "Current [A]", "Current", "current", "Cur"])
    voltage_col = find_column_by_pattern(df, ["Vol - V", "Voltage (V)", "Voltage [V]", "Voltage", "voltage", "Vol"])
    rpm1_col = find_column_by_pattern(df, [
        "RPM", "RPM1 - RPM", "RPM1",
        "Motor Electrical Speed (RPM)", "Electrical Speed (RPM)",
        "Rotational Speed (RPM)", "Rotational Speed",
    ])
    rpm2_col = find_column_by_pattern(df, ["RPM2 - RPM", "RPM2", "Motor Optical Speed (RPM)", "Optical Speed (RPM)"])
    thrust_col = find_column_by_pattern(df, ["Thrust - gf", "Thrust (gf)", "Thrust (kgf)", "Thrust [g]", "Thrust", "thrust"])
    torque_col = find_column_by_pattern(df, [
        "Torque - N*m", "Torque (N*m)", "Torque (N·m)",
        "Torque (Nm)", "Torque (N.m)", "Torque [N·m]", "Torque [N*m]",
        "Torque", "torque",
    ])
    motorpower_col = find_column_by_pattern(df, [
        "MotorPower - W", "MotorPower",
        "Mechanical Power (W)", "Mechanical (W)",
        "Electrical Power (W)", "Electrical (W)",
        "InPower - W", "InPower", "Power",
    ])
    if not all([throttle_col, current_col, voltage_col, thrust_col, torque_col]):
        return None, None

    # Resolve throttle settings: prefer explicit overrides from the selected template,
    # otherwise fall back to the current session-state controls.
    if isinstance(throttle_override, dict):
        _start = throttle_override.get("start_throttle")
        _end = throttle_override.get("end_throttle")
        _interval = throttle_override.get("throttle_interval")
        _ramp = throttle_override.get("ramp_mode")
    else:
        _start = _end = _interval = _ramp = None

    start_val = (
        _start
        if isinstance(_start, (int, float))
        else st.session_state.get("single_file_throttle_min_input", 0.0)
    )
    end_val = (
        _end
        if isinstance(_end, (int, float))
        else st.session_state.get("single_file_throttle_max_input", 100.0)
    )
    throttle_min = min(start_val, end_val)
    throttle_max = max(start_val, end_val)

    if isinstance(_interval, (int, float)):
        throttle_interval = float(_interval)
    else:
        throttle_interval = st.session_state.get("single_file_throttle_interval_input", 5.0)

    if isinstance(_ramp, str) and _ramp:
        ramp_mode = _ramp
    else:
        ramp_mode = st.session_state.get("single_file_ramp_mode_select", "ramp_up")
    result_df = process_throttle_aggregation(
        df, throttle_col, current_col, voltage_col,
        rpm1_col, rpm2_col, thrust_col, torque_col,
        motorpower_col, mode=ramp_mode,
        throttle_min=throttle_min,
        throttle_max=throttle_max,
        throttle_interval=throttle_interval,
    )
    return df, result_df


class _BytesFile:
    """File-like wrapper for in-memory bytes (e.g. from zip or folder) so they work like UploadedFile."""
    def __init__(self, name, data):
        self.name = name
        self._data = data if isinstance(data, bytes) else bytes(data)
        self._pos = 0
    def seek(self, pos=0, whence=0):
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = len(self._data) + pos
        else:
            raise ValueError("whence must be 0, 1, or 2")
    def read(self, size=-1):
        if size == -1:
            out = self._data[self._pos:]
            self._pos = len(self._data)
            return out
        out = self._data[self._pos:self._pos + size]
        self._pos += len(out)
        return out
    @property
    def size(self):
        return len(self._data)


def _build_report_raw_data_df(raw_df, throttle_override=None):
    """Build report_raw_data_df with Throttle Range (10%) and Time range for summary graphs.
    Returns a dataframe suitable for ensure_summary_graphs_for_current_file, or None.
    """
    if raw_df is None or getattr(raw_df, "empty", True):
        return None
    _sum = raw_df.copy()
    _sum = fix_duplicate_columns(_sum)
    _throttle_col = find_column_by_pattern(_sum, ["Throttle - %", "Throttle Input (%)", "Throttle (%)", "Throttle", "throttle"])
    # Fallback: derive throttle percentage from ESC/PWM input when explicit
    # throttle column is absent (common in client bench exports).
    if _throttle_col is None or _throttle_col not in _sum.columns:
        _esc_col = find_column_by_pattern(
            _sum,
            [
                "ESC signal (µs)",
                "ESC signal (us)",
                "ESC Signal (µs)",
                "ESC Signal (us)",
                "ESC signal",
                "ESC",
                "PWM",
                "PWM (µs)",
                "PWM (us)",
            ],
        )
        if _esc_col is not None and _esc_col in _sum.columns:
            _pwm = pd.to_numeric(_sum[_esc_col], errors="coerce")
            _valid = _pwm.dropna()
            if not _valid.empty:
                _vmin = float(_valid.min())
                _vmax = float(_valid.max())
                # Typical PWM microseconds (1000-2000) -> 0-100%.
                if _vmax > 120 and _vmin >= 800:
                    _sum["Throttle - %"] = ((_pwm - 1000.0) / 1000.0 * 100.0).clip(0.0, 100.0)
                else:
                    # Already in percentage-like range.
                    _sum["Throttle - %"] = _pwm.clip(0.0, 100.0)
                _throttle_col = "Throttle - %"
    if _throttle_col is None or _throttle_col not in _sum.columns:
        return None
    _current_col = find_column_by_pattern(_sum, ["Cur - A", "Current (A)", "Current [A]", "Current", "current", "Cur"])

    if isinstance(throttle_override, dict):
        _start = throttle_override.get("start_throttle")
        _end = throttle_override.get("end_throttle")
        _interval = throttle_override.get("throttle_interval")
        _ramp = throttle_override.get("ramp_mode")
    else:
        _start = _end = _interval = _ramp = None

    _start_val = (
        _start
        if isinstance(_start, (int, float))
        else st.session_state.get("single_file_throttle_min_input", 0.0)
    )
    _end_val = (
        _end
        if isinstance(_end, (int, float))
        else st.session_state.get("single_file_throttle_max_input", 100.0)
    )
    _throttle_min = min(_start_val, _end_val)
    _throttle_max = max(_start_val, _end_val)
    _throttle_interval = (
        float(_interval)
        if isinstance(_interval, (int, float))
        else float(st.session_state.get("single_file_throttle_interval_input", 5.0))
    )
    _ramp_mode = (
        _ramp
        if isinstance(_ramp, str) and _ramp
        else st.session_state.get("single_file_ramp_mode_select", "ramp_up")
    )

    # Apply ramp/throttle filtering only when an explicit override is provided.
    # Otherwise, keep full-file raw data so summary dwell time matches actual test duration.
    _has_explicit_throttle_override = (
        isinstance(throttle_override, dict)
        and any(v is not None for v in (_start, _end, _interval, _ramp))
    )
    if _has_explicit_throttle_override and _current_col is not None and _current_col in _sum.columns:
        _sum = filter_df_by_ramp_mode(
            _sum,
            _throttle_col,
            _current_col,
            mode=_ramp_mode,
            throttle_min=_throttle_min,
            throttle_max=_throttle_max,
            throttle_interval=_throttle_interval,
        )
        if _sum is None or getattr(_sum, "empty", True):
            return None
        _sum = fix_duplicate_columns(_sum)
        _throttle_col = find_column_by_pattern(_sum, ["Throttle - %", "Throttle Input (%)", "Throttle (%)", "Throttle", "throttle"])
        if _throttle_col is None or _throttle_col not in _sum.columns:
            return None

    if "Index" not in _sum.columns:
        _sum.insert(0, "Index", range(1, len(_sum) + 1))
    _thresh = np.arange(0, 101, 10)
    _labels = [f"{i}-{i+10}" for i in range(0, 100, 10)]
    _t_vals = pd.to_numeric(_sum[_throttle_col], errors="coerce")
    # If throttle column exists but is mostly empty, try to backfill from ESC/PWM.
    if _t_vals.isna().sum() > len(_t_vals) // 2:
        _esc_col = find_column_by_pattern(
            _sum,
            [
                "ESC signal (µs)",
                "ESC signal (us)",
                "ESC Signal (µs)",
                "ESC Signal (us)",
                "ESC signal",
                "ESC",
                "PWM",
                "PWM (µs)",
                "PWM (us)",
            ],
        )
        if _esc_col is not None and _esc_col in _sum.columns:
            _pwm = pd.to_numeric(_sum[_esc_col], errors="coerce")
            _valid = _pwm.dropna()
            if not _valid.empty:
                _vmin = float(_valid.min())
                _vmax = float(_valid.max())
                if _vmax > 120 and _vmin >= 800:
                    _t_vals = ((_pwm - 1000.0) / 1000.0 * 100.0)
                else:
                    _t_vals = _pwm
    _t_vals = _t_vals.clip(0, 100)
    _sum["Throttle Range (10%)"] = pd.cut(_t_vals, bins=_thresh, labels=_labels, include_lowest=True)
    _time_col = find_column_by_pattern(_sum, ["Time (s)", "Time (secs)", "timestamp_seconds", "Time"])
    if _time_col is None or _time_col not in _sum.columns:
        return None
    _t_series = pd.to_numeric(_sum[_time_col], errors="coerce")
    if _t_series.isna().all() or _t_series.isna().sum() > len(_t_series) // 2:
        _raw_time = _sum[_time_col].astype(str)
        _t_series = _raw_time.apply(lambda x: mmss_to_seconds(x) if x and str(x).strip().lower() not in ("", "nan", "none") else np.nan)
    if "Throttle Range (10%)" in _sum.columns:
        _seg = (_sum["Throttle Range (10%)"] != _sum["Throttle Range (10%)"].shift()).cumsum()
        _seg_first = _t_series.groupby(_seg).first()
        _ref_per_row = _seg.map(_seg_first)
        _sum["Time range"] = (_t_series - _ref_per_row).round(4)
    else:
        _first_ts = _t_series.iloc[0] if len(_t_series) else 0
        _sum["Time range"] = (_t_series - _first_ts).round(4)
    if "Throttle Range (10%)" in _sum.columns and "Time range" in _sum.columns:
        return _sum
    return None


def _refresh_report_raw_data_df(throttle_override=None):
    """Recompute report_raw_data_df from multi_param_raw_df (Throttle Range / Time range).

    Preview and PDF generation must use this so summary graphs match batch report logic.
    """
    raw = st.session_state.get("multi_param_raw_df")
    if raw is None or getattr(raw, "empty", True):
        return False
    built = _build_report_raw_data_df(raw, throttle_override=throttle_override)
    st.session_state["report_raw_data_df"] = built
    return built is not None


# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RotriDash",
    page_icon="\U0001f680",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize all session state variables
init_session_state()

# Attempt to restore session from browser localStorage on page reload
# (sub-second flicker on first load when a saved session exists, then auto-restores)
try_restore_session_from_browser()



def _register_plot_shown():
    """Increment plots_generated_count once per run when any chart is shown."""
    if not st.session_state.get("_plots_counted_this_run", False):
        st.session_state.plots_generated_count = st.session_state.get("plots_generated_count", 0) + 1
        st.session_state._plots_counted_this_run = True


def _render_report_history_tab():
    """Render the History tab inside the Report Generation page.

    Access control:
      - org admin / viewer  → all organisation reports
      - editor              → only their own reports
    """
    def _fmt_ist(dt_str: str) -> str:
        if not dt_str:
            return "—"
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ist = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
            return ist.strftime("%b %d, %Y %I:%M %p IST")
        except Exception:
            return dt_str[:16].replace("T", " ")

    user_id = st.session_state.get("user_id")
    org_id = st.session_state.get("organization_id")
    role = st.session_state.get("user_role", "viewer")

    # Require an approved profile before showing history downloads.
    if not is_approved():
        render_pending_approval_screen()
        return

    if not user_id or not org_id:
        st.info("No organization assigned yet.")
        return

    # Brief description mirroring the report mode wording
    st.markdown(
        "History mode shows all reports downloaded . "
    )

    # Date filter
    _hc1, _hc2, _ = st.columns([1, 1, 2])
    with _hc1:
        _h_from = st.date_input("From", value=datetime.now() - timedelta(days=90), key="rh_tab_from")
    with _hc2:
        _h_to = st.date_input("To", value=datetime.now(), key="rh_tab_to")

    # Fetch reports based on role
    if role in ("org_admin", "super_admin", "viewer"):
        # Org admin + viewer → all org reports
        raw_reports = list_reports_for_org(org_id, limit=200)
    else:
        # Editor → own reports only
        raw_reports = list_reports_for_user(org_id, user_id, limit=200)

    # Client-side date filter
    reports = []
    for r in raw_reports:
        gen_at = r.get("generated_at", "")
        if isinstance(gen_at, str) and len(gen_at) >= 10:
            try:
                r_date = datetime.fromisoformat(gen_at[:10]).date()
                if _h_from <= r_date <= _h_to:
                    reports.append(r)
            except (ValueError, TypeError):
                reports.append(r)
        else:
            reports.append(r)

    if not reports:
        st.info("No reports found for the selected date range.")
        return

    st.caption(f"**{len(reports)}** report(s) found")

    # Select all + download options + download button (single row)
    _sa_col, _mode_col, _dl_col = st.columns([1, 3, 2])

    # Track previous value so we know when user unchecks it
    prev_sel_all = st.session_state.get("rh_tab_sel_all_prev", False)

    with _sa_col:
        _sel_all = st.checkbox("Select all", key="rh_tab_sel_all")
        if _sel_all:
            # Check all
            for i in range(len(reports)):
                st.session_state[f"rh_tab_cb_{i}"] = True
        elif prev_sel_all and not _sel_all:
            # Just transitioned from checked → unchecked: clear all
            for i in range(len(reports)):
                st.session_state[f"rh_tab_cb_{i}"] = False

    # Remember current state for next rerun
    st.session_state["rh_tab_sel_all_prev"] = _sel_all

    with _mode_col:
        download_mode = st.radio(
            "Download as",
            ["PDF", "CSV", "Both"],
            index=0,
            horizontal=True,
            key="rh_tab_download_mode",
        )

    sel_idx = [i for i in range(len(reports)) if st.session_state.get(f"rh_tab_cb_{i}", False)]
    include_pdf = download_mode in ("PDF", "Both")
    include_csv = download_mode in ("CSV", "Both")

    with _dl_col:
        if not sel_idx or not (include_pdf or include_csv):
            st.caption("Select reports and download type")
        else:
            _sel_reports = [reports[i] for i in sel_idx]

            # Case 1: single report, single format -> direct file download
            if len(_sel_reports) == 1 and (include_pdf ^ include_csv):
                r = _sel_reports[0]
                rname = r.get("report_name", "report")
                rname_base = os.path.splitext(str(rname))[0]
                safe_name = "".join(
                    c if c.isalnum() or c in "-_ " else "_"
                    for c in str(rname_base)
                ) or "report"
                if include_pdf and r.get("pdf_storage_path"):
                    pdf_bytes = download_file(r["pdf_storage_path"]) or None
                    if pdf_bytes:
                        st.download_button(
                            "Download PDF",
                            data=pdf_bytes,
                            file_name=f"{safe_name}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="rh_tab_download_pdf_btn",
                        )
                elif include_csv and r.get("csv_storage_path"):
                    csv_bytes = download_file(r["csv_storage_path"]) or None
                    if csv_bytes:
                        st.download_button(
                            "Download CSV",
                            data=csv_bytes,
                            file_name=f"{safe_name}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="rh_tab_download_csv_btn",
                        )
            else:
                # Case 2: multiple reports OR both formats -> ZIP
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    used_names: set[str] = set()
                    for r in _sel_reports:
                        rname = r.get("report_name", "report")
                        rname_base = os.path.splitext(str(rname))[0]
                        safe_name = "".join(
                            c if c.isalnum() or c in "-_ " else "_"
                            for c in str(rname_base)
                        ) or "report"
                        if include_pdf and r.get("pdf_storage_path"):
                            pdf_bytes = download_file(r["pdf_storage_path"], silent=True) or None
                            if pdf_bytes:
                                pdf_name = f"{safe_name}.pdf"
                                # Ensure each entry in the ZIP has a unique name
                                while pdf_name in used_names:
                                    pdf_name = pdf_name.replace(".pdf", "_1.pdf")
                                used_names.add(pdf_name)
                                zf.writestr(pdf_name, pdf_bytes)
                        if include_csv and r.get("csv_storage_path"):
                            csv_bytes = download_file(r["csv_storage_path"], silent=True) or None
                            if csv_bytes:
                                csv_name = f"{safe_name}.csv"
                                while csv_name in used_names:
                                    csv_name = csv_name.replace(".csv", "_1.csv")
                                used_names.add(csv_name)
                                zf.writestr(csv_name, csv_bytes)
                zip_buffer.seek(0)
                zip_bytes = zip_buffer.getvalue()

                if zip_bytes:
                    date_str = datetime.now().strftime("%Y%m%d")
                    zip_name = f"reports_{date_str}.zip"
                    st.download_button(
                        "Download selected reports",
                        data=zip_bytes,
                        file_name=zip_name,
                        mime="application/zip",
                        use_container_width=True,
                        key="rh_tab_download_btn",
                    )
                else:
                    st.caption("No files available for the chosen type.")

    # Report list
    with st.container(height=450):
        for i, report in enumerate(reports):
            rname = report.get("report_name") or "Report"
            gen_at = report.get("generated_at") or ""
            gen_at_display = _fmt_ist(gen_at)
            pdf_path = report.get("pdf_storage_path")
            csv_path = report.get("csv_storage_path")
            tags = []
            if pdf_path:
                tags.append("PDF")
            if csv_path:
                tags.append("CSV")
            tag_str = f" [{' | '.join(tags)}]" if tags else ""
            label = f"{html.escape(rname)} · {gen_at_display}{tag_str}"
            st.checkbox(label, key=f"rh_tab_cb_{i}")


def main():
    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  MAIN APPLICATION ENTRY POINT                                  ║
    # ║  This single function drives the entire Streamlit app.         ║
    # ║  Major sections (search by ═══ to jump between them):          ║
    # ║    1. CSS & Layout Setup                                       ║
    # ║    2. Header Bar, Logo & Profile Popover                       ║
    # ║    3. Auth Routing (login, signup, profile setup, approval)     ║
    # ║    4. Admin Dashboards & File Upload                           ║
    # ║    5. Analysis Type Selection & Batch Report Generation        ║
    # ║    6. Single-file RotriDash workflow (Summary/Data/Plot)         ║
    # ║    7. Multi-File Comparison Mode                               ║
    # ╚══════════════════════════════════════════════════════════════════╝
    st.session_state._plots_counted_this_run = False  # Reset each run so we can count at most one "plot view" per run

    # Keep session in sync with DB so approved users see dashboard after admin approves (no refresh needed)
    if is_authenticated():
        sync_profile_status_from_db()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: CSS & LAYOUT SETUP
    # Global styles, fixed header bar, responsive positioning
    # ═══════════════════════════════════════════════════════════════════

    # Inject all global CSS styles (header bar, layouts, responsive breakpoints)
    from styles import inject_global_styles
    inject_global_styles()


    # Section 2: Header bar, logo & profile popover (extracted to header.py)
    from header import render_header_bar
    render_header_bar()

    # Front page (landing) — extracted to front_page.py
    if st.session_state.show_front_page:
        from front_page import render_front_page
        render_front_page()

    # ----- Login panel (slideshow left + login form right) -----
    elif st.session_state.get('show_login_form', False) and not is_authenticated():
        render_login_panel()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: AUTH ROUTING
    # Determines which screen to show based on authentication state:
    #   - Not logged in → front page or login panel
    #   - Logged in, pending_setup → profile form
    #   - Logged in, rejected → re-submit screen
    #   - Logged in, pending_approval → waiting screen
    #   - Logged in, approved → dashboard (next section)
    # ═══════════════════════════════════════════════════════════════════

    # ----- Profile setup (after login, before approval) -----
    elif is_authenticated() and get_profile_status() == 'pending_setup':
        render_profile_setup(is_edit=False)

    # ----- Profile rejected — re-submit -----
    elif is_authenticated() and get_profile_status() == 'rejected':
        render_rejected_screen()

    # ----- Pending approval -----
    elif is_authenticated() and get_profile_status() == 'pending_approval':
        render_pending_approval_screen()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4: ADMIN DASHBOARDS & FILE UPLOAD
    # Routes super_admin → admin_dashboard, admin → org dashboard,
    # viewers/editors → profile editor + file upload area.
    # File upload supports drag-and-drop, ZIP, GitHub URL, and folder.
    # ═══════════════════════════════════════════════════════════════════

    # ----- Profile editor (explicitly opened from header) -----
    elif st.session_state.get("show_profile_editor", False):
        if not is_authenticated():
            st.session_state.show_profile_editor = False
            st.session_state.show_login_form = True
            st.rerun()
        # Always show profile editor here, even for admins/super_admins
        render_profile_setup(is_edit=True)

    # ----- Author form / Admin dashboard page -----
    elif st.session_state.show_author_form:
        if not is_authenticated():
            st.session_state.show_author_form = False
            st.session_state.show_login_form = True
            st.rerun()
        # If user is super_admin and approved, show the admin dashboard
        if check_role(["super_admin"]) and is_approved():
            admin_dashboard.render()
        elif check_role(["org_admin"]) and is_approved():
            # Org admin — show org-scoped admin dashboard
            admin_org_dashboard.render()
        else:
            # For approved non-admin users: show only the profile editor (no duplicate slideshow layer)
            render_profile_setup(is_edit=True)

    # ----- Report History page (all authenticated users) -----
    elif st.session_state.get("show_report_history", False):
        if not is_authenticated() or not is_approved():
            st.session_state.show_report_history = False
            st.session_state.show_calculators = False
            st.session_state.show_front_page = True
            if not is_authenticated():
                st.session_state.show_login_form = True
            st.rerun()
        import report_history
        report_history.render()

    # ----- Calculators hub (approved authenticated users) -----
    elif SHOW_CALCULATORS_BUTTON and st.session_state.get("show_calculators", False):
        if not is_authenticated() or not is_approved():
            st.session_state.show_calculators = False
            st.session_state.show_front_page = True
            if not is_authenticated():
                st.session_state.show_login_form = True
            st.rerun()
        import calculators
        calculators.render()

    # File Upload Section (auth gate: require authenticated and approved)
    elif st.session_state.show_upload_area:
        if not is_authenticated() or not is_approved():
            st.session_state.show_upload_area = False
            st.session_state.show_calculators = False
            st.session_state.show_front_page = True
            if not is_authenticated():
                st.session_state.show_login_form = True
            st.rerun()
        # Viewers cannot upload files — redirect to Report History
        if check_role(["viewer"]):
            st.session_state.show_upload_area = False
            st.session_state.show_calculators = False
            st.session_state.show_report_history = True
            st.rerun()
        st.markdown("""
        <style>
        .upload-section {
            background: #F8FAFC;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            border: 2px solid #E2E8F0;
            transition: all 0.3s ease;
        }
        .upload-section.active {
            border-color: #1B6CA8;
            background: #EFF6FF;
            box-shadow: 0 4px 12px rgba(27, 108, 168, 0.15);
        }
        .upload-section:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        }
        .file-preview-card {
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
        }
        .file-preview-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-color: #1B6CA8;
            transform: translateY(-1px);
        }
        .file-actions {
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }
        .file-action-btn {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .file-action-btn:hover {
            background: #F1F5F9;
            border-color: #94A3B8;
        }
        .file-action-btn.primary {
            background: #1B6CA8;
            color: white;
            border-color: #1B6CA8;
        }
        .file-action-btn.primary:hover {
            background: #155D91;
        }
        .file-action-btn.danger {
            background: #dc3545;
            color: white;
            border-color: #dc3545;
        }
        .file-action-btn.danger:hover {
            background: #c82333;
        }
        .upload-zone {
            border: 2px dashed #E2E8F0;
            border-radius: 8px;
            padding: 30px;
            text-align: center;
            background: white;
            transition: all 0.2s;
            cursor: pointer;
        }
        .upload-zone:hover {
            border-color: #1B6CA8;
            background: #F8FAFF;
            transform: scale(1.02);
        }
        .upload-zone.dragover {
            border-color: #1B6CA8;
            background: #EFF6FF;
        }
        .file-stats {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .file-stats h6 {
            color: rgba(255,255,255,0.9);
            margin-bottom: 8px;
        }
        .file-stats .stat-value {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .file-stats .stat-label {
            font-size: 12px;
            opacity: 0.8;
        }
        .bulk-actions {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .bulk-actions h6 {
            color: #475569;
            margin-bottom: 10px;
        }
        .tab-content {
            animation: fadeIn 0.3s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .file-type-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .file-type-badge.csv {
            background: #d4edda;
            color: #155724;
        }
        .file-type-badge.ulg {
            background: #d1ecf1;
            color: #0c5460;
        }
        .fm-file-management-wrap {
            margin-top: -5.5rem !important;
            padding-top: 0 !important;
        }
        /* Pull file upload block up after the title */
        div:has(.fm-file-management-wrap) + div {
            margin-top: -3rem !important;
            padding-top: 0 !important;
        }
        .fm-section-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #0F2B3D;
            text-align: center;
            margin-top: 0 !important;
            margin-bottom: 0.25rem;
            letter-spacing: -0.02em;
            padding-top: 0 !important;
        }
        .fm-section-title::after {
            content: "";
            display: block;
            width: 48px;
            height: 3px;
            background: linear-gradient(90deg, transparent, #1B6CA8, transparent);
            margin: 0.35rem auto 0;
            border-radius: 2px;
        }
        .fm-cards-row {
            display: flex;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .upload-section-enhanced {
            flex: 1;
            background: #F8FAFC;
            border-radius: 18px;
            padding: 1.75rem 1.5rem;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.04), 0 1px 3px rgba(0,0,0,0.04);
            transition: box-shadow 0.25s ease, border-color 0.25s ease;
        }
        .upload-section-enhanced:hover {
            box-shadow: 0 8px 28px rgba(27, 108, 168, 0.08);
            border-color: #1B6CA8;
        }
        .fm-submit-wrap {
            margin-top: 1.5rem;
            padding-top: 1.25rem;
            border-top: 1px solid #E2E8F0;
        }
        .fm-uniform-wrap {
            max-width: min(88%, 1400px);
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
        section[data-testid="stFileUploader"] {
            max-width: 100%;
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
            margin-top: 0 !important;
        }
        /* Remove gap between file uploader and the next block (KPI / summary) */
        div:has(section[data-testid="stFileUploader"]) + div,
        div:has(> section[data-testid="stFileUploader"]) + div {
            margin-top: 0 !important;
            padding-top: 0.25rem !important;
        }
        section[data-testid="stFileUploader"] > div {
            border-radius: 12px !important;
            border: 1px solid #E2E8F0 !important;
            background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%) !important;
            box-shadow: 0 2px 10px rgba(10, 46, 66, 0.06) !important;
            transition: box-shadow 0.2s ease, border-color 0.2s ease;
            min-height: 48px !important;
            height: auto !important;
            padding: 0.4rem 1rem !important;
        }
        section[data-testid="stFileUploader"] > div:hover {
            border-color: #1B6CA8 !important;
            box-shadow: 0 4px 18px rgba(27, 108, 168, 0.12) !important;
        }
        section[data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #0A2E42 0%, #0F3D5C 45%, #1B6CA8 100%) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 5px !important;
            font-weight: 600 !important;
            font-size: 0.65rem !important;
            padding: 0.2rem 0.4rem !important;
            min-height: unset !important;
            line-height: 1.2 !important;
            max-width: 5.5rem !important;
            width: auto !important;
            box-shadow: 0 2px 8px rgba(10, 46, 66, 0.2) !important;
            transition: transform 0.15s ease, box-shadow 0.2s ease !important;
        }
        section[data-testid="stFileUploader"] button:hover {
            box-shadow: 0 4px 14px rgba(27, 108, 168, 0.28) !important;
            transform: translateY(-1px);
        }
        .fm-empty-state {
            border-radius: 16px;
            border: 2px dashed #E2E8F0;
            background: #F8FAFC;
            padding: 2rem;
            text-align: center;
        }
        .fm-welcome {
            font-size: clamp(1.3rem, 2vw, 1.7rem);
            font-weight: 600;
            color: #0F2B3D;
            margin-top: 0 !important;
            margin-bottom: 1rem;
            max-width: min(88%, 1400px);
            margin-left: auto;
            margin-right: auto;
        }
        .fm-kpi-card {
            max-width: min(88%, 1400px);
            margin-left: auto;
            margin-right: auto;
            background: linear-gradient(135deg, #0A2E42 0%, #0F3D5C 35%, #185F80 60%, #1B6CA8 100%);
            border-radius: 16px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 1.5rem;
            flex-wrap: wrap;
            position: relative;
            overflow: hidden;
            max-height: 180px;
            min-height: 120px;
        }
        .fm-kpi-card::before {
            content: "";
            position: absolute;
            top: -20%; right: -10%;
            width: 40%; height: 140%;
            background: radial-gradient(ellipse at center, rgba(255,255,255,0.08) 0%, transparent 70%);
            pointer-events: none;
        }
        .fm-kpi-icon-wrap {
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: rgba(255,255,255,0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            position: relative;
        }
        .fm-kpi-icon-wrap .fm-kpi-badge {
            position: absolute;
            bottom: -4px;
            left: 50%;
            transform: translateX(-50%);
            background: #1B6CA8;
            color: #fff;
            font-size: 0.6rem;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 4px;
            letter-spacing: 0.05em;
        }
        .fm-kpi-content {
            flex: 1;
            min-width: 200px;
        }
        .fm-kpi-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #fff;
            margin: 0 0 0.75rem 0;
        }
        .fm-kpi-row {
            display: flex;
            align-items: stretch;
            gap: 0;
        }
        .fm-kpi-item {
            flex: 1;
            text-align: center;
            padding: 0 1rem;
        }
        .fm-kpi-item:not(:last-child) {
            border-right: 1px solid rgba(255,255,255,0.4);
        }
        .fm-kpi-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #fff;
            display: block;
            margin-bottom: 0.25rem;
        }
        .fm-kpi-label {
            font-size: 0.8rem;
            color: rgba(255,255,255,0.9);
        }
        .fm-kpi-dots {
            display: flex;
            gap: 6px;
            margin-top: 0.75rem;
            justify-content: center;
        }
        .fm-kpi-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: rgba(255,255,255,0.5);
        }
        .fm-kpi-dot.active {
            background: #1B6CA8;
        }
        .fm-summary-sidebar {
            background: linear-gradient(180deg, #0A2E42 0%, #0F3D5C 50%, #185F80 100%);
            border-radius: 12px;
            padding: 1rem;
            color: #fff;
        }
        .fm-summary-sidebar .fm-summary-title { font-size: 0.85rem; font-weight: 700; margin-bottom: 0.75rem; }
        .fm-summary-sidebar .fm-summary-item { margin-bottom: 0.6rem; font-size: 0.8rem; }
        .fm-summary-sidebar .fm-summary-value { font-size: 1.25rem; font-weight: 700; display: block; }
        </style>
        """, unsafe_allow_html=True)

        # Build welcome greeting with user name only (company name removed from file management area)
        _user_display = (st.session_state.get("user_name") or st.session_state.get("author_name") or "").strip() or "Guest"
        _welcome_html = f'<p class="fm-welcome">Welcome, {html.escape(_user_display)}</p>'
        st.markdown(_welcome_html, unsafe_allow_html=True)

        # KPI values for Usage Summary slide in carousel (after welcome)
        # Defaults (session-based) – used as a fallback.
        _carousel_kpi_files = len(st.session_state.uploaded_files)
        _carousel_kpi_plots = st.session_state.get("plots_generated_count", 0)
        _carousel_kpi_reports = st.session_state.get("reports_exported_count", 0)

        # If using local Postgres auth, try to pull a fresh 24‑hour summary
        # from the UsageEvent table so the KPIs always reflect real usage.
        try:
            org_id = st.session_state.get("organization_id")
            if org_id:
                import db_queries as dbq

                now_utc = datetime.now(timezone.utc)
                since_24h = (now_utc - timedelta(hours=24)).isoformat()

                _carousel_kpi_files = dbq.count_events_where(
                    event_type="file_uploaded",
                    org_id=org_id,
                    created_after=since_24h,
                )
                _carousel_kpi_plots = dbq.count_events_where(
                    event_type="plot_created",
                    org_id=org_id,
                    created_after=since_24h,
                )
                _carousel_kpi_reports = dbq.count_events_where(
                    event_type="report_generated",
                    org_id=org_id,
                    created_after=since_24h,
                )
        except Exception:
            # If anything fails (missing table, env, etc.), fall back to session counters.
            pass

        _carousel_slide_summary = (
            '<div class="fm-detail-slide fm-summary-slide active" data-slide="0">'
            '<div class="fm-detail-icon"><span style="font-size:3.75rem;">📋</span><span class="fm-detail-badge">SUMMARY</span></div>'
            '<div class="fm-detail-text fm-summary-content">'
            '<p class="fm-detail-title fm-summary-title">Your Usage Summary</p>'
            '<div class="fm-summary-sections">'
            '<div class="fm-summary-section fm-summary-section-input">'
            '<span class="fm-summary-section-label">Input</span>'
            '<div class="fm-summary-kpi-tile">'
            '<span class="fm-summary-kpi-icon">📁</span>'
            '<span class="fm-summary-kpi-value">' + str(_carousel_kpi_files) + '</span>'
            '<span class="fm-summary-kpi-label">Files Uploaded</span>'
            '</div></div>'
            '<div class="fm-summary-divider" aria-hidden="true"></div>'
            '<div class="fm-summary-section fm-summary-section-output">'
            '<span class="fm-summary-section-label">Output</span>'
            '<div class="fm-summary-kpi-row">'
            '<div class="fm-summary-kpi-tile">'
            '<span class="fm-summary-kpi-icon">📈</span>'
            '<span class="fm-summary-kpi-value">' + str(_carousel_kpi_plots) + '</span>'
            '<span class="fm-summary-kpi-label">Plots Generated</span>'
            '</div>'
            '<div class="fm-summary-kpi-tile">'
            '<span class="fm-summary-kpi-icon">📄</span>'
            '<span class="fm-summary-kpi-value">' + str(_carousel_kpi_reports) + '</span>'
            '<span class="fm-summary-kpi-label">Reports Exported</span>'
            '</div></div></div>'
            '</div></div></div>'
        )
        _carousel_html = (
        """
        <!DOCTYPE html>
        <html>
        <head><meta charset="UTF-8">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
        * { box-sizing: border-box; }
        body { margin: 0; padding: 0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .fm-detail-banner {
            background: linear-gradient(135deg, #0A2E42 0%, #0F3D5C 35%, #185F80 60%, #1B6CA8 100%);
            border-radius: 16px;
            padding: 1.5rem 1.75rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            position: relative;
            overflow: hidden;
            min-height: 208px;
        }
        .fm-detail-banner::before {
            content: "";
            position: absolute;
            top: -20%; right: -10%;
            width: 40%; height: 140%;
            background: radial-gradient(ellipse at center, rgba(255,255,255,0.08) 0%, transparent 70%);
            pointer-events: none;
        }
        .fm-detail-slides { position: relative; width: 100%; min-height: 138px; }
        .fm-detail-slide {
            position: absolute; left: 0; right: 0; top: 0;
            display: flex; align-items: center; gap: 1.5rem;
            opacity: 0; visibility: hidden; transition: opacity 0.45s ease, visibility 0.45s;
            pointer-events: none;
        }
        .fm-detail-slide.active {
            position: relative; opacity: 1; visibility: visible; pointer-events: auto;
        }
        .fm-detail-icon {
            width: 120px; height: 120px; min-width: 120px;
            border-radius: 50%;
            background: #fff;
            border: 1px solid rgba(255,255,255,0.6);
            display: flex; align-items: center; justify-content: center;
            position: relative;
            flex-shrink: 0;
            margin-top: 1.6rem;
            margin-left: 1.5rem;
        }
        .fm-detail-icon .fm-detail-badge {
            position: absolute; bottom: -9px; left: 50%; transform: translateX(-50%);
            background: #fff;
            color: #154360;
            font-size: 0.88rem; font-weight: 700;
            padding: 6px 14px; border-radius: 6px;
            letter-spacing: 0.06em;
            white-space: nowrap;
        }
        .fm-detail-text { flex: 1; min-width: 0; position: relative; }
        .fm-detail-title { font-size: 2.5rem; font-weight: 700; color: #fff; margin: 0 0 0.45rem 0; line-height: 1.28;padding-top: 1.5rem; }
        .fm-detail-sub { font-size: 1.85rem; color: rgba(255,255,255,0.92); margin: 0; line-height: 1.45; }
        .fm-detail-kpi-row { display: flex; gap: 2rem; margin-top: 0.5rem; flex-wrap: wrap; }
        .fm-detail-kpi-item { text-align: center; }
        .fm-detail-kpi-value { font-size: 2rem; font-weight: 700; color: #fff; display: block; }
        .fm-detail-kpi-label { font-size: 0.9rem; color: rgba(255,255,255,0.9); }
        /* Elite summary slide: sections + tiles; reduce top spacing to match other slides height */
        .fm-summary-slide .fm-detail-icon { margin-top: 0; }
        .fm-summary-slide .fm-detail-text { align-self: stretch; display: flex; flex-direction: column; }
        .fm-summary-slide .fm-summary-title { padding-top: 0; }
        .fm-summary-content { padding-top: 0; margin-top: 0; }
        .fm-summary-title { font-size: 1.85rem; margin-bottom: 0.2rem; letter-spacing: -0.02em; }
        .fm-summary-subtitle { font-size: 0.8rem; color: rgba(255,255,255,0.75); margin: 0 0 1rem 0; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase; }
        .fm-summary-sections { display: flex; align-items: stretch; gap: 0; flex: 1; min-height: 0; }
        .fm-summary-section { flex: 1; display: flex; flex-direction: column; gap: 0.6rem; padding: 0.85rem 1rem; background: rgba(255,255,255,0.06); border-radius: 12px; border: 1px solid rgba(255,255,255,0.12); min-width: 0; }
        .fm-summary-section-input { border-top-right-radius: 0; border-bottom-right-radius: 0; }
        .fm-summary-section-output { border-top-left-radius: 0; border-bottom-left-radius: 0; flex: 1.2; }
        .fm-summary-divider { width: 1px; background: linear-gradient(180deg, transparent, rgba(255,255,255,0.25), transparent); flex-shrink: 0; }
        .fm-summary-section-label { font-size: 0.7rem; font-weight: 700; color: rgba(255,255,255,0.7); letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.25rem; }
        .fm-summary-kpi-row { display: flex; gap: 0.75rem; flex: 1; flex-wrap: wrap; }
        .fm-summary-kpi-tile { flex: 1; min-width: 90px; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 0.65rem 0.5rem; background: rgba(255,255,255,0.08); border-radius: 10px; border: 1px solid rgba(255,255,255,0.15); box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
        .fm-summary-kpi-icon { font-size: 1.35rem; margin-bottom: 0.35rem; opacity: 0.95; }
        .fm-summary-kpi-tile .fm-summary-kpi-value { font-size: 1.75rem; font-weight: 800; color: #fff; display: block; line-height: 1.1; letter-spacing: -0.03em; }
        .fm-summary-kpi-tile .fm-summary-kpi-label { font-size: 0.75rem; color: rgba(255,255,255,0.88); font-weight: 500; }
        .fm-detail-dots { display: flex; gap: 8px; margin-top: 0.35rem; justify-content: center; }
        .fm-kpi-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: transparent;
            border: 1.5px solid rgba(255,255,255,0.85);
            cursor: pointer; transition: background 0.2s, border-color 0.2s;
        }
        .fm-kpi-dot:hover { border-color: rgba(255,255,255,1); }
        .fm-kpi-dot.active { background: #1B6CA8; border-color: #1B6CA8; }
        .fm-carousel-wrap { max-width: 88%; margin: 0 auto 1.5rem auto; max-height: 220px; overflow: hidden; }
        </style></head>
        <body>
        <div class="fm-carousel-wrap">
        <div id="fm-detail-carousel" class="fm-detail-banner">
            <div class="fm-detail-slides">
                """
        + _carousel_slide_summary
        + """
                <div class="fm-detail-slide" data-slide="1">
                    <div class="fm-detail-icon"><span style="font-size:3.75rem;">📊</span><span class="fm-detail-badge">INSIGHTS</span></div>
                    <div class="fm-detail-text">
                        <p class="fm-detail-title">Expedite decision making with actionable insights.</p>
                        <p class="fm-detail-sub">Turn motor data into clear plots and professional PDF reports—from upload to export in one flow.</p>
                    </div>
                </div>
                <div class="fm-detail-slide" data-slide="2">
                    <div class="fm-detail-icon"><span style="font-size:3.75rem;">📈</span><span class="fm-detail-badge">ANALYSIS</span></div>
                    <div class="fm-detail-text">
                        <p class="fm-detail-title">Multi-param analysis at your fingertips.</p>
                        <p class="fm-detail-sub">Plot thrust, efficiency, vibration, and more from your data with configurable X and dual Y-axes—ready to export.</p>
                    </div>
                </div>
                <div class="fm-detail-slide" data-slide="3">
                    <div class="fm-detail-icon"><span style="font-size:3.75rem;">📄</span><span class="fm-detail-badge">REPORTS</span></div>
                    <div class="fm-detail-text">
                        <p class="fm-detail-title">Professional PDF reports in one click.</p>
                        <p class="fm-detail-sub">Generate polished reports with plots and summaries - ready to share with stakeholders.</p>
                    </div>
                </div>
                <div class="fm-detail-slide" data-slide="4">
                    <div class="fm-detail-icon"><span style="font-size:3.75rem;">⚡</span><span class="fm-detail-badge">FAST</span></div>
                    <div class="fm-detail-text">
                        <p class="fm-detail-title">From upload to insight in minutes.</p>
                        <p class="fm-detail-sub">Upload your test data, choose analysis type, and get results without leaving the dashboard.</p>
                    </div>
                </div>
            </div>
            <div class="fm-detail-dots">
                <span class="fm-kpi-dot active" data-dot="0" title="Usage Summary"></span>
                <span class="fm-kpi-dot" data-dot="1" title="Insights"></span>
                <span class="fm-kpi-dot" data-dot="2" title="Analysis"></span>
                <span class="fm-kpi-dot" data-dot="3" title="Reports"></span>
                <span class="fm-kpi-dot" data-dot="4" title="Fast"></span>
            </div>
        </div>
        </div>
        <script>
        (function(){
            var carousel = document.getElementById("fm-detail-carousel");
            if (!carousel) return;
            var slides = carousel.querySelectorAll(".fm-detail-slide");
            var dots = carousel.querySelectorAll(".fm-kpi-dot");
            var total = slides.length;
            var current = 0;
            var interval = 5000;
            var timer = null;
            function goTo(idx) {
                current = (idx + total) % total;
                slides.forEach(function(s, i){ s.classList.toggle("active", i === current); });
                dots.forEach(function(d, i){ d.classList.toggle("active", i === current); });
            }
            function next() { goTo(current + 1); }
            function startTimer() {
                if (timer) clearInterval(timer);
                timer = setInterval(next, interval);
            }
            dots.forEach(function(dot, i){
                dot.addEventListener("click", function(){ goTo(i); startTimer(); });
            });
            startTimer();
        })();
        </script>
        </body></html>
        """
        )
        st_html_component(_carousel_html, height=300)

        _fm_left, _fm_center, _fm_right = st.columns([0.06, 0.88, 0.06])
        with _fm_center:
            st.markdown(
                "<div class='fm-file-management-wrap'>"
                "<h3 class='fm-section-title'>File Management</h3>"
                "</div>",
                unsafe_allow_html=True
            )
            with st.container():
                st.markdown("""
                <style>
                section[data-testid="stFileUploader"] > div {
                    min-height: 48px !important;
                    height: auto !important;
                    display: flex !important;
                    align-items: center !important;
                }
                </style>
                """, unsafe_allow_html=True)
                _uploader_key = f"desktop_uploader_{st.session_state.get('uploader_key_counter', 0)}"
                uploaded_files = st.file_uploader(
                    "Choose files to upload",
                    type=["csv", "xlsx", "zip"],
                    key=_uploader_key,
                    label_visibility="collapsed",
                    accept_multiple_files=True,
                    help="CSV, Excel (.xlsx), or ZIP of CSVs. Drag and drop or click to browse.")
            # Process uploaded files
            if uploaded_files:
                new_files_added = False
                existing_names = [f.name for f in st.session_state.uploaded_files]
                for uploaded_file in uploaded_files:
                    name_lower = (uploaded_file.name or "").lower()
                    # Direct CSV or Excel upload – add as-is
                    if name_lower.endswith(".csv") or name_lower.endswith(".xlsx"):
                        if uploaded_file.name not in existing_names:
                            st.session_state.uploaded_files.append(uploaded_file)
                            existing_names.append(uploaded_file.name)
                            new_files_added = True
                            auto_track_file_upload(uploaded_file.name, getattr(uploaded_file, "size", 0) or 0)
                    # ZIP upload – extract contained CSV files into individual in-memory files
                    elif name_lower.endswith(".zip"):
                        try:
                            uploaded_file.seek(0)
                            zip_bytes = uploaded_file.read()
                            uploaded_file.seek(0)
                            with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zf:
                                for entry in zf.namelist():
                                    # Skip directories
                                    if entry.endswith("/") or entry.rstrip("/").rstrip("\\").endswith("/"):
                                        continue
                                    base = os.path.basename(entry.rstrip("/"))
                                    if not base or base.startswith("."):
                                        continue
                                    if os.path.splitext(base)[-1].lower() != ".csv":
                                        continue
                                    try:
                                        data = zf.read(entry)
                                    except Exception:
                                        continue
                                    if base not in existing_names:
                                        st.session_state.uploaded_files.append(_BytesFile(base, data))
                                        existing_names.append(base)
                                        new_files_added = True
                                        auto_track_file_upload(base, len(data))
                        except Exception:
                            # If anything goes wrong with the ZIP, fall back to ignoring its contents
                            pass
                    else:
                        # Any other file type – keep behaviour the same as before
                        if uploaded_file.name not in existing_names:
                            st.session_state.uploaded_files.append(uploaded_file)
                            existing_names.append(uploaded_file.name)
                            new_files_added = True
                            auto_track_file_upload(uploaded_file.name, getattr(uploaded_file, "size", 0) or 0)
                if new_files_added:
                    # Clear heavy per-session caches when the file set changes
                    cleanup_stale_session_data()
                    st.rerun()
            # Load last 10 stored files (same org/user) to show in the file list
            _org_id = st.session_state.get("organization_id")
            _user_id = st.session_state.get("user_id")
            _recent_stored = []
            if _org_id and _user_id:
                try:
                    _recent_stored = list_recent_files(_org_id, _user_id, limit=10)
                except Exception:
                    _recent_stored = []
            _has_stored = len(_recent_stored) > 0
            _has_session = len(st.session_state.uploaded_files) > 0
            # Show file preview when we have stored files and/or session uploads
            if _has_stored or _has_session:
                # Header: File Preview & Management only (usage summary is in KPI card below)
                st.markdown("<h4 style='margin-top: 0px; color: #495057;'>📋 File Preview & Management</h4>", unsafe_allow_html=True)
                st.markdown("<h5 style='color: #6c757d; margin-bottom: 15px;'>📎 Uploaded Files</h5>", unsafe_allow_html=True)
                # Scrollable container: stored files (last 10) then session uploads
                with st.container(height=420):
                    # Stored files (last 10) — download only
                    for _idx, rec in enumerate(_recent_stored):
                        name = rec.get("original_filename") or "file"
                        size = rec.get("file_size") or 0
                        path = rec.get("storage_path")
                        up_at = rec.get("uploaded_at") or ""
                        if isinstance(up_at, str) and len(up_at) >= 10:
                            up_at = up_at[:10]
                        _r1, _r2 = st.columns([10, 2])
                        with _r1:
                            st.markdown(f"""
                            <div style="font-weight: 600; color: #495057;">
                                📄 {html.escape(name)}
                                <span style="font-size: 12px; color: #6c757d; margin-left: 10px;">
                                    Size: {(size or 0) / (1024*1024):.1f} MB · {up_at}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                        with _r2:
                            if path:
                                _url = get_download_url(path)
                                if _url:
                                    st.link_button("Download", _url)
                    # Session uploads: preview, rename, remove
                    for i, file in enumerate(st.session_state.uploaded_files):
                        file_ext = file.name.split('.')[-1].lower() if '.' in file.name else 'unknown'
                        file_type_badge = f"<span class='file-type-badge {file_ext}'>{file_ext}</span>"
                        file_cols = st.columns([12, 1, 1, 1])
                        with file_cols[0]:
                            st.markdown(f"""
                            <div style="font-weight: 600; color: #495057;">
                                📄 {file.name}
                                <span style="font-size: 12px; color: #6c757d; margin-left: 10px;">
                                    Size: {file.size / (1024*1024):.1f} MB | Type: {getattr(file, 'type', 'Unknown') or 'Unknown'} {file_type_badge}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                        with file_cols[1]:
                            if st.button("🔍", key=f"preview_btn_{i}", use_container_width=True, help="Quick Preview"):
                                st.session_state[f"preview_mode_{i}"] = not st.session_state.get(f"preview_mode_{i}", False)
                                st.rerun()
                        with file_cols[2]:
                            if st.button("✏️", key=f"rename_btn_{i}", use_container_width=True, help="Rename"):
                                st.session_state.file_rename_mode[i] = not st.session_state.file_rename_mode.get(i, False)
                                st.rerun()
                        with file_cols[3]:
                            if st.button("🗑️", key=f"remove_btn_{i}", use_container_width=True, help="Remove"):
                                st.session_state.uploaded_files.pop(i)
                                cleanup_stale_session_data()
                                st.session_state.uploader_key_counter = st.session_state.get("uploader_key_counter", 0) + 1
                                st.rerun()
                        # Quick Preview UI (full width below row)
                        if st.session_state.get(f"preview_mode_{i}", False):
                            with st.expander("Quick Preview", expanded=True):
                                file.seek(0)
                                file_ext = file.name.split('.')[-1].lower() if '.' in file.name else 'unknown'
                                if file_ext == "csv":
                                    try:
                                        df = pd.read_csv(file, nrows=10)
                                        st.dataframe(df, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Could not preview CSV: {e}")
                                else:
                                    st.warning("Preview not supported for this file type.")
                                file.seek(0)
                    # Rename UI (single row of columns at same level, not nested)
                    if st.session_state.file_rename_mode.get(i, False):
                        st.markdown("**✏️ Rename File**")
                        col_rename1, col_rename2, col_rename3 = st.columns([2, 1, 1])
                        with col_rename1:
                            new_name = st.text_input(
                                "New file name:",
                                value=file.name,
                                key=f"rename_input_{i}"
                            )
                        with col_rename2:
                            if st.button("✅ Save", key=f"save_rename_{i}", use_container_width=True):
                                if new_name and new_name != file.name:
                                    file.name = new_name
                                    st.success(f"File renamed to: {new_name}")
                                st.session_state.file_rename_mode[i] = False
                                st.rerun()
                        with col_rename3:
                            if st.button("❌ Cancel", key=f"cancel_rename_{i}", use_container_width=True):
                                st.session_state.file_rename_mode[i] = False
                                st.rerun()
            else:
                # Empty state (elite styling)
                st.markdown("""
                <div class="fm-empty-state">
                    <div style="font-size: 48px; margin-bottom: 16px;">📁</div>
                    <h4 style="color: #374151; margin-bottom: 8px; font-weight: 600;">No files uploaded yet</h4>
                    <p style="color: #6b7280; margin-bottom: 12px; font-size: 0.95rem;">Upload your data files to begin analysis</p>
                    <p style="font-size: 0.8rem; color: #9ca3af; letter-spacing: 0.02em;">Supported formats include CSV and compatible log files.</p>
                </div>
                """, unsafe_allow_html=True)

            # Submit button (elite spacing)
            st.markdown("""
            <div class="fm-submit-wrap"></div>
            """, unsafe_allow_html=True)
            col_center = st.columns([5, 1.7, 5])
            with col_center[1]:  # Use the center column
                if st.button("✅ Submit Files", type="primary", use_container_width=True):
                    st.session_state.files_submitted = True
                    st.session_state.show_upload_area = False
                    st.session_state.upload_opened_by_plus = False
                    st.rerun()

    # Only show analysis type and content on the analysis page (auth gate: require authenticated and approved)
    else:
        if not is_authenticated() or not is_approved():
            st.session_state.show_front_page = True
            st.session_state.show_upload_area = False
            st.session_state.show_calculators = False
            if not is_authenticated():
                st.session_state.show_login_form = True
            st.rerun()
        
        # st.session_state.files_submitted and not st.session_state.show_upload_area
        has_multiple_files = len(st.session_state.uploaded_files) >= 2
        options_analysis = ["Multi-Parameter Analysis", "Multi-File Comparison"]
        if has_multiple_files and SHOW_ANALYSIS_TYPE_SELECTOR:
            col1, _ = st.columns([8, 0.75])
            with col1:
                if (
                    "analysis_type_radio" not in st.session_state
                    or st.session_state["analysis_type_radio"] not in options_analysis
                ):
                    last_analysis_type = st.session_state.get("analysis_type", "Multi-Parameter Analysis")
                    if last_analysis_type not in options_analysis:
                        last_analysis_type = "Multi-Parameter Analysis"
                    st.session_state["analysis_type_radio"] = last_analysis_type
                analysis_type = st.radio(
                    "Choose the type of analysis you want to perform",
                    options_analysis,
                    key="analysis_type_radio",
                    horizontal=True,
                )
        else:
            analysis_type = "Multi-Parameter Analysis"
        st.session_state.analysis_type = analysis_type

        # ═══════════════════════════════════════════════════════════════
        # SECTION 5: ANALYSIS TYPE SELECTION & BATCH REPORT GENERATION
        # User picks "Multi-Parameter Analysis" (single file) or
        # "Multi-File Comparison". Batch mode generates reports for
        # all uploaded files sequentially with memory cleanup between.
        # ═══════════════════════════════════════════════════════════════

        if analysis_type == "Multi-Parameter Analysis":
            # Row 1: heading
            # st.markdown("### 📈 Multi-Parameter Analysis (Single File)")

            # File selection (single file) - stateless defaults for immediate updates
            file_options = ["None"] + [f.name for f in st.session_state.uploaded_files]

            # Build or refresh lightweight per-file insights used to enrich the
            # file selector labels (total runtime, max thrust, max RPM).
            insights_cache = st.session_state.multi_param_file_insights
            for f in st.session_state.uploaded_files:
                name = f.name
                file_ext_local = os.path.splitext(name)[-1].lower()
                # Use file size as a simple version token so renamed/overwritten
                # files will have their insights recomputed.
                size_token = getattr(f, "size", None)
                cached = insights_cache.get(name, {})
                if cached.get("_size_token") == size_token and all(
                    k in cached for k in ("runtime_s", "max_thrust", "max_rpm")
                ):
                    continue
                metrics = compute_basic_file_insights(f, file_ext_local)
                metrics["_size_token"] = size_token
                insights_cache[name] = metrics

            # Prepare labels for the selectbox; keep it clean and show only the
            # raw file name. Detailed metrics are shown in the insights strip
            # above the throttle summary row instead.
            def _format_file_option(option_name: str) -> str:
                if option_name == "None":
                    return "None"
                return option_name
            # Robust single-file selector using widget state (avoids double-selection)
            options_single = file_options
            
            # Get the previously selected file (preserved across mode switches)
            preserved_file = st.session_state.get("multi_param_selected_file", "None")
            
            
            # Determine the correct index to use
            # Priority: widget state value > preserved file > default (0)
            initial_index = 0
            widget_state_exists = "multi_param_file_selector" in st.session_state
            widget_state_value = st.session_state.get("multi_param_file_selector", None)
            
            # Validate and ensure index is within bounds
            max_index = len(options_single) - 1 if options_single else 0
            
            # Check if widget state value is valid and in options
            if widget_state_exists and widget_state_value is not None and widget_state_value in options_single:
                # Widget state exists and is valid - use it to determine index
                try:
                    initial_index = options_single.index(widget_state_value)
                    # Ensure index is within bounds
                    initial_index = min(initial_index, max_index)
                except (ValueError, AttributeError, IndexError):
                    # Fallback to preserved file or default
                    if preserved_file in options_single and preserved_file != "None":
                        try:
                            initial_index = options_single.index(preserved_file)
                            initial_index = min(initial_index, max_index)
                        except (ValueError, IndexError):
                            initial_index = 0
                    else:
                        initial_index = 0
            elif preserved_file in options_single and preserved_file != "None":
                # Widget state doesn't exist or is invalid - use preserved file
                try:
                    initial_index = options_single.index(preserved_file)
                    initial_index = min(initial_index, max_index)
                except (ValueError, IndexError):
                    initial_index = 0
            else:
                # Default to index 0 ("None")
                initial_index = 0
            
            # Ensure index is always valid
            if initial_index < 0 or initial_index > max_index:
                initial_index = 0
            
            # If widget state exists but value is not in options, clear it to avoid conflicts
            if widget_state_exists and widget_state_value is not None and widget_state_value not in options_single:
                del st.session_state["multi_param_file_selector"]

            # Report generation page (full page when user clicked Proceed from template settings)
            if st.session_state.get("report_generation_page", False):
                # Template settings preview in sidebar (always shown, dark text for readability)
                with st.sidebar:
                    st.markdown(
                        "<style>"
                        "[data-testid='stSidebar'] .stMarkdown, [data-testid='stSidebar'] p, "
                        "[data-testid='stSidebar'] .stCaptionContainer { color: #1f2937 !important; } "
                        "[data-testid='stSidebar'] .stCaptionContainer label { color: #374151 !important; }"
                        "</style>",
                        unsafe_allow_html=True,
                    )
                    st.subheader("Report generation")
                    _gen_profiles = _load_sorted_report_profiles()
                    _gen_sel_idx = st.session_state.get("report_profile_selected_idx", None)
                    _gen_has_template = isinstance(_gen_sel_idx, int) and 0 <= _gen_sel_idx < len(_gen_profiles)
                    st.markdown("**Template preview**")
                    if _gen_has_template:
                        _p = _gen_profiles[_gen_sel_idx]
                        _name = (_p.get("name") or "").strip() or "—"
                        _desc = (_p.get("description") or "").strip() or "—"
                        _s = "<div style='color:#1f2937;'>"
                        _s += "<p style='margin:0.35rem 0 0.15rem 0;'><strong>1. Template</strong></p>"
                        _s += f"<p style='color:#374151;margin:0 0 0 0.75rem;'>Name: {html.escape(_name)}</p>"
                        if _desc != "—":
                            _s += f"<p style='color:#374151;margin:0 0 0.5rem 0.75rem;'>Description: {html.escape(_desc)}</p>"
                        else:
                            _s += "<p style='margin-bottom:0.5rem;'></p>"
                        _th = _p.get("throttle_aggregation") or {}
                        _s += "<p style='margin:0.35rem 0 0.15rem 0;'><strong>2. Throttle</strong></p><ul style='color:#374151;margin:0 0 0.5rem 1rem;padding-left:1rem;'>"
                        if _th.get("start_throttle") is not None:
                            _s += f"<li>Start: {_th['start_throttle']}</li>"
                        if _th.get("end_throttle") is not None:
                            _s += f"<li>End: {_th['end_throttle']}</li>"
                        if _th.get("throttle_interval") is not None:
                            _s += f"<li>Interval: {_th['throttle_interval']}</li>"
                        if _th.get("ramp_mode"):
                            _s += f"<li>Ramp: {_th['ramp_mode']}</li>"
                        if not any([_th.get("start_throttle") is not None, _th.get("end_throttle") is not None, _th.get("throttle_interval") is not None, _th.get("ramp_mode")]):
                            _s += "<li>—</li>"
                        _s += "</ul>"
                        _s += "<p style='margin:0.35rem 0 0.15rem 0;'><strong>3. Plot</strong></p><ol style='color:#374151;margin:0 0 0 1rem;padding-left:1.25rem;'>"
                        for gi, g in enumerate((_p.get("saved_graphs") or [])[:6]):
                            _gx = g.get("x_axis") or "—"
                            _gl = g.get("left_y_axes") or []
                            _gr = g.get("right_y_axes") or []
                            _line = f"X={_gx}"
                            if _gl or _gr:
                                _line += f", L={_gl}, R={_gr}"
                            _s += f"<li style='margin:0.2rem 0;'>{html.escape(_line)}</li>"
                        _s += "</ol></div>"
                        st.markdown(_s, unsafe_allow_html=True)
                    else:
                        st.markdown("<p style='color:#6b7280;'>No template selected.</p>", unsafe_allow_html=True)
                _head_col, _back_col = st.columns([3, 1])
                with _head_col:
                    # Use the same styled header as the template settings page
                    st.markdown(
                        "<p style='font-size: 1.35rem; font-weight: 600; color: #262730; margin-bottom: 0.5rem;'>Report generation</p>",
                        unsafe_allow_html=True,
                    )
                with _back_col:
                    if st.button("← Back to template settings", key="report_gen_back_btn"):
                        st.session_state["report_generation_page"] = False
                        st.session_state.pop("report_gen_zip_bytes", None)
                        st.session_state.pop("report_gen_zip_filename", None)
                        st.rerun()

                # ── Toggle: Report | History ──
                _rg_mode = st.radio(
                    "View", ["📄 Report", "📋 History"],
                    horizontal=True, key="report_gen_mode", label_visibility="collapsed",
                )

                if _rg_mode == "📋 History":
                    _render_report_history_tab()
                    render_footer()
                    return

                st.markdown(
                    "Use the selected template to generate PDF reports. "
                    "Upload CSV or ZIP files on the left, then use the play button to start or stop report generation."
                )

                # Organization name on reports:
                # allow editing only for a specific REUDE-managed test account.
                _email_norm = (st.session_state.get("user_email") or "").strip().lower()
                _org_name = (st.session_state.get("organization_name") or "").strip()
                _is_reude_org = _org_name.lower() == "reude technologies"
                _can_edit_org_name = (
                    _email_norm == "testuser@reude.tech"
                    and _is_reude_org
                )
                _current_company = (st.session_state.get("author_company") or _org_name or "").strip()
                if _can_edit_org_name:
                    _edited_company = st.text_input(
                        "Organization Name (Report Cover)",
                        value=_current_company,
                        key="report_org_name_override",
                        help="Only this REUDE-managed test account can edit organization name for report generation.",
                    ).strip()
                    st.session_state["author_company"] = _edited_company or _org_name
                else:
                    # Keep the company value pinned to the profile/org-managed value.
                    st.session_state["author_company"] = _org_name or _current_company
                    if _current_company:
                        st.caption(f"Organization Name (managed): {_current_company}")

                # ── Daily report generation quota banner ──
                _quota_uid = st.session_state.get("user_id", "")
                _quota_oid = st.session_state.get("organization_id", "")
                _quota_limit = (
                    get_user_quota(_quota_uid, _quota_oid)
                    if _quota_uid
                    else DEFAULT_DAILY_REPORT_QUOTA
                )
                _quota_used = get_daily_report_count(_quota_uid) if _quota_uid else 0
                _quota_remaining = max(0, _quota_limit - _quota_used)
                _quota_pct = min(100, int(_quota_used / max(_quota_limit, 1) * 100))
                _quota_color = "#22c55e" if _quota_pct < 70 else ("#f59e0b" if _quota_pct < 90 else "#ef4444")

                if _quota_remaining == 0:
                    st.error(
                        f"🚫 **Daily report quota exhausted.** You have generated {_quota_used}/{_quota_limit} reports today. "
                        "Your quota resets tomorrow (UTC). Contact support to increase the limit."
                    )
                else:
                    _qc1, _qc2, _qc3 = st.columns(3)
                    _qc1.metric("📄 Generated today", _quota_used)
                    _qc2.metric("📋 Daily quota", _quota_limit)
                    _qc3.metric("🆓 Remaining", _quota_remaining)
                    st.markdown(f"""
                    <div style="margin-bottom: 0.5rem;">
                        <div style="background: #e5e7eb; border-radius: 6px; height: 10px; overflow: hidden;">
                            <div style="background: {_quota_color}; height: 100%; width: {_quota_pct}%; border-radius: 6px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Files available for report generation (all uploaded). New uploads are handled in the left column.
                existing_names = [f.name for f in st.session_state.uploaded_files]

                # Initialise report-generation session state
                if "report_gen_results" not in st.session_state:
                    st.session_state["report_gen_results"] = []
                if "report_gen_queue" not in st.session_state:
                    st.session_state["report_gen_queue"] = []
                if "report_gen_index" not in st.session_state:
                    st.session_state["report_gen_index"] = 0
                if "report_gen_running" not in st.session_state:
                    st.session_state["report_gen_running"] = False

                # Three-column layout: left = files, center = play/stop, right = generated reports
                left_col, center_col, right_col = st.columns([5, .6, 5])

                # LEFT COLUMN: uploads + selectable file list
                with left_col:
                    # st.markdown("#### Files")
                    _rg_uploader_key = f"report_gen_file_uploader_{st.session_state.get('uploader_key_counter', 0)}"
                    extra_uploads = st.file_uploader(
                        "Upload files (optional)",
                        type=["csv", "xlsx", "zip"],
                        accept_multiple_files=True,
                        key=_rg_uploader_key,
                        help=(
                            "CSV, Excel (.xlsx), or ZIP. Drop files or ZIPs (CSV inside ZIPs are extracted). "
                            "Select multiple or a folder if your browser allows."
                        ),
                    )
                    if extra_uploads:
                        for up in extra_uploads:
                            name_lower = (up.name or "").lower()
                            if name_lower.endswith(".csv") or name_lower.endswith(".xlsx"):
                                if up.name not in existing_names:
                                    st.session_state.uploaded_files.append(up)
                                    existing_names.append(up.name)
                                    auto_track_file_upload(up.name, getattr(up, "size", 0) or 0)
                            elif name_lower.endswith(".zip"):
                                try:
                                    up.seek(0)
                                    zip_bytes = up.read()
                                    up.seek(0)
                                    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zf:
                                        for entry in zf.namelist():
                                            if entry.endswith("/") or entry.rstrip("/").rstrip("\\").endswith("/"):
                                                continue
                                            base = os.path.basename(entry.rstrip("/"))
                                            if not base or base.startswith("."):
                                                continue
                                            if os.path.splitext(base)[-1].lower() != ".csv":
                                                continue
                                            try:
                                                data = zf.read(entry)
                                            except Exception:
                                                continue
                                            if base not in existing_names:
                                                st.session_state.uploaded_files.append(_BytesFile(base, data))
                                                existing_names.append(base)
                                                auto_track_file_upload(base, len(data))
                                except Exception as e:
                                    st.warning(f"Could not read ZIP '{up.name}': {e}")

                    gen_files = [f.name for f in st.session_state.uploaded_files]
                    if not gen_files:
                        st.info(
                            "Upload files here or go back to template settings after uploading on the main page."
                        )
                        render_footer()
                        return

                    # Header row for source files: Select all + Edit/Remove buttons on the same row
                    col_sel, col_space, col_edit, col_rm = st.columns([1, 1, 1, 1])

                    with col_sel:
                        # Select all <-> individual file checkboxes two-way sync
                        prev_select_all = st.session_state.get("report_gen_select_all_prev", False)
                        all_selected = all(
                            st.session_state.get(f"report_gen_cb_{i}", True)
                            for i in range(len(gen_files))
                        )
                        if prev_select_all and not all_selected:
                            # User unchecked at least one file -> only "Select all" becomes unchecked; leave other files as-is
                            st.session_state["report_gen_select_all"] = False
                        select_all = st.checkbox("Select all", key="report_gen_select_all")
                        if select_all:
                            for i in range(len(gen_files)):
                                st.session_state[f"report_gen_cb_{i}"] = True
                        elif prev_select_all and all_selected:
                            # User just unchecked "Select all" -> uncheck all files
                            for i in range(len(gen_files)):
                                st.session_state[f"report_gen_cb_{i}"] = False

                    # Determine how many source files are currently selected (after select-all logic)
                    selected_file_indices = [
                        i
                        for i in range(len(gen_files))
                        if st.session_state.get(f"report_gen_cb_{i}", False)
                    ]

                    # Action buttons only visible when there is a selection
                    if selected_file_indices:
                        with col_edit:
                            if len(selected_file_indices) == 1:
                                if st.button("Edit", key="report_gen_files_edit_btn"):
                                    st.session_state["report_gen_active_file_index"] = selected_file_indices[0]
                        with col_rm:
                            if st.button("Remove", key="report_gen_files_remove_btn"):
                                # Remove all selected files from the upload list
                                remaining_files = [
                                    f
                                    for i, f in enumerate(st.session_state.uploaded_files)
                                    if i not in selected_file_indices
                                ]
                                st.session_state.uploaded_files = remaining_files
                                cleanup_stale_session_data()
                                # Clear corresponding checkbox states
                                for i in range(len(gen_files)):
                                    st.session_state.pop(f"report_gen_cb_{i}", None)
                                st.session_state.uploader_key_counter = st.session_state.get("uploader_key_counter", 0) + 1
                                st.rerun()

                    # When selection is no longer exactly one, exit edit mode so the edit UI hides
                    if len(selected_file_indices) != 1 and "report_gen_active_file_index" in st.session_state:
                        st.session_state.pop("report_gen_active_file_index", None)

                    # Inline rename UI when a single file has been marked for editing (shown before the list)
                    active_file_index = st.session_state.get("report_gen_active_file_index", None)
                    if isinstance(active_file_index, int) and 0 <= active_file_index < len(gen_files):
                        current_name = gen_files[active_file_index]
                        new_name = st.text_input(
                            "Edit selected file name",
                            value=current_name,
                            key=f"report_gen_edit_name_{active_file_index}",
                        )
                        if st.button("Save name", key="report_gen_files_save_name_btn"):
                            try:
                                file_obj = st.session_state.uploaded_files[active_file_index]
                                if hasattr(file_obj, "name"):
                                    file_obj.name = new_name
                            except Exception:
                                pass
                            st.session_state.pop("report_gen_active_file_index", None)
                            st.rerun()

                    with st.container(height=430):
                        st.caption("Select files for report generation:")
                        for i, fname in enumerate(gen_files):
                            st.checkbox(fname, key=f"report_gen_cb_{i}")

                    st.session_state["report_gen_select_all_prev"] = st.session_state.get(
                        "report_gen_select_all", False
                    )
                    selected_for_report = [
                        gen_files[i]
                        for i in range(len(gen_files))
                        if st.session_state.get(f"report_gen_cb_{i}", False)
                    ]

                # CENTER COLUMN: play / stop control
                with center_col:
                    # Extra vertical spacer so the play/stop control sits a bit lower
                    st.markdown("<div style='height:20rem'></div>", unsafe_allow_html=True)
                    # st.markdown("#### Control")
                    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
                    st.markdown('<div class="report-gen-control-wrap">', unsafe_allow_html=True)
                    running = st.session_state.get("report_gen_running", False)
                    queue = st.session_state.get("report_gen_queue", [])

                    if running:
                        if st.button("■", key="report_gen_stop_btn", help="Stop report generation"):
                            # Immediately stop any further processing
                            st.session_state["report_gen_running"] = False
                            st.session_state["report_gen_queue"] = []
                            st.session_state["report_gen_index"] = 0
                            # Use st.rerun() + return to stop the current pass
                            # The rerun will see report_gen_running=False and skip processing
                            st.rerun()
                    else:
                        disabled = not selected_for_report
                        if st.button(
                            "▶",
                            key="report_gen_play_btn",
                            disabled=disabled,
                            type="primary",
                            help="Start report generation",
                        ):
                            # If the user is not yet approved, show the approval screen instead of starting generation.
                            if not is_approved():
                                render_pending_approval_screen()
                                st.stop()
                            _gen_uid = st.session_state.get("user_id", "")
                            _gen_oid = st.session_state.get("organization_id", "")
                            can_run = True
                            if _gen_uid and _gen_oid:
                                _remaining = get_remaining_quota(_gen_uid, _gen_oid)
                                num_selected = len(selected_for_report)
                                if _remaining <= 0:
                                    import uuid
                                    uid = uuid.uuid4().hex[:6]
                                    st.markdown(
                                        f'''
                                        <style>
                                        @keyframes fadeOut_{uid} {{ 0% {{ opacity: 1; }} 80% {{ opacity: 1; }} 100% {{ opacity: 0; display: none; visibility: hidden; }} }}
                                        .quota-popup-{uid} {{
                                            position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                                            background-color: white; color: black; padding: 25px 40px; border-radius: 12px;
                                            box-shadow: 0 8px 24px rgba(0,0,0,0.2); font-size: 1.3rem; font-weight: 500;
                                            z-index: 9999999; text-align: center; border-left: 8px solid #ff4b4b;
                                            animation: fadeOut_{uid} 5s forwards; pointer-events: none; white-space: nowrap;
                                        }}
                                        </style>
                                        <div class="quota-popup-{uid}">🚫 Daily report quota reached. Please wait for the next day or contact support.</div>
                                        ''',
                                        unsafe_allow_html=True
                                    )
                                    can_run = False
                                elif num_selected > _remaining:
                                    import uuid
                                    uid = uuid.uuid4().hex[:6]
                                    st.markdown(
                                        f'''
                                        <style>
                                        @keyframes fadeOut_{uid} {{ 0% {{ opacity: 1; }} 80% {{ opacity: 1; }} 100% {{ opacity: 0; display: none; visibility: hidden; }} }}
                                        .quota-popup-{uid} {{
                                            position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                                            background-color: white; color: black; padding: 25px 40px; border-radius: 12px;
                                            box-shadow: 0 8px 24px rgba(0,0,0,0.2); font-size: 1.3rem; font-weight: 500;
                                            z-index: 9999999; text-align: center; border-left: 8px solid #fbdc00;
                                            animation: fadeOut_{uid} 5s forwards; pointer-events: none; white-space: nowrap;
                                        }}
                                        </style>
                                        <div class="quota-popup-{uid}">⚠️ You can only select up to <b>{_remaining}</b> files based on your remaining daily quota.</div>
                                        ''',
                                        unsafe_allow_html=True
                                    )
                                    can_run = False
                            
                            if can_run:
                                # Start a new run based on current selection, appending to any existing reports
                                st.session_state["report_gen_queue"] = list(selected_for_report)
                                st.session_state["report_gen_index"] = 0
                                st.session_state["report_gen_running"] = True
                                # Force an immediate rerun so the center control switches
                                # from Play to Stop before processing begins.
                                st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Simple textual status, centered under the button
                    queue = st.session_state.get("report_gen_queue", [])
                    idx = st.session_state.get("report_gen_index", 0)
                    if queue:
                        if running and idx < len(queue):
                            st.caption(f"Generating {idx + 1} of {len(queue)}")
                        elif not running and idx >= len(queue) and len(queue) > 0:
                            st.caption("Generation complete.")

                # RIGHT COLUMN: generated reports list and download-all control
                with right_col:
                    st.markdown("#### Generated reports")
                    results = st.session_state.get("report_gen_results", [])
                    # Load user's saved reports (from storage) to show in the same list
                    _ro = st.session_state.get("organization_id")
                    _ru = st.session_state.get("user_id")
                    saved_reports = []
                    if _ro and _ru:
                        try:
                            saved_reports = list_reports_for_user(_ro, _ru, limit=50)
                        except Exception:
                            saved_reports = []
                    if not results and not saved_reports:
                        with st.container(height=500):
                            st.caption("Reports will appear here as they are generated.")
                    else:
                        # Header row: Select all + download type + Remove + Download button (one row)
                        col_sel_r, col_mode_r, col_rm_r, col_dl_r = st.columns([1, 2, 1, 2])

                        with col_sel_r:
                            # Select all <-> individual report checkboxes two-way sync
                            prev_results_select_all = st.session_state.get(
                                "report_gen_results_select_all_prev", False
                            )
                            all_results_selected = all(
                                st.session_state.get(f"report_gen_result_cb_{i}", True)
                                for i in range(len(results))
                            )
                            if prev_results_select_all and not all_results_selected:
                                # User unchecked at least one report -> only "Select all" becomes unchecked
                                st.session_state["report_gen_results_select_all"] = False

                            select_all_results = st.checkbox(
                                "Select all", key="report_gen_results_select_all"
                            )
                            if select_all_results:
                                for i in range(len(results)):
                                    st.session_state[f"report_gen_result_cb_{i}"] = True
                            elif prev_results_select_all and all_results_selected:
                                # User just unchecked "Select all" -> uncheck all reports
                                for i in range(len(results)):
                                    st.session_state[f"report_gen_result_cb_{i}"] = False

                        with col_mode_r:
                            # Choose which artefacts to include when downloading reports
                            download_mode = st.radio(
                                "Download as",
                                ["PDF", "CSV", "Both"],
                                index=0,
                                horizontal=True,
                                key="report_gen_download_mode",
                            )

                        # Download/remove controls for generated reports
                        selected_result_indices = [
                            i
                            for i in range(len(results))
                            if st.session_state.get(f"report_gen_result_cb_{i}", False)
                        ]

                        # Only show buttons when there is at least one selected report
                        if selected_result_indices:
                            with col_rm_r:
                                if st.button(
                                    "Remove",
                                    key="report_gen_results_remove_btn",
                                ):
                                    remaining = [
                                        r
                                        for i, r in enumerate(results)
                                        if i not in selected_result_indices
                                    ]
                                    st.session_state["report_gen_results"] = remaining
                                    for i in range(len(results)):
                                        st.session_state.pop(f"report_gen_result_cb_{i}", None)
                                    st.rerun()

                        with col_dl_r:
                                # Build download payload based on selection count and mode
                                selected_reports = [
                                    r
                                    for i, r in enumerate(results)
                                    if i in selected_result_indices and (r.get("pdf_bytes") or r.get("s3_pdf_key"))
                                ]
                                include_pdf = download_mode in ("PDF", "Both")
                                include_csv = download_mode in ("CSV", "Both")
                                if not selected_reports or not (include_pdf or include_csv):
                                    pass
                                # Case 1: single report, single format -> direct file download (no ZIP)
                                elif len(selected_reports) == 1 and (include_pdf ^ include_csv):
                                    # If the user is not yet approved, redirect them to the approval screen.
                                    if not is_approved():
                                        render_pending_approval_screen()
                                        st.stop()
                                    r = selected_reports[0]
                                    base = os.path.splitext(str(r["filename"]))[0]
                                    if include_pdf:
                                        pdf_bytes = r.get("pdf_bytes")
                                        if not pdf_bytes and r.get("s3_pdf_key"):
                                            pdf_bytes = download_file(r["s3_pdf_key"]) or None
                                        if pdf_bytes:
                                            st.download_button(
                                                "Download PDF",
                                                data=pdf_bytes,
                                                file_name=f"{base}.pdf",
                                                mime="application/pdf",
                                                key="report_gen_results_download_pdf_btn",
                                                use_container_width=True,
                                            )
                                    elif include_csv:
                                        csv_bytes = r.get("csv_bytes")
                                        if not csv_bytes and r.get("s3_csv_key"):
                                            csv_bytes = download_file(r["s3_csv_key"]) or None
                                        if csv_bytes:
                                            st.download_button(
                                                "Download CSV",
                                                data=csv_bytes,
                                                file_name=f"{base}_sorted.csv",
                                                mime="text/csv",
                                                key="report_gen_results_download_csv_btn",
                                                use_container_width=True,
                                            )
                                # Case 2: multiple reports OR both formats -> ZIP
                                else:
                                    # If the user is not yet approved, redirect them to the approval screen.
                                    if not is_approved():
                                        render_pending_approval_screen()
                                        st.stop()
                                    zip_buffer = BytesIO()
                                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                                        for r in selected_reports:
                                            base = os.path.splitext(str(r["filename"]))[0]
                                            if include_pdf:
                                                pdf_bytes = r.get("pdf_bytes")
                                                if not pdf_bytes and r.get("s3_pdf_key"):
                                                    pdf_bytes = download_file(r["s3_pdf_key"], silent=True) or None
                                                if pdf_bytes:
                                                    pdf_path = f"{base}/{base}.pdf"
                                                    zf.writestr(pdf_path, pdf_bytes)
                                            if include_csv:
                                                csv_bytes = r.get("csv_bytes")
                                                if not csv_bytes and r.get("s3_csv_key"):
                                                    csv_bytes = download_file(r["s3_csv_key"], silent=True) or None
                                                if csv_bytes:
                                                    csv_path = f"{base}/sorted_performance_table.csv"
                                                    zf.writestr(csv_path, csv_bytes)
                                    zip_buffer.seek(0)
                                    zip_bytes = zip_buffer.getvalue()

                                    # Company-based filename: CompanyName_report_YYYYMMDD.zip
                                    company = (st.session_state.get("author_company") or "Reports").strip()
                                    if not company:
                                        company = "Reports"
                                    company_safe = company.replace(" ", "_")
                                    date_str = datetime.now().strftime("%Y%m%d")
                                    zip_name = f"{company_safe}_report_{date_str}.zip"

                                    st.download_button(
                                        "Download selected reports",
                                        data=zip_bytes,
                                        file_name=zip_name,
                                        mime="application/zip",
                                        key="report_gen_results_download_btn",
                                        use_container_width=True,
                                    )
                        with st.container(height=500):
                            st.caption("Reports ready for download:")
                            for i, res in enumerate(results):
                                fname = res.get("filename", "Unknown file")
                                status = res.get("status", "done")
                                # Show only the base name (without extension)
                                base_name = os.path.splitext(fname)[0] if isinstance(fname, str) else str(fname)
                                label = f"{base_name} — Error" if status == "error" else base_name
                                st.checkbox(label, key=f"report_gen_result_cb_{i}")

                        st.session_state["report_gen_results_select_all_prev"] = st.session_state.get(
                            "report_gen_results_select_all", False
                        )

                # Incremental generation: process one file per run while running
                # Re-check the running flag — the Stop button sets it to False
                # and calls st.rerun(), but the generation's own st.rerun()
                # can race. This guard ensures Stop takes effect immediately.
                if st.session_state.get("report_gen_running", False):
                    queue = st.session_state.get("report_gen_queue", [])
                    idx = st.session_state.get("report_gen_index", 0)

                    # Live quota enforcement: re-check before each file
                    _gen_uid = st.session_state.get("user_id", "")
                    _gen_oid = st.session_state.get("organization_id", "")
                    if _gen_uid and _gen_oid:
                        _gen_remaining = get_remaining_quota(_gen_uid, _gen_oid)
                        if _gen_remaining <= 0:
                            st.session_state["report_gen_running"] = False
                            st.session_state["report_gen_queue"] = []
                            st.session_state["report_gen_index"] = 0
                            st.warning("🚫 Daily report quota reached. Generation stopped.")
                            st.rerun()

                    # Bail out if Stop was pressed between reruns
                    if not st.session_state.get("report_gen_running", False):
                        pass
                    elif queue and idx < len(queue):
                        fname = queue[idx]
                        try:
                            # Use the currently selected template's throttle settings, if any,
                            # so batch report generation always honours the template configuration.
                            _profiles_for_gen = _load_sorted_report_profiles()
                            _sel_for_gen = st.session_state.get("report_profile_selected_idx", None)
                            _th_override = None
                            if (
                                isinstance(_sel_for_gen, int)
                                and 0 <= _sel_for_gen < len(_profiles_for_gen)
                            ):
                                _prof = _profiles_for_gen[_sel_for_gen]
                                _th = _prof.get("throttle_aggregation") or {}
                                if isinstance(_th, dict):
                                    _th_override = {
                                        "start_throttle": _th.get("start_throttle"),
                                        "end_throttle": _th.get("end_throttle"),
                                        "throttle_interval": _th.get("throttle_interval"),
                                        "ramp_mode": _th.get("ramp_mode"),
                                    }

                            _raw_df, _sorted_df = _load_file_and_build_report_data(
                                fname,
                                st.session_state.uploaded_files,
                                throttle_override=_th_override,
                            )
                            if _raw_df is not None:
                                st.session_state["multi_param_raw_df"] = _raw_df
                                st.session_state["report_sorted_table_df"] = _sorted_df
                                st.session_state["multi_param_selected_file"] = fname
                                st.session_state["report_loaded_file_name"] = fname
                                _has_sorted = _sorted_df is not None and not _sorted_df.empty
                                if _has_sorted:
                                    st.session_state["report_graphs_data_source"] = (
                                        "Sorted performance table"
                                    )
                                _report_raw = _build_report_raw_data_df(
                                    _raw_df,
                                    throttle_override=_th_override,
                                )
                                st.session_state["report_raw_data_df"] = _report_raw
                                ensure_summary_graphs_for_current_file()
                                ensure_report_graphs_for_current_config()
                                _graph_entries = getattr(
                                    st.session_state, "report_graph_entries", {}
                                ) or {}
                                _graph_keys = list(_graph_entries.keys())
                                _table_keys = (
                                    ["Sorted Performance Table"]
                                    if _sorted_df is not None and not _sorted_df.empty
                                    else []
                                )
                                _company = (st.session_state.get("author_company") or "").strip()
                                _user = (st.session_state.get("user_name") or st.session_state.get("author_name") or "").strip()
                                _org_id = st.session_state.get("organization_id")
                                _logo_path = get_org_logo_path(_org_id) if _org_id else None
                                # Prefer auto-detected file info; otherwise, fall back to manual text if provided.
                                _auto_info = (st.session_state.get("report_file_info_text") or "").strip()
                                _manual_info = (st.session_state.get("manual_file_info_text") or "").strip()
                                if _manual_info and not _auto_info:
                                    st.session_state.report_file_info_text = _manual_info
                                _pdf_bytes = build_report_pdf(
                                    include_info=True,
                                    selected_graph_keys=_graph_keys,
                                    selected_table_keys=_table_keys,
                                    include_cover_page=True,
                                    include_table_of_contents=True,
                                    cover_company_name=_company,
                                    cover_user_name=_user,
                                    cover_logo_path=_logo_path,
                                )
                            else:
                                _pdf_bytes = None
                        except Exception:
                            _pdf_bytes = None

                        results = st.session_state.get("report_gen_results", [])
                        if _pdf_bytes and len(_pdf_bytes) > 0:
                            csv_bytes = None
                            if _sorted_df is not None and not _sorted_df.empty:
                                try:
                                    csv_bytes = _sorted_df.to_csv(index=False).encode("utf-8")
                                except Exception:
                                    csv_bytes = None
                            base_name = fname.replace(".csv", "").replace(".ulg", "")
                            s3_pdf_key = None
                            s3_csv_key = None
                            # Upload via storage.py (org/user-scoped paths + metadata)
                            _uid = st.session_state.get("user_id", "")
                            _oid = st.session_state.get("organization_id", "")
                            if _uid and _oid and _pdf_bytes:
                                _pdf_path, _csv_path = upload_report(
                                    _pdf_bytes, csv_bytes, base_name, _uid, _oid
                                )
                                s3_pdf_key = _pdf_path
                                s3_csv_key = _csv_path
                                if s3_pdf_key:
                                    _pdf_bytes = None
                                    csv_bytes = None
                            results.append(
                                {
                                    "filename": fname,
                                    "status": "done",
                                    "pdf_bytes": _pdf_bytes,
                                    "csv_bytes": csv_bytes,
                                    "s3_pdf_key": s3_pdf_key,
                                    "s3_csv_key": s3_csv_key,
                                }
                            )
                            auto_track_report(fname, fname)
                        else:
                            results.append(
                                {"filename": fname, "status": "error", "pdf_bytes": None, "csv_bytes": None, "s3_pdf_key": None, "s3_csv_key": None}
                            )
                        st.session_state["report_gen_results"] = results

                        # Free memory before next report (critical on low-RAM instances e.g. 2.4 GB)
                        st.session_state.pop("multi_param_raw_df", None)
                        st.session_state.pop("report_sorted_table_df", None)
                        st.session_state.pop("report_raw_data_df", None)
                        _pdf_bytes = None
                        csv_bytes = None
                        _raw_df = None
                        _sorted_df = None
                        gc.collect()

                        st.session_state["report_gen_index"] = idx + 1
                        if st.session_state["report_gen_index"] >= len(queue):
                            st.session_state["report_gen_running"] = False
                        else:
                            # Short delay before next file to reduce sustained CPU load during bulk generation
                            time.sleep(1.5)
                        st.rerun()

                render_footer()
                return

            # ═══════════════════════════════════════════════════════════
            # SECTION 6: SINGLE-FILE PERFORMANCE ASSESSMENT
            # Left column: file selector + PDF preview
            # Right column: report config (color scheme, cover page, etc.)
            # Sub-tabs: Summary (throttle graphs), Data (raw + sorted),
            #           Plot (custom graph builder with up to 5 graphs)
            # ═══════════════════════════════════════════════════════════

                # Report layout: RotriDash with left preview column (file selector + cover) and right config column
            st.markdown("## RotriDash")
            st.markdown("Generate and manage performance assessment reports from your data.")
            col_preview, col_right = st.columns([0.3, 0.7])
            with col_preview:
                # st.markdown("<p style='font-size: 1.35rem; font-weight: 600; color: #262730; margin-bottom: 0.5rem;'>Preview</p>", unsafe_allow_html=True)
                try:
                    selected_file = st.selectbox(
                        "Select File for Report Preview",
                        options_single,
                        index=initial_index,
                        key="multi_param_file_selector",
                        format_func=_format_file_option,
                    )
                except Exception:
                    if "multi_param_file_selector" in st.session_state:
                        del st.session_state["multi_param_file_selector"]
                    selected_file = st.selectbox(
                        "Select File",
                        options_single,
                        index=0,
                        key="multi_param_file_selector",
                    )
                # Left column: show template preview (generated PDF pages) when available; otherwise placeholder cover + hint
                _preview_images = st.session_state.get("report_preview_images", [])
                _preview_pdf_bytes = st.session_state.get("report_preview_pdf_bytes")
                if _preview_images:
                    with st.container():
                        for _i, _img_bytes in enumerate(_preview_images):
                            if _img_bytes:
                                try:
                                    st.image(BytesIO(_img_bytes), use_container_width=True)
                                except Exception:
                                    st.image(BytesIO(_img_bytes))
                elif _preview_pdf_bytes:
                    b64_pdf = base64.b64encode(_preview_pdf_bytes).decode("utf-8")
                    st.markdown(
                        f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="800" type="application/pdf" style="border:1px solid #e0e0e0;border-radius:8px;"></iframe>',
                        unsafe_allow_html=True,
                    )
                else:
                    # Cover page placeholder: PNG cover image with company name in the "for" blank space, in a container
                    _app_dir = os.path.dirname(os.path.abspath(__file__))
                    _parent_dir = os.path.dirname(_app_dir)
                    # Always use the neutral "Cover Page.png" for the placeholder.
                    _cover_names = ["Cover Page.png"]
                    _cover_png = None
                    for _d in (_app_dir, _parent_dir):
                        for _name in _cover_names:
                            _c = os.path.join(_d, _name)
                            if os.path.exists(_c):
                                _cover_png = _c
                                break
                        if _cover_png is not None:
                            break
                    _company = (st.session_state.get("author_company") or "").strip()
                    _org_id = st.session_state.get("organization_id")
                    _logo_path = get_org_logo_path(_org_id) if _org_id else None
                    _cover_ok = _cover_png is not None
                    if _cover_ok:
                        # When no company name and no logo, show image by path (most reliable);
                        # otherwise compose with PIL and show buffer (logo + company name).
                        if not _company and not _logo_path:
                            with st.container():
                                st.image(_cover_png)
                        else:
                            try:
                                from PIL import Image, ImageDraw, ImageFont
                                img = Image.open(_cover_png).convert("RGBA")
                                w, h = img.size

                                # Overlay organization logo on the same relative position
                                # as in the PDF cover if a logo file is available.
                                if _logo_path and os.path.exists(_logo_path):
                                    try:
                                        logo_img = Image.open(_logo_path).convert("RGBA")
                                        # Map from A4 mm coordinates (210 x 297) to pixels (match PDF cover layout).
                                        scale_x = w / 210.0
                                        scale_y = h / 297.0
                                        # Use a slightly smaller logo than the PDF so
                                        # the on-screen placeholder matches the preview.
                                        logo_w_px = int(80 * scale_x)
                                        logo_h_px = int(50 * scale_y)
                                        x_logo = int((w - logo_w_px) / 2)
                                        # 8 mm margin from the top (image coordinates are top‑left origin).
                                        y_logo = int(30 * scale_y)
                                        logo_img = logo_img.resize((logo_w_px, logo_h_px), Image.LANCZOS)
                                        img.alpha_composite(logo_img, (x_logo, y_logo))
                                    except Exception:
                                        pass

                                # Overlay company name text in the "for" area, when provided.
                                if _company:
                                    try:
                                        draw = ImageDraw.Draw(img)
                                        font_size = max(28, min(w, h) // 22)
                                        font = None
                                        for _path in [
                                            os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arial.ttf"),
                                            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                            "arial.ttf",
                                        ]:
                                            if _path and os.path.exists(_path):
                                                try:
                                                    font = ImageFont.truetype(_path, font_size)
                                                    break
                                                except OSError:
                                                    continue
                                        if font is None:
                                            font = ImageFont.load_default()
                                        text_x, text_y = w // 2, int(h * 0.52)
                                        if hasattr(draw, "textbbox"):
                                            bbox = draw.textbbox((0, 0), _company, font=font)
                                            tw = bbox[2] - bbox[0]
                                        else:
                                            tw, _ = draw.textsize(_company, font=font)
                                        draw.text((text_x - tw // 2, text_y), _company, fill=(0, 0, 80, 255), font=font)
                                    except Exception:
                                        pass

                                buf = BytesIO()
                                img.convert("RGB").save(buf, format="PNG")
                                buf.seek(0)
                                with st.container():
                                    st.image(buf)
                            except Exception as _e:
                                # Fallback: show image without overlay
                                with st.container():
                                    st.image(_cover_png)
                    if not _cover_ok:
                        with st.container():
                            st.markdown(
                                """
                                <div style="
                                    border: 1px solid #e0e0e0;
                                    border-radius: 8px;
                                    overflow-y: auto;
                                    min-height: 400px;
                                    max-height: 75vh;
                                    background: #f8f9fa;
                                    padding: 1rem;
                                    font-size: 0.9rem;
                                    color: #374151;
                                ">
                                    <div style="text-align: center; padding: 2rem 1rem;">
                                        <p style="font-weight: 600; margin-bottom: 0.5rem;">Cover page</p>
                                        <p style="margin: 0;">Cover Page.png (placeholder)</p>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    st.caption("Select a template or configure Add New, then click **Update report preview** to generate the full report preview.")

            # Persist the current selection so it can be restored after switching modes
            # Only update if a valid file is selected (not "None")
            if selected_file != "None":
                st.session_state.multi_param_selected_file = selected_file
            else:
                st.session_state.multi_param_file_selection = None

            # Load file and prepare data only when a file is selected
            if selected_file and selected_file != "None":
                # Update the old session state variable for compatibility
                st.session_state.multi_param_file_selection = selected_file

                file_ext = os.path.splitext(selected_file)[-1].lower()
                try:
                    file = [f for f in st.session_state.uploaded_files if f.name == selected_file][0]
                except IndexError:
                    st.error("Selected file not found. Please re-select the file.")
                    render_footer()
                    return

                file.seek(0)
                content = file.read()
                file.seek(0)

                df = None
                selected_topic_name = None

                # Load data depending on file type
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    try:
                        if isinstance(content, str):
                            tmp_file.write(content.encode("utf-8"))
                        else:
                            tmp_file.write(content)
                        tmp_file.flush()

                        if file_ext == ".ulg":
                            dfs_dict, topics = load_ulog(tmp_file.name)
                            if not dfs_dict:
                                st.error("No usable topics found in the log file.")
                                render_footer()
                                return

                            # Map topics to assessment names where possible, but allow raw topic selection too
                            topic_keys = list(dfs_dict.keys())
                            topic_display = topic_keys
                            default_topic = st.session_state.get("multi_param_ulog_topic")
                            default_index = topic_keys.index(default_topic) if default_topic in topic_keys else 0

                            selected_topic_name = st.selectbox(
                                "Select Topic",
                                topic_display,
                                index=default_index,
                                key="multi_param_ulog_topic_selector",
                            )
                            st.session_state["multi_param_ulog_topic"] = selected_topic_name
                            df = dfs_dict[selected_topic_name].copy()
                        else:
                            df, _ = load_data(tmp_file.name, file_ext, key_suffix="_multi_param")
                            if df is None or df.empty:
                                st.error("No data found in selected file.")
                                render_footer()
                                return
                    finally:
                        try:
                            os.unlink(tmp_file.name)
                        except Exception:
                            pass

                # Ensure numeric data and columns
                if df is None or df.empty:
                    st.error("No data available after loading the file.")
                    render_footer()
                    return

                df = ensure_seconds_column(df)
                # Add Index column if it doesn't exist (needed for CSV files)
                if 'Index' not in df.columns:
                    df.insert(0, 'Index', range(1, len(df) + 1))
                # Include Index and timestamp_seconds so Time (s) is available for Graph 4/5 presets
                numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                if not numeric_cols:
                    st.error("No numeric columns available for multi-parameter analysis.")
                    render_footer()
                    return

                # Add derived Throttle - % column for raw data if missing (from Servo/ESC PWM), at end of table
                throttle_col_existing = find_column_by_pattern(df, ["Throttle - %", "Throttle Input (%)", "Throttle (%)", "Throttle", "throttle"])
                if throttle_col_existing is None:
                    pwm_col = None
                    for col in df.columns:
                        col_lower = str(col).lower()
                        if "servo" in col_lower or "esc" in col_lower:
                            series = pd.to_numeric(df[col], errors="coerce")
                            if series.notna().sum() == 0:
                                continue
                            valid_mask = series.between(800, 2200)
                            if valid_mask.mean() >= 0.5:
                                pwm_col = col
                                break
                    if pwm_col is not None:
                        pwm_series = pd.to_numeric(df[pwm_col], errors="coerce")
                        pwm_min, pwm_max = 1000.0, 2000.0
                        throttle_series = (pwm_series - pwm_min) / (pwm_max - pwm_min) * 100.0
                        throttle_series = throttle_series.clip(lower=0.0, upper=100.0)
                        new_throttle_col = "Throttle - %"
                        if new_throttle_col in df.columns:
                            new_throttle_col = "Throttle - %_derived"
                        df[new_throttle_col] = throttle_series
                        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

                # Single source of truth for raw data: Plot tab and report both use this when "Raw data" is selected.
                st.session_state["multi_param_raw_df"] = df.copy()

                # Helper: find column by name (v1-style). Accepts single string or list of alternatives (for renamed columns e.g. DWC1).
                def find_column_by_name(cols, target_name_or_list):
                    """Find a column that contains the target name (case-insensitive). target_name_or_list can be str or list of str (tried in order)."""
                    names = [target_name_or_list] if isinstance(target_name_or_list, str) else target_name_or_list
                    for target_name in names:
                        target_lower = target_name.lower()
                        for col in cols:
                            if col.lower() == target_lower:
                                return col
                        for col in cols:
                            if target_lower in col.lower():
                                return col
                    return None

                # RotriX-style datasets should prefer elapsed time on X-axis.
                _df_col_lowers = {str(c).strip().lower() for c in df.columns}
                is_rotrix_data = any(
                    key in _df_col_lowers
                    for key in {"testrecordid", "timestamp (hh:mm:ss)", "test type", "time type"}
                )
                time_x_default = find_column_by_name(
                    numeric_cols, ["timestamp_seconds", "Time (s)", "Time", "Timestamp (hh:mm:ss)"]
                )

                # --- Robust initialization for Graph 1 & Graph 2 (no forced resets) ---
                # X-axis default for Graph 1 (prefer RPM, then Time – works with "Motor Electrical Speed (RPM)" etc.)
                if (
                    "multi_param_x_axis_selector" not in st.session_state
                    or st.session_state["multi_param_x_axis_selector"] not in numeric_cols
                ):
                    if is_rotrix_data and time_x_default:
                        x_default = time_x_default
                    else:
                        x_default = (
                            find_column_by_name(numeric_cols, "RPM")
                            or find_column_by_name(numeric_cols, "Time")
                        )
                    st.session_state["multi_param_x_axis_selector"] = x_default or (numeric_cols[0] if numeric_cols else None)

                x_axis_init = st.session_state.get("multi_param_x_axis_selector")
                y_candidates_init = [c for c in numeric_cols if c != x_axis_init]

                # Left Y defaults (prefer Thrust – matches "Thrust (gf)" etc.)
                left_existing = st.session_state.get("multi_param_left_y_axis_selector", [])
                left_valid = [c for c in left_existing if c in y_candidates_init]
                if not left_valid:
                    thrust_col = find_column_by_name(y_candidates_init, "Thrust")
                    left_valid = [thrust_col] if thrust_col else ([y_candidates_init[0]] if y_candidates_init else [])
                st.session_state["multi_param_left_y_axis_selector"] = left_valid

                # Right Y defaults (prefer SysEffect / efficiency – matches "Overall Efficiency (gf/W)" etc.)
                remaining_for_right_init = [c for c in y_candidates_init if c not in left_valid]
                right_existing = st.session_state.get("multi_param_right_y_axis_selector", [])
                right_valid = [c for c in right_existing if c in remaining_for_right_init]
                if not right_valid and remaining_for_right_init:
                    syseffect_col = find_column_by_name(
                        remaining_for_right_init,
                        ["SysEffect", "Overall Efficiency", "efficiency", "Efficiency"]
                    )
                    right_valid = [syseffect_col] if syseffect_col else [remaining_for_right_init[0]]
                st.session_state["multi_param_right_y_axis_selector"] = right_valid


                # Ensure multi_param_saved_graphs exists
                if "multi_param_saved_graphs" not in st.session_state:
                    st.session_state.multi_param_saved_graphs = []

                # Preset graphs from spec table: Graph 2–6 with robust column matching
                if len(st.session_state.multi_param_saved_graphs) == 0:
                    # Graph 2: Load Characteristic – X=|Torque| (N·m), Left=Thrust (gf), Right=SysEffect (gf/W)
                    torque_col = find_column_by_name(
                        numeric_cols,
                        ["Torque (N·m)", "Torque (N*m)", "|Torque| (N·m)", "Torque - N*m", "Torque", "torque"],
                    )
                    graph2_x = (
                        time_x_default
                        if is_rotrix_data and time_x_default
                        else (torque_col or (numeric_cols[0] if numeric_cols else None))
                    )
                    thrust_col = find_column_by_name(numeric_cols, ["Thrust (gf)", "Thrust - gf", "Thrust"])
                    syseffect_col = find_column_by_name(
                        numeric_cols,
                        ["SysEffect (gf/W)", "SysEffect - gf/W", "SysEffect", "Overall Efficiency", "efficiency", "Efficiency"],
                    )
                    graph2_left = [thrust_col] if thrust_col else []
                    graph2_right = [syseffect_col] if syseffect_col else []

                    if graph2_x:
                        st.session_state.multi_param_saved_graphs.append({
                            "x_axis": graph2_x,
                            "left_y_axes": graph2_left,
                            "right_y_axes": graph2_right,
                            "smoothing_enabled": False,
                            "smoothing_method": "savgol",
                            "smoothing_window": 5,
                            "fig": None,
                        })
                        # Graph 4 & 5 presets depend on having vibration and/or AccX/AccY/AccZ data.
                        # If the raw data has neither, we keep only 3 graphs (1–3) and skip these presets.
                        time_col = find_column_by_name(numeric_cols, ["Time (s)", "Time", "timestamp_seconds", "Time (secs)"])
                        g4_thrust = find_column_by_name(numeric_cols, ["Thrust (gf)", "Thrust - gf", "Thrust"])
                        vibration_col = find_column_by_name(
                            numeric_cols,
                            ["Vibration RMS (g)", "Vibration (g)", "Vibration - g", "Vibration"],
                        )
                        # Graph 5: Acceleration Response – X=Time (s),
                        # Left=AccX, AccY (g), Right=AccZ (g)
                        accx = find_column_by_name(numeric_cols, ["AccX (g)", "AccX", "accx"])
                        accy = find_column_by_name(numeric_cols, ["AccY (g)", "AccY", "accy"])
                        accz = find_column_by_name(numeric_cols, ["AccZ (g)", "AccZ corrected", "AccZ", "accz"])
                        g5_left = [c for c in [accx, accy] if c]
                        g5_right = [accz] if accz else []

                        has_vibration = vibration_col is not None
                        has_acc_axes = bool(g5_left)

                        # If neither vibration nor acc* columns exist, don't create Graphs 4 and 5 (only 3 graphs total)
                        if has_vibration or has_acc_axes:
                            # Graph 4: Ramp Response – X=Time (s), Left=Thrust (gf) & Overall Efficiency (gf/W), Right=Vibration (g)
                            graph4_x = time_col or (numeric_cols[0] if numeric_cols else None)
                            if has_vibration and graph4_x:
                                g4_left = [c for c in [g4_thrust, syseffect_col] if c]
                                st.session_state.multi_param_saved_graphs.append({
                                    "x_axis": graph4_x,
                                    "left_y_axes": g4_left,
                                    "right_y_axes": [vibration_col],
                                    "smoothing_enabled": False,
                                    "smoothing_method": "savgol",
                                    "smoothing_window": 5,
                                    "fig": None,
                                })

                            # Graph 5: only if we have AccX/AccY/AccZ (and time)
                            if time_col and has_acc_axes:
                                st.session_state.multi_param_saved_graphs.append({
                                    "x_axis": time_col,
                                    "left_y_axes": g5_left,
                                    "right_y_axes": g5_right,
                                    "smoothing_enabled": False,
                                    "smoothing_method": "savgol",
                                    "smoothing_window": 5,
                                    "fig": None,
                                })

            # Right column: report configuration (Single/Multi, Plot, Data)
            with col_right:
                # Original styled header for template settings
                st.markdown(
                    "<p style='font-size: 1.35rem; font-weight: 600; color: #262730; margin-bottom: 0.5rem;'>Report template</p>",
                    unsafe_allow_html=True,
                )

                # Organization branding: logo used on the PDF cover page
                _org_id = st.session_state.get("organization_id")
                _current_logo_path = get_org_logo_path(_org_id) if _org_id else None
                _has_logo = bool(_current_logo_path or st.session_state.get("org_logo_path"))
                _expander_title = "Branding logo already set (click to update)" if _has_logo else "Branding (company logo for cover page)"
                with st.expander(_expander_title, expanded=False):
                    if _has_logo:
                        st.info("A company logo is already set and will appear on report cover pages. You can upload a new one below to update it.")
                    if _current_logo_path:
                        try:
                            st.image(
                                _current_logo_path,
                                caption="Current organization logo (used on the report cover page)",
                                use_container_width=False,
                            )
                        except Exception:
                            st.caption("A logo is set for this organization, but it could not be displayed.")

                    # If a logo has ever been saved for this org (even if the
                    # file cannot currently be displayed), show an "Update"
                    # label instead of the initial placeholder.
                    _logo_label = (
                        "Update company logo (PNG/JPG, used on the report cover page)"
                        if _has_logo
                        else "Upload company logo (PNG/JPG, used on the report cover page)"
                    )
                    _logo_file = st.file_uploader(
                        _logo_label,
                        type=["png", "jpg", "jpeg"],
                        accept_multiple_files=False,
                        key="org_logo_uploader",
                    )
                    if _logo_file is not None:
                        _logo_bytes = _logo_file.getvalue()
                        if _org_id:
                            _saved_path = save_org_logo(_org_id, _logo_bytes)
                            if _saved_path:
                                st.session_state["org_logo_path"] = _saved_path
                                st.success("Organization logo updated. Future reports will use this logo on the cover page.")
                        else:
                            st.warning("No organization assigned; cannot save a shared logo.")

                add_new_view = st.session_state.get("report_multi_add_new_view", False)
                show_save_form = st.session_state.get("report_show_save_profile_form", False)
                if add_new_view and show_save_form:
                    # Save template page: name, description, Cancel (back to Add New), Confirm Save (save and go to template list)
                    _is_edit_mode = st.session_state.get("report_edit_mode", False)
                    # In edit mode, ensure name/description are pre-filled from the stored template
                    if _is_edit_mode:
                        _edit_idx = st.session_state.get("report_profile_edit_idx", None)
                        if isinstance(_edit_idx, int):
                            _profiles = _load_sorted_report_profiles()
                            if 0 <= _edit_idx < len(_profiles):
                                _p = _profiles[_edit_idx]
                                if "report_save_profile_name" not in st.session_state or st.session_state.get("report_save_profile_name") == "":
                                    st.session_state["report_save_profile_name"] = (_p.get("name") or "").strip()
                                if "report_save_profile_desc" not in st.session_state:
                                    st.session_state["report_save_profile_desc"] = (_p.get("description") or "").strip()
                        st.markdown("<p style='font-size: 1rem; color: #555; margin-bottom: 1rem;'>Update template settings</p>", unsafe_allow_html=True)
                        st.caption("Name and description are from the stored template. Edit if needed, then confirm to save.")
                    else:
                        st.markdown("<p style='font-size: 1rem; color: #555; margin-bottom: 1rem;'>Save current settings as a template</p>", unsafe_allow_html=True)
                        st.caption("Enter a name and optional description, then confirm to save.")
                    _save_name = st.text_input("Profile name", key="report_save_profile_name", placeholder="e.g. My Report Template")
                    _save_desc = st.text_input("Description (optional)", key="report_save_profile_desc", placeholder="Optional description")
                    _confirm_col, _form_cancel_col = st.columns([1, 1])
                    with _confirm_col:
                        if st.button("Confirm Save", key="report_confirm_save_btn"):
                            if (_save_name or "").strip():
                                _final_name = (_save_name or "").strip()
                                _final_desc = (_save_desc or "").strip()
                                _new_profile = profile_from_session_state(_final_name, _final_desc)
                                profiles = _load_sorted_report_profiles()
                                if st.session_state.get("report_edit_mode", False):
                                    _edit_idx = st.session_state.get("report_profile_edit_idx", None)
                                    if isinstance(_edit_idx, int) and 0 <= _edit_idx < len(profiles):
                                        _existing = profiles[_edit_idx]
                                        for _field in ("created_at", "created_by"):
                                            if _existing.get(_field) is not None:
                                                _new_profile[_field] = _existing.get(_field)
                                        profiles[_edit_idx] = _new_profile
                                        save_profiles(profiles)
                                        st.session_state["report_edit_mode"] = False
                                        st.session_state["report_profile_edit_idx"] = None
                                        st.session_state["report_show_save_profile_form"] = False
                                        st.session_state.report_multi_add_new_view = False
                                        st.success("Template updated successfully!")
                                        st.rerun()
                                else:
                                    profiles.append(_new_profile)
                                    save_profiles(profiles)
                                    st.session_state["report_show_save_profile_form"] = False
                                    st.session_state.report_multi_add_new_view = False
                                    st.success("Profile saved successfully!")
                                    st.rerun()
                            else:
                                st.warning("Please enter a profile name.")
                    with _form_cancel_col:
                        if st.button("Cancel", key="report_save_form_cancel_btn"):
                            if st.session_state.get("report_edit_mode", False):
                                st.session_state["report_show_save_profile_form"] = False
                                st.session_state["report_edit_mode"] = False
                                st.session_state["report_profile_edit_idx"] = None
                                st.session_state.report_multi_add_new_view = False
                            else:
                                st.session_state["report_show_save_profile_form"] = False
                            st.rerun()
                elif add_new_view:
                    # Summary, Plot, and Data tabs (Cancel button is at bottom with Update report preview)
                    # Require a file to be selected for configuring the report preview
                    bar_df_raw = None
                    try:
                        if selected_file == "None":
                            st.warning("Please select a file for the preview.")
                        else:
                            _raw = st.session_state.get("multi_param_raw_df", None)
                            bar_df_raw = _build_report_raw_data_df(_raw) if _raw is not None else None
                        has_tr_cols = (
                            bar_df_raw is not None
                            and not bar_df_raw.empty
                            and "Throttle Range (10%)" in bar_df_raw.columns
                            and "Time range" in bar_df_raw.columns
                        )
                        # Scrollable tab content so the page itself doesn't scroll
                        st.markdown(
                            """
                            <style>
                            /* Summary, Plot, Data tab content: fixed height + scroll */
                            div[data-testid="stTabs"] [role="tabpanel"] {
                                max-height: 70vh !important;
                                overflow-y: auto !important;
                            }
                            div[data-testid="stTabs"] > div > div[style*="overflow"] {
                                max-height: 70vh !important;
                                overflow-y: auto !important;
                            }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )
                        with st.container(height=700):
                            tab_file_info, tab_summary, tab_data_report, tab_plot_report = st.tabs(
                                ["File Info", "Summary", "Data", "Plot"]
                            )

                            # ---- File Info (optional, comes first) ----
                            with tab_file_info:
                                st.markdown("#### File Info (optional)")
                                st.caption(
                                    "If your data file does not contain a header/file-info block, you can enter it here. "
                                    "When the data already includes file info, that automatic header is used and these fields are ignored."
                                )
                                auto_info = (st.session_state.get("report_file_info_text") or "").strip()
                                disabled = bool(auto_info)
                                if auto_info:
                                    st.info(
                                        "This file already has embedded file information detected from the data header. "
                                        "The manual file info below will be **ignored** for this file."
                                    )

                                col_l, col_r = st.columns(2)
                                with col_l:
                                    fi_test_time = st.text_input(
                                        "Test Time",
                                        key="fi_test_time",
                                        disabled=disabled,
                                    )
                                    fi_motor_model = st.text_input(
                                        "Motor Model",
                                        key="fi_motor_model",
                                        disabled=disabled,
                                    )
                                    fi_prop_model = st.text_input(
                                        "Propeller Model",
                                        key="fi_prop_model",
                                        disabled=disabled,
                                    )
                                    fi_power_model = st.text_input(
                                        "Power Model",
                                        key="fi_power_model",
                                        disabled=disabled,
                                    )
                                with col_r:
                                    fi_tester = st.text_input(
                                        "Tester",
                                        key="fi_tester",
                                        disabled=disabled,
                                    )
                                    fi_pole_pairs = st.text_input(
                                        "Pole Pairs",
                                        key="fi_pole_pairs",
                                        disabled=disabled,
                                    )
                                    fi_esc_model = st.text_input(
                                        "Electric Control Model",
                                        key="fi_esc_model",
                                        disabled=disabled,
                                    )
                                    fi_blade_num = st.text_input(
                                        "Blade Num",
                                        key="fi_blade_num",
                                        disabled=disabled,
                                    )

                                fi_remarks = st.text_area(
                                    "Remarks",
                                    key="fi_remarks",
                                    height=80,
                                    disabled=disabled,
                                )

                                # Build manual_file_info_text only when we're allowed to override (no auto info)
                                if not disabled:
                                    parts: list[str] = []

                                    def _add(label: str, value: str) -> None:
                                        v = (value or "").strip()
                                        if v:
                                            parts.append(f"{label} : {v}")

                                    _add("Test Time", fi_test_time)
                                    _add("Tester", fi_tester)
                                    _add("Motor Model", fi_motor_model)
                                    _add("Pole Pairs", fi_pole_pairs)
                                    _add("Propeller Model", fi_prop_model)
                                    _add("Electric Control Model", fi_esc_model)
                                    _add("Power Model", fi_power_model)
                                    _add("Blade Num", fi_blade_num)
                                    _add("Remarks", fi_remarks)

                                    st.session_state["manual_file_info_text"] = "\n".join(parts)

                            # ---- Summary tab ----
                            with tab_summary:
                                # Row 1: Key statistics
                                selected_file_name = st.session_state.get("multi_param_selected_file")
                                insights_cache = st.session_state.get("multi_param_file_insights", {})
                                metrics = insights_cache.get(selected_file_name or "", {})
                                if selected_file_name and metrics:
                                    runtime_s = metrics.get("runtime_s")
                                    max_thrust = metrics.get("max_thrust")
                                    max_rpm = metrics.get("max_rpm")
                                    max_power = metrics.get("max_power")
                                    runtime_str = seconds_to_mmss(runtime_s) if isinstance(runtime_s, (int, float)) else "00:00"
                                    thrust_str = f"{max_thrust:.0f}" if isinstance(max_thrust, (int, float)) else "N/A"
                                    rpm_str = f"{max_rpm:.0f}" if isinstance(max_rpm, (int, float)) else "N/A"
                                    power_str = f"{max_power:.0f}" if isinstance(max_power, (int, float)) else "N/A"
                                    st.markdown(
                                        f"""
                                        <div style="margin-bottom: 16px;">
                                            <div style="display: flex; flex-wrap: wrap; gap: 16px; font-size: 1.5rem; color: #000000;">
                                                <span><strong>Total runtime (mm:ss):</strong> {runtime_str}</span>
                                                <span><strong>Max thrust (gf):</strong> {thrust_str}</span>
                                                <span><strong>Max speed (RPM):</strong> {rpm_str}</span>
                                                <span><strong>Max power (W):</strong> {power_str}</span>
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                # Rows 2–4: dwell-time bars, thrust line, time-share bars
                                if has_tr_cols:
                                    fig_dwell_sec = build_throttle_dwell_bar_figure(bar_df_raw)
                                    # Row 2: Total elapsed time per throttle band (sum of Δt)
                                    st.markdown("<h5 style='text-align: center;'>Time Spent in Each Throttle Operating Range</h5>", unsafe_allow_html=True)
                                    if fig_dwell_sec is not None:
                                        _register_plot_shown()
                                        st.plotly_chart(fig_dwell_sec, use_container_width=True)
                                        st.session_state["report_throttle_bar_fig"] = fig_dwell_sec
                                    else:
                                        st.info("Could not compute time in each band (need a monotonic time column and throttle bins).")
                                    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
                                    # Row 3: Line chart (match multi_param_app: use Time range as-is)
                                    st.markdown("<h5 style='text-align: center;'>Thrust Evolution Over Time Across Throttle Bands</h5>", unsafe_allow_html=True)
                                    line_df = bar_df_raw.copy()
                                    line_df["Time range"] = pd.to_numeric(line_df["Time range"], errors="coerce")
                                    line_df["Throttle Range (10%)"] = line_df["Throttle Range (10%)"].astype(str)
                                    line_df = line_df[line_df["Throttle Range (10%)"] != "0-10"]
                                    thrust_col_line = None
                                    for c in line_df.columns:
                                        if "thrust" in str(c).lower() and ("gf" in str(c).lower() or "g)" in str(c)):
                                            thrust_col_line = c
                                            break
                                    if thrust_col_line is None:
                                        for c in line_df.columns:
                                            if "thrust" in str(c).lower():
                                                thrust_col_line = c
                                                break
                                    fig_line = go.Figure()
                                    if thrust_col_line is not None and thrust_col_line in line_df.columns:
                                        line_df[thrust_col_line] = pd.to_numeric(line_df[thrust_col_line], errors="coerce")
                                        line_df = line_df.dropna(subset=["Time range", thrust_col_line])
                                        colors_line = [
                                            "rgba(231, 76, 60, 0.9)", "rgba(52, 152, 219, 0.9)", "rgba(46, 204, 113, 0.9)",
                                            "rgba(241, 196, 15, 0.9)", "rgba(155, 89, 182, 0.9)", "rgba(26, 188, 156, 0.9)",
                                            "rgba(230, 126, 34, 0.9)", "rgba(149, 165, 166, 0.9)", "rgba(192, 57, 43, 0.9)",
                                            "rgba(41, 128, 185, 0.9)",
                                        ]
                                        order_labels_line = [f"{i}-{i+10}" for i in range(10, 100, 10)]
                                        for color_idx, (tr_label, grp) in enumerate(line_df.groupby("Throttle Range (10%)", observed=True, sort=False)):
                                            grp = grp.sort_values("Time range")
                                            if grp.empty:
                                                continue
                                            valid = np.isfinite(grp["Time range"].values) & np.isfinite(grp[thrust_col_line].values)
                                            if not np.any(valid):
                                                continue
                                            fig_line.add_trace(
                                                go.Scatter(
                                                    x=grp["Time range"].values[valid],
                                                    y=grp[thrust_col_line].values[valid],
                                                    mode="lines",
                                                    name=tr_label,
                                                    line=dict(color=colors_line[color_idx % len(colors_line)], width=2),
                                                )
                                            )
                                    fig_line.update_layout(
                                        xaxis=dict(title="Time range (s)", title_font=dict(size=22, color="black"), tickfont=dict(size=18, color="black"), nticks=10, showgrid=True, gridcolor='#e0e0e0', gridwidth=1, showline=True, linecolor='#333333', linewidth=1.5),
                                        yaxis=dict(title=thrust_col_line if thrust_col_line else "Thrust", title_font=dict(size=22, color="black"), tickfont=dict(size=18, color="black"), nticks=10, showgrid=True, gridcolor='#e0e0e0', gridwidth=1, showline=True, linecolor='#333333', linewidth=1.5),
                                        margin=dict(l=60, r=60, t=40, b=60),
                                        template="plotly_white",
                                        hovermode="x unified",
                                        showlegend=True,
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=16, color="black"), title=dict(text="Throttle Range"), title_font=dict(size=16, color="black"), bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1),
                                    )
                                    if len(fig_line.data) > 0:
                                        _register_plot_shown()
                                        st.plotly_chart(fig_line, use_container_width=True)
                                        st.session_state["report_throttle_line_fig"] = fig_line
                                    else:
                                        st.info("No valid Time range / Throttle Range segments for the line chart.")
                                else:
                                    st.info("Use raw data and open the Data tab first so Throttle Range (10%) and Time range columns exist.")

                            # (File Info tab is defined above, before Summary)
                            with tab_data_report:
                                # Throttle-Based Aggregation (Sorted Performance Table) — shown first; raw data at bottom
                                st.markdown("#### ⚙️ Throttle Aggregation Settings")

                                # Use the raw report data already loaded for this file, if available.
                                df = st.session_state.get("report_raw_data_df")
                                if df is None:
                                    df = st.session_state.get("multi_param_raw_df")

                                if df is None or getattr(df, "empty", True):
                                    st.error("No data available in selected file.")
                                else:
                                    # Auto-detect column names (inline patterns: legacy + DWC1/renamed + wingflyingtech bench)
                                    throttle_col = find_column_by_pattern(df, [
                                        "Throttle - %", "Throttle Input (%)", "Throttle (%)", "Throttle", "throttle",
                                    ])

                                    # If no explicit throttle column, derive from Servo/ESC PWM (e.g. DWC1 has "ESC signal (µs)", "Servo 3 (µs)")
                                    if throttle_col is None:
                                        pwm_col = None
                                        for col in df.columns:
                                            col_lower = str(col).lower()
                                            if "servo" in col_lower or "esc" in col_lower:
                                                series = pd.to_numeric(df[col], errors="coerce")
                                                if series.notna().sum() == 0:
                                                    continue
                                                valid_mask = series.between(800, 2200)
                                                if valid_mask.mean() >= 0.5:
                                                    pwm_col = col
                                                    break
                                        if pwm_col is not None:
                                            pwm_series = pd.to_numeric(df[pwm_col], errors="coerce")
                                            pwm_min, pwm_max = 1000.0, 2000.0
                                            throttle_series = (pwm_series - pwm_min) / (pwm_max - pwm_min) * 100.0
                                            throttle_series = throttle_series.clip(lower=0.0, upper=100.0)
                                            new_throttle_col = "Throttle - %"
                                            if new_throttle_col in df.columns:
                                                new_throttle_col = new_throttle_col + "_derived"
                                            df[new_throttle_col] = throttle_series
                                            throttle_col = new_throttle_col

                                    current_col = find_column_by_pattern(df, ["Cur - A", "Current (A)", "Current [A]", "Current", "current", "Cur"])
                                    voltage_col = find_column_by_pattern(df, ["Vol - V", "Voltage (V)", "Voltage [V]", "Voltage", "voltage", "Vol"])
                                    rpm1_col = find_column_by_pattern(df, [
                                        "RPM", "RPM1 - RPM", "RPM1", "rpm1",
                                        "Motor Electrical Speed (RPM)", "Motor Electrical Speed",
                                        "Electrical Speed (RPM)", "Electrical Speed",
                                        "Rotational Speed (RPM)", "Rotational Speed",
                                    ])
                                    rpm2_col = find_column_by_pattern(df, [
                                        "RPM2 - RPM", "RPM2", "rpm2",
                                        "Motor Optical Speed (RPM)", "Motor Optical Speed",
                                        "Optical Speed (RPM)", "Optical Speed",
                                    ])
                                    thrust_col = find_column_by_pattern(df, [
                                        "Thrust - gf", "Thrust (gf)", "Thrust (kgf)", "Thrust [g]", "Thrust", "thrust",
                                    ])
                                    torque_col = find_column_by_pattern(df, [
                                        "Torque - N*m", "Torque (N*m)", "Torque (N·m)",
                                        "Torque (Nm)", "Torque (N.m)", "Torque [N·m]", "Torque [N*m]",
                                        "Torque", "torque",
                                    ])
                                    motorpower_col = find_column_by_pattern(df, [
                                        "MotorPower - W", "MotorPower", "motorpower",
                                        "Mechanical Power (W)", "Mechanical (W)",
                                        "Electrical Power (W)", "Electrical (W)",
                                        "InPower - W", "InPower", "Power",
                                    ])
                    
                                    # Check required columns
                                    required_cols = {
                                        "Throttle": throttle_col,
                                        "Current": current_col,
                                        "Voltage": voltage_col,
                                        "Thrust": thrust_col,
                                        "Torque": torque_col
                                    }
                    
                                    missing_cols = [name for name, col in required_cols.items() if col is None]
                    
                                    if missing_cols:
                                        st.error(f"❌ Required columns not found: {', '.join(missing_cols)}")
                                        st.info("Please ensure your file contains columns like: Throttle, Current, Voltage, Thrust, Torque")
                                    else:
                                        # Get throttle range from data
                                        if throttle_col in df.columns:
                                            throttle_values = df[throttle_col].dropna().tolist()
                                            if throttle_values:
                                                data_min = float(min(throttle_values))
                                                data_max = float(max(throttle_values))
                                            else:
                                                data_min, data_max = 0.0, 100.0
                                        else:
                                            data_min, data_max = 0.0, 100.0
                        
                                        # User inputs
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            start_throttle = st.number_input(
                                                "Start Throttle (%)",
                                                min_value=0.0,
                                                max_value=100.0,
                                                value=float(data_min),
                                                step=1.0,
                                                key="single_file_throttle_min_input",
                                                help="Starting throttle percentage for the range"
                                            )
                                        with col2:
                                            end_throttle = st.number_input(
                                                "End Throttle (%)",
                                                min_value=0.0,
                                                max_value=100.0,
                                                value=float(data_max),
                                                step=1.0,
                                                key="single_file_throttle_max_input",
                                                help="Ending throttle percentage for the range"
                                            )
                                        # Normalise start/end so processing always has min/max in correct order
                                        throttle_min = min(start_throttle, end_throttle)
                                        throttle_max = max(start_throttle, end_throttle)

                                        with col3:
                                            throttle_interval = st.number_input(
                                                "Throttle Interval (%)",
                                                min_value=0.1,
                                                max_value=10.0,
                                                value=5.0,
                                                step=0.5,
                                                key="single_file_throttle_interval_input",
                                                help="Bin size for throttle grouping (e.g., 5% = 0-5, 5-10, etc.)"
                                            )
                                        with col4:
                                            ramp_mode = st.selectbox(
                                                "Ramp Mode",
                                                ["ramp_up", "ramp_down", "bi_directional"],
                                                index=0,
                                                key="single_file_ramp_mode_select",
                                                help="ramp_up: only increasing throttle, ramp_down: only decreasing, bi_directional: all data"
                                            )

                                        st.session_state["_throttle_cfg_start"] = float(start_throttle)
                                        st.session_state["_throttle_cfg_end"] = float(end_throttle)
                                        st.session_state["_throttle_cfg_interval"] = float(throttle_interval)
                                        st.session_state["_throttle_cfg_ramp_mode"] = str(ramp_mode)

                                        # Automatically generate and display sorted table
                                        with st.spinner("Processing file..."):
                                            # Process the file
                                            result_df = process_throttle_aggregation(
                                                df, throttle_col, current_col, voltage_col,
                                                rpm1_col, rpm2_col, thrust_col, torque_col,
                                                motorpower_col, mode=ramp_mode,
                                                throttle_min=throttle_min,
                                                throttle_max=throttle_max,
                                                throttle_interval=throttle_interval
                                            )
                            
                                            # Display results
                                            if result_df is not None and not result_df.empty:
                                                # Store for reuse in Plot tab
                                                st.session_state["multi_param_sorted_table"] = result_df
                                                prev_sorted = getattr(st.session_state, "report_sorted_table_df", None)
                                                sorted_changed = prev_sorted is None or prev_sorted.shape != result_df.shape or list(prev_sorted.columns) != list(result_df.columns)
                                                st.session_state.report_sorted_table_df = result_df
                                                if sorted_changed:
                                                    invalidate_report_after_data_change()
                                                    _refresh_report_raw_data_df()
                                                    # Keep summary graphs in sync immediately after
                                                    # sorted-table regeneration in Template Settings.
                                                    try:
                                                        ensure_summary_graphs_for_current_file()
                                                    except Exception:
                                                        pass
                                                # Cache throttle regime labels using RAW data time distribution
                                                try:
                                                    regimes = detect_throttle_regimes_from_raw(
                                                        df,
                                                        throttle_col,
                                                        "timestamp_seconds",
                                                        result_df,
                                                        throttle_interval,
                                                    )
                                                    if regimes is not None:
                                                        st.session_state["throttle_regime_cache"] = regimes
                                                except Exception:
                                                    pass
                                
                                                st.markdown("#### 📊 Sorted Performance Table")
                                
                                                # Calculate dynamic height based on number of rows (up to 10 rows)
                                                num_rows = len(result_df)
                                                # Approximate height: header (~35px) + row height (~35px per row)
                                                # For up to 10 rows, calculate exact height; after 10, use fixed height for scrolling
                                                if num_rows <= 10:
                                                    table_height = 35 + (num_rows * 35)  # Header + rows
                                                else:
                                                    table_height = 35 + (10 * 35)  # Header + 10 rows (then scroll)
                                
                                                st.dataframe(result_df, use_container_width=True, height=table_height)
                                            else:
                                                st.warning("⚠️ No data available after filtering. Please check your settings.")
                                                st.session_state["multi_param_sorted_table"] = None
                                                st.session_state.report_sorted_table_df = None
                                                invalidate_report_after_data_change()
                                                _refresh_report_raw_data_df()
                                                try:
                                                    ensure_summary_graphs_for_current_file()
                                                except Exception:
                                                    pass

                                # Raw data at bottom of Data tab
                                st.markdown("### 📋 Raw Data")
                                if df is None or df.empty:
                                    st.info("No data available. Select a file to see raw data.")
                                else:
                                    df_display = df.copy()
                                    df_display = fix_duplicate_columns(df_display)
                                    if 'Index' not in df_display.columns:
                                        df_display.insert(0, 'Index', range(1, len(df_display) + 1))
                                    throttle_col_display = find_column_by_pattern(df_display, ["Throttle - %", "Throttle Input (%)", "Throttle (%)", "Throttle", "throttle"])
                                    if throttle_col_display is None and df is not None and not df.empty:
                                        for col in df.columns:
                                            col_lower = str(col).lower()
                                            if "servo" in col_lower or "esc" in col_lower:
                                                series = pd.to_numeric(df[col], errors="coerce")
                                                if series.notna().sum() == 0:
                                                    continue
                                                valid_mask = series.between(800, 2200)
                                                if valid_mask.mean() >= 0.5:
                                                    pwm_series = pd.to_numeric(df[col], errors="coerce")
                                                    throttle_series = (pwm_series - 1000.0) / (2000.0 - 1000.0) * 100.0
                                                    throttle_series = throttle_series.clip(lower=0.0, upper=100.0)
                                                    new_name = "Throttle - %" if "Throttle - %" not in df.columns else "Throttle - %_derived"
                                                    df[new_name] = throttle_series
                                                    df_display[new_name] = throttle_series.values
                                                    throttle_col_display = new_name
                                                    break
                                    if throttle_col_display is not None and throttle_col_display in df_display.columns:
                                        thresh = np.arange(0, 101, 10)
                                        labels = [f"{i}-{i+10}" for i in range(0, 100, 10)]
                                        t_vals = pd.to_numeric(df_display[throttle_col_display], errors="coerce").clip(0, 100)
                                        df_display["Throttle Range (10%)"] = pd.cut(t_vals, bins=thresh, labels=labels, include_lowest=True)
                                        if df is not None and throttle_col_display in df.columns:
                                            t_df = pd.to_numeric(df[throttle_col_display], errors="coerce").clip(0, 100)
                                            df["Throttle Range (10%)"] = pd.cut(t_df, bins=thresh, labels=labels, include_lowest=True)
                                    time_col_display = find_column_by_pattern(df_display, ["Time (s)", "Time (secs)", "timestamp_seconds", "Time"])
                                    if time_col_display is not None and time_col_display in df_display.columns:
                                        t_series = pd.to_numeric(df_display[time_col_display], errors="coerce")
                                        if t_series.isna().all() or t_series.isna().sum() > len(t_series) // 2:
                                            raw_time = df_display[time_col_display].astype(str)
                                            t_series = raw_time.apply(lambda x: mmss_to_seconds(x) if x and x.strip().lower() not in ("", "nan", "none") else np.nan)
                                        if "Throttle Range (10%)" in df_display.columns:
                                            seg = (df_display["Throttle Range (10%)"] != df_display["Throttle Range (10%)"].shift()).cumsum()
                                            seg_first = t_series.groupby(seg).first()
                                            ref_per_row = seg.map(seg_first)
                                            df_display["Time range"] = (t_series - ref_per_row).round(4)
                                            if df is not None and "Throttle Range (10%)" in df.columns and time_col_display in df.columns:
                                                t_df = pd.to_numeric(df[time_col_display], errors="coerce")
                                                if t_df.isna().all() or t_df.isna().sum() > len(t_df) // 2:
                                                    raw_time_df = df[time_col_display].astype(str)
                                                    t_df = raw_time_df.apply(lambda x: mmss_to_seconds(x) if x and x.strip().lower() not in ("", "nan", "none") else np.nan)
                                                seg_df = (df["Throttle Range (10%)"] != df["Throttle Range (10%)"].shift()).cumsum()
                                                seg_first_df = t_df.groupby(seg_df).first()
                                                ref_per_row_df = seg_df.map(seg_first_df)
                                                df["Time range"] = (t_df - ref_per_row_df).round(4)
                                        else:
                                            first_ts = t_series.iloc[0] if len(t_series) else 0
                                            df_display["Time range"] = (t_series - first_ts).round(4)
                                            if df is not None and time_col_display in df.columns:
                                                t_df = pd.to_numeric(df[time_col_display], errors="coerce")
                                                if t_df.isna().all() or t_df.isna().sum() > len(t_df) // 2:
                                                    raw_time_df = df[time_col_display].astype(str)
                                                    t_df = raw_time_df.apply(lambda x: mmss_to_seconds(x) if x and x.strip().lower() not in ("", "nan", "none") else np.nan)
                                                first_df = t_df.iloc[0] if len(t_df) else 0
                                                df["Time range"] = (t_df - first_df).round(4)
                                    if "timestamp_seconds" in df_display.columns:
                                        df_display = df_display.drop(columns=["timestamp_seconds"])
                                    all_cols = list(dict.fromkeys(list(df_display.columns)))
                                    enable_column_selection = st.checkbox(
                                        "Select columns to display",
                                        value=False,
                                        key="multi_param_enable_column_selection",
                                        help="Check this to select specific columns. By default, all columns are displayed."
                                    )
                                    if enable_column_selection:
                                        default_selected = all_cols[:20] if len(all_cols) > 20 else all_cols
                                        selected_cols = st.multiselect(
                                            "Columns",
                                            all_cols,
                                            default=default_selected,
                                            key="multi_param_data_column_selector",
                                            help="Select columns to display in the data table"
                                        )
                                        if not selected_cols:
                                            selected_cols = all_cols.copy()
                                        selected_cols = list(dict.fromkeys(selected_cols))
                                        if 'Index' in selected_cols:
                                            selected_cols.remove('Index')
                                        selected_cols = ['Index'] + selected_cols
                                        selected_cols = [col for col in selected_cols if col in df_display.columns]
                                    else:
                                        selected_cols = all_cols.copy()
                                        if 'Index' in selected_cols:
                                            selected_cols.remove('Index')
                                        selected_cols = ['Index'] + selected_cols
                                    st.dataframe(df_display[selected_cols].rename(columns=COLUMN_DISPLAY_NAMES), use_container_width=True, height=400)
                                    # Keep report_raw_data_df independent from table column selection.
                                    # Summary graphs require full internal columns such as Throttle Range,
                                    # Time range, and thrust/vibration metrics even when hidden in UI table.
                                    report_df = df_display.copy()
                                    if COLUMN_DISPLAY_NAMES:
                                        report_df = report_df.rename(columns=COLUMN_DISPLAY_NAMES)
                                    prev_raw = getattr(st.session_state, "report_raw_data_df", None)
                                    raw_changed = prev_raw is None or prev_raw.shape != report_df.shape or list(prev_raw.columns) != list(report_df.columns)
                                    st.session_state.report_raw_data_df = report_df
                                    st.session_state["report_loaded_file_name"] = st.session_state.get("multi_param_selected_file")
                                    if raw_changed:
                                        invalidate_report_after_data_change()
                                        # Regenerate summary graphs right away to avoid
                                        # empty summary section after table/plot updates.
                                        try:
                                            ensure_summary_graphs_for_current_file()
                                        except Exception:
                                            pass

                            with tab_plot_report:
                                # Plot tab: data source selector and saved graphs (summary stats + three charts are in Summary tab)
                                # Decide which data the plots should use: raw vs sorted performance table
                                has_sorted = (
                                    st.session_state.get("multi_param_sorted_table") is not None
                                    and not getattr(st.session_state["multi_param_sorted_table"], "empty", True)
                                )
                                if has_sorted:
                                    plot_source_options = ["Raw data", "Sorted performance table"]
                                    default_plot_source = "Sorted performance table"
                                else:
                                    plot_source_options = ["Raw data"]
                                    default_plot_source = "Raw data"

                                current_source = st.session_state.get("multi_param_plot_data_source", default_plot_source)
                                if current_source not in plot_source_options:
                                    current_source = default_plot_source
                                # Selector for plot data source (shown before Graph 1)
                                st.radio(
                                    "Plot data source",
                                    plot_source_options,
                                    index=plot_source_options.index(current_source),
                                    key="multi_param_plot_data_source",
                                    help="Choose whether Graphs want to use the original raw data or the aggregated sorted performance table.",
                                    horizontal=True,
                                )

                                # Apply selected source (report_graphs_data_source is set at end of Plot tab after graphs render)
                                plot_source = st.session_state.get("multi_param_plot_data_source", default_plot_source)

                                if plot_source == "Sorted performance table" and has_sorted:
                                    df = st.session_state["multi_param_sorted_table"].copy()
                                    # Recompute numeric columns for plotting based on sorted table
                                    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                                    # When using Sorted data, set Graph 4 to Throttle % / Thrust & SysEffect / Vibration if user hasn't (e.g. SysEffect missing or X from Raw)
                                    if len(st.session_state.multi_param_saved_graphs) > 1:
                                        g4 = st.session_state.multi_param_saved_graphs[1]
                                        left_y = g4.get("left_y_axes") or []
                                        has_syseffect = any(
                                            "SysEffect" in str(c) or "sys effect" in str(c).lower() or "efficiency" in str(c).lower()
                                            for c in left_y
                                        )
                                        x_valid = (g4.get("x_axis") or "") in numeric_cols
                                        if not has_syseffect or not x_valid:
                                            throttle_col_sorted = find_column_by_name(
                                                numeric_cols,
                                                ["Throttle - %", "Throttle", "throttle"],
                                            )
                                            thrust_col_sorted = find_column_by_name(
                                                numeric_cols,
                                                ["Thrust (gf)", "Thrust - gf", "Thrust"],
                                            )
                                            syseffect_col_sorted = find_column_by_name(
                                                numeric_cols,
                                                ["SysEffect - gf/W", "SysEffect (gf/W)", "SysEffect", "Overall Efficiency", "efficiency", "Efficiency"],
                                            )
                                            vibration_col_sorted = find_column_by_name(
                                                numeric_cols,
                                                ["Vibration (g)", "Vibration - g", "Vibration RMS (g)", "Vibration"],
                                            )
                                            if throttle_col_sorted and thrust_col_sorted and syseffect_col_sorted and vibration_col_sorted:
                                                st.session_state.multi_param_saved_graphs[1] = {
                                                    **g4,
                                                    "x_axis": throttle_col_sorted,
                                                    "left_y_axes": [thrust_col_sorted, syseffect_col_sorted],
                                                    "right_y_axes": [vibration_col_sorted],
                                                }
                                                # Clear Graph 4 widget state so UI reflects the new preset
                                                for key in list(st.session_state.keys()):
                                                    if key.startswith("multi_param_saved_4_") or key == "multi_param_graph_4":
                                                        del st.session_state[key]
                                else:
                                    # For "Raw data", use the same raw df stored at file load (single source of truth)
                                    raw_df = st.session_state.get("multi_param_raw_df")
                                    if raw_df is not None and not getattr(raw_df, "empty", True):
                                        df = raw_df.copy()
                                    elif 'df' not in locals() or df is None or df.empty:
                                        st.error("No data available. Please select a file first.")
                                        render_footer()
                                        return
                                    # Ensure numeric_cols is defined for raw data (keep timestamp_seconds —
                                    # RotriX defaults X-axis to elapsed time and selectbox options must
                                    # include it or Streamlit raises when session state holds that value).
                                    if 'numeric_cols' not in locals() or not numeric_cols:
                                        numeric_cols = [
                                            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
                                        ]

                                # RotriX "time on X" must use a column that exists in the *current* plot frame.
                                # Sorted performance table usually has no timestamp_seconds — file-load
                                # time_x_default would point at a missing column and break selectboxes.
                                time_x_plot = None
                                if is_rotrix_data:
                                    time_x_plot = find_column_by_name(
                                        numeric_cols,
                                        [
                                            "timestamp_seconds",
                                            "Time (s)",
                                            "Time",
                                            "Time range",
                                            "Timestamp (hh:mm:ss)",
                                        ],
                                    )
                                if numeric_cols:
                                    for _fix_i in (1, 2, 3, 4, 5):
                                        _wk = f"multi_param_saved_{_fix_i}_x_axis"
                                        if _wk in st.session_state and st.session_state[_wk] not in numeric_cols:
                                            st.session_state[_wk] = numeric_cols[0]

                                # NOTE: Graph 1 now uses the same saved-graph logic as Graph 2+ via
                                # the render_saved_graph helper defined below. There is no dedicated
                                # Graph 1 plotting/table block here any more.

                                # Graphs are rendered below using the shared helper.
                
                                # Helper function to render a graph with its parameters
                                def render_saved_graph(idx, saved_data=None, is_empty=False):
                                    graph_key = f"multi_param_graph_{idx}"
                    
                                    # Initialize graph parameters
                                    if graph_key not in st.session_state:
                                        if is_empty and idx == 3:
                                            # Graph 3: Energy Cost of Thrust – X=Thrust (gf), Left=Electrical Power (W), Right=Motor Efficiency (%)
                                            thrust_col = find_column_by_name(numeric_cols, ["Thrust (gf)", "Thrust - gf", "Thrust"])
                                            elec_col = find_column_by_name(
                                                numeric_cols,
                                                [
                                                    "Electrical Power (W)",
                                                    "Electrical Power - W",
                                                    "Electrical Power",
                                                    "InPower - W",
                                                    "InPower",
                                                    "PowerInLine - W",
                                                    "PowerInLine",
                                                    "Power",
                                                ],
                                            )
                                            eff_col = find_column_by_name(
                                                numeric_cols,
                                                [
                                                    "Motor Efficiency (%)",
                                                    "Motor Efficiency",
                                                    "MotorRate - %",
                                                    "Overall Efficiency",
                                                    "SysEffect (gf/W)",
                                                    "SysEffect - gf/W",
                                                    "SysEffect",
                                                    "efficiency",
                                                    "Efficiency",
                                                    "PropellerEffect - gf/W",
                                                ],
                                            )
                                            x_default = thrust_col or (numeric_cols[0] if numeric_cols else None)
                                            st.session_state[graph_key] = {
                                                "x_axis": x_default,
                                                "left_y_axes": [elec_col] if elec_col else [],
                                                "right_y_axes": [eff_col] if eff_col else [],
                                                "smoothing_enabled": False,
                                                "smoothing_method": "savgol",
                                                "smoothing_window": 5,
                                            }
                                        elif is_empty:
                                            # Empty graph (Graph 3+ when no special defaults)
                                            st.session_state[graph_key] = {
                                                "x_axis": numeric_cols[0] if numeric_cols else None,
                                                "left_y_axes": [],
                                                "right_y_axes": [],
                                                "smoothing_enabled": False,
                                                "smoothing_method": "savgol",
                                                "smoothing_window": 5,
                                            }
                                        elif saved_data and isinstance(saved_data, dict) and "x_axis" in saved_data:
                                            # Initialize from saved data
                                            st.session_state[graph_key] = {
                                                "x_axis": saved_data.get("x_axis"),
                                                "left_y_axes": saved_data.get("left_y_axes") or [],
                                                "right_y_axes": saved_data.get("right_y_axes") or [],
                                                "smoothing_enabled": saved_data.get("smoothing_enabled", False),
                                                "smoothing_method": saved_data.get("smoothing_method", "savgol"),
                                                "smoothing_window": saved_data.get("smoothing_window", 5),
                                            }
                    
                                    graph_params = st.session_state[graph_key]

                                    # Enforce default X-axis for Graph 1–3 even when prior
                                    # template/session widget state exists.
                                    _forced_x = None
                                    if is_rotrix_data and time_x_plot and idx in (1, 2, 3):
                                        _forced_x = time_x_plot
                                    elif idx == 1:
                                        _forced_x = find_column_by_name(
                                            numeric_cols,
                                            ["RPM", "RPM1 - RPM", "Motor Electrical Speed (RPM)", "Motor Optical Speed (RPM)"],
                                        )
                                    elif idx == 2:
                                        _forced_x = find_column_by_name(
                                            numeric_cols,
                                            ["Torque (N·m)", "Torque (N*m)", "Torque (Nm)", "Torque - N*m", "Torque", "torque"],
                                        )
                                    elif idx == 3:
                                        _forced_x = find_column_by_name(
                                            numeric_cols,
                                            ["Thrust (gf)", "Thrust - gf", "Thrust (kgf)", "Thrust"],
                                        )
                                    if _forced_x and _forced_x in numeric_cols:
                                        graph_params["x_axis"] = _forced_x
                                        st.session_state[graph_key]["x_axis"] = _forced_x
                                        st.session_state[f"multi_param_saved_{idx}_x_axis"] = _forced_x
                    
                                    # Human-friendly titles for the first five graphs in the Plot tab
                                    graph_titles = {
                                        1: "Speed-Based Performance Trends",
                                        2: "Load-Based Performance Trends",
                                        3: "Power and Efficiency Trends",
                                        4: "Thrust and Vibration Response",
                                        # Graph 5 now uses only acceleration parameters (no SysEffect),
                                        # so the title reflects pure acceleration behaviour.
                                        5: "Acceleration Response",
                                    }
                                    if idx in graph_titles:
                                        graph_title = graph_titles[idx]
                                        # If vibration data is missing entirely, keep generic titles for Graph 4 and 5
                                        if idx in (4, 5):
                                            has_vibration_anywhere = any(
                                                "vibration" in str(col).lower() for col in df.columns
                                            )
                                            if not has_vibration_anywhere:
                                                graph_title = f"Graph {idx}"
                                    else:
                                        # For graphs beyond 5, label as Additional Graph 1, 2, ...
                                        graph_title = f"Additional Graph {idx - 5}"
                                    st.markdown(f"##### {graph_title}")
                                    param_col_saved, plot_col_saved = st.columns([0.25, 0.75])

                                    with param_col_saved:
                                        # Dedicated (editable) parameter section for this saved graph
                                        st.markdown("""
                                        <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                                            <span style='font-size: 1.2rem;'>🧮</span>
                                            <span style='font-size: 1.1rem; font-weight: 600;'>Parameters - Graph {idx}</span>
                                        </div>
                                        """.format(idx=idx), unsafe_allow_html=True)

                                        # X-axis (editable selectbox)
                                        # Get current X from session state if widget exists, otherwise from graph_params
                                        if f"multi_param_saved_{idx}_x_axis" in st.session_state:
                                            current_x = st.session_state[f"multi_param_saved_{idx}_x_axis"]
                                        else:
                                            current_x = graph_params.get("x_axis", numeric_cols[0] if numeric_cols else None)
                        
                                        if current_x in numeric_cols:
                                            x_index = numeric_cols.index(current_x)
                                        else:
                                            x_index = 0
                                        saved_x = st.selectbox(
                                            "X-Axis",
                                            numeric_cols,
                                            index=x_index,
                                            key=f"multi_param_saved_{idx}_x_axis",
                                        )
                                        # Sync back to graph_params and session state
                                        graph_params["x_axis"] = saved_x
                                        st.session_state[graph_key]["x_axis"] = saved_x

                                        # Y-axis candidates based on saved X
                                        y_candidates_saved = [c for c in numeric_cols if c != saved_x]
                        
                                        # Update graph_params if X-axis changed
                                        if graph_params.get("x_axis") != saved_x:
                                            # Filter out invalid Y-axis selections
                                            graph_params["left_y_axes"] = [c for c in graph_params.get("left_y_axes", []) if c in y_candidates_saved]
                                            graph_params["right_y_axes"] = [c for c in graph_params.get("right_y_axes", []) if c in y_candidates_saved]

                                        # Left Y-axis (editable multiselect)
                                        saved_left_key = f"multi_param_saved_{idx}_left_y"
                        
                                        # Get current selection: prioritize widget state (user's latest selection), then graph_params
                                        if saved_left_key in st.session_state:
                                            previous_left = st.session_state[saved_left_key]
                                        else:
                                            previous_left = graph_params.get("left_y_axes", [])
                        
                        
                                        # Use set intersection to ensure all default values are valid
                                        y_candidates_saved_set = set(y_candidates_saved)
                                        preserved_left = [col for col in previous_left if col in y_candidates_saved_set]
                                        # Final safety check: ensure all preserved values are in y_candidates_saved
                                        valid_default_left_saved = [col for col in preserved_left if col in y_candidates_saved]

                                        # If nothing valid is preserved (e.g., after switching data source), fall back to sensible defaults
                                        if not valid_default_left_saved and y_candidates_saved:
                                            if idx == 3:
                                                # For Graph 3, prefer Electrical Power on left Y (Thrust is X)
                                                fallback_left_saved = find_column_by_name(
                                                    y_candidates_saved,
                                                    [
                                                        "Electrical Power - W",
                                                        "Electrical Power (W)",
                                                        "Electrical Power",
                                                        "InPower - W",
                                                        "InPower",
                                                        "PowerInLine - W",
                                                        "PowerInLine",
                                                        "Power",
                                                    ],
                                                )
                                            else:
                                                # Other graphs: prefer Thrust on left Y
                                                fallback_left_saved = find_column_by_name(y_candidates_saved, "Thrust")
                                            if fallback_left_saved:
                                                valid_default_left_saved = [fallback_left_saved]
                        
                                        # Only update widget session state if it's missing or invalid (don't overwrite user's selection)
                                        # Cap at 2 params per axis for left Y
                                        max_left = 2
                                        if saved_left_key not in st.session_state or not all(col in y_candidates_saved for col in st.session_state.get(saved_left_key, [])):
                                            st.session_state[saved_left_key] = valid_default_left_saved[:max_left]
                                        else:
                                            st.session_state[saved_left_key] = (st.session_state.get(saved_left_key) or [])[:max_left]
                        
                                        saved_left = st.multiselect(
                                            "Left Y-Axis Parameters",
                                            y_candidates_saved,
                                            default=st.session_state.get(saved_left_key, valid_default_left_saved)[:max_left],
                                            key=saved_left_key,
                                            max_selections=max_left,
                                        )
                                        # Sync back to graph_params and session state (widget updates its own key automatically)
                                        graph_params["left_y_axes"] = saved_left
                                        st.session_state[graph_key]["left_y_axes"] = saved_left
                        

                                        # Right Y-axis (editable multiselect)
                                        right_candidates_saved = [c for c in y_candidates_saved if c not in saved_left]
                        
                                        saved_right_key = f"multi_param_saved_{idx}_right_y"
                        
                                        # Get current selection: prioritize widget state (user's latest selection), then graph_params
                                        if saved_right_key in st.session_state:
                                            previous_right = st.session_state[saved_right_key]
                                        else:
                                            previous_right = graph_params.get("right_y_axes", [])
                        
                                        # Use set intersection to ensure all default values are valid
                                        right_candidates_saved_set = set(right_candidates_saved)
                                        preserved_right = [col for col in previous_right if col in right_candidates_saved_set]
                                        # Final safety check: ensure all preserved values are in right_candidates_saved
                                        valid_default_right_saved = [col for col in preserved_right if col in right_candidates_saved]

                                        # If nothing valid is preserved, fall back to efficiency defaults
                                        # For Graph 3 we intentionally leave right Y blank by default
                                        if idx != 3 and not valid_default_right_saved and right_candidates_saved:
                                            fallback_right_saved = find_column_by_name(
                                                right_candidates_saved,
                                                ["SysEffect", "Overall Efficiency", "efficiency", "Efficiency"]
                                            )
                                            if fallback_right_saved:
                                                valid_default_right_saved = [fallback_right_saved]
                        
                                        # Only update widget session state if it's missing or invalid (don't overwrite user's selection)
                                        # Cap at 2 params per axis for right Y
                                        max_right = 2
                                        if saved_right_key not in st.session_state or not all(col in right_candidates_saved for col in st.session_state.get(saved_right_key, [])):
                                            st.session_state[saved_right_key] = valid_default_right_saved[:max_right]
                                        else:
                                            st.session_state[saved_right_key] = (st.session_state.get(saved_right_key) or [])[:max_right]
                        
                                        saved_right = st.multiselect(
                                            "Right Y-Axis Parameters",
                                            right_candidates_saved,
                                            default=st.session_state.get(saved_right_key, valid_default_right_saved)[:max_right],
                                            key=saved_right_key,
                                            max_selections=max_right,
                                        )
                                        # Sync back to graph_params and session state (widget updates its own key automatically)
                                        graph_params["right_y_axes"] = saved_right
                                        st.session_state[graph_key]["right_y_axes"] = saved_right

                                    with plot_col_saved:
                                        # Recreate plot with current parameters; use only columns that exist in data (respects renamed columns)
                                        df_filtered_saved = df.copy()
                                        plot_saved_x = saved_x if saved_x and saved_x in df.columns else (numeric_cols[0] if numeric_cols and numeric_cols[0] in df.columns else None)
                                        plot_saved_left = [c for c in (saved_left or []) if c in df.columns]
                                        plot_saved_right = [c for c in (saved_right or []) if c in df.columns]

                                        # If Graph 2 lost its Y-axes (e.g. after switching data source), re-seed sensible defaults
                                        if idx == 2 and not plot_saved_left and not plot_saved_right:
                                            fallback_left = find_column_by_name(
                                                [c for c in numeric_cols if c in df.columns],
                                                "Thrust"
                                            )
                                            fallback_right = find_column_by_name(
                                                [c for c in numeric_cols if c in df.columns],
                                                ["SysEffect", "Overall Efficiency", "efficiency", "Efficiency"]
                                            )
                                            fallback_x = plot_saved_x or find_column_by_name(
                                                [c for c in numeric_cols if c in df.columns],
                                                "Torque"
                                            ) or (numeric_cols[0] if numeric_cols else None)
                                            if fallback_x:
                                                plot_saved_x = fallback_x
                                                graph_params["x_axis"] = fallback_x
                                                st.session_state[graph_key]["x_axis"] = fallback_x
                                            if fallback_left:
                                                plot_saved_left = [fallback_left]
                                                graph_params["left_y_axes"] = plot_saved_left
                                                st.session_state[graph_key]["left_y_axes"] = plot_saved_left
                                            if fallback_right:
                                                plot_saved_right = [fallback_right]
                                                graph_params["right_y_axes"] = plot_saved_right
                                                st.session_state[graph_key]["right_y_axes"] = plot_saved_right

                                        # Create figure with dual Y-axes
                                        saved_fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                                        # Check if Y-axis parameters are selected
                                        if not plot_saved_left and not plot_saved_right:
                                            # No internal title; heading above the graph (Plot tab / report) shows the name
                                            saved_fig.update_layout(
                                                template="plotly_white",
                                                xaxis=dict(title=get_axis_title(saved_x)),
                                            )
                                            _register_plot_shown()
                                            st.plotly_chart(saved_fig, use_container_width=True)
                            
                                            # Store empty graph for report with same heading as Plot tab
                                            store_graph_for_report(f"Graph {idx}", saved_fig, table=None, heading=graph_title)
                                            # Skip the rest of the plot creation code - table will be shown below
                                        else:
                                            # Only create full plot if Y-axes are selected
                                            # Calculate individual Y-axis ranges for validated columns
                                            saved_left_ranges = {}
                                            if plot_saved_left:
                                                for col in plot_saved_left:
                                                    if col in df.columns:
                                                        col_values = df[col].dropna().tolist()
                                                        if col_values:
                                                            col_min = float(min(col_values))
                                                            col_max = float(max(col_values))
                                                            if col_min > 0:
                                                                col_min = 0.0
                                                            if col_max > 0:
                                                                col_buffer = max(col_max * 0.05, col_max * 0.02)
                                                                col_max = col_max + col_buffer
                                                            else:
                                                                col_max = 1.0
                                                            saved_left_ranges[col] = (col_min, col_max)
                            
                                            saved_right_ranges = {}
                                            if plot_saved_right:
                                                for col in plot_saved_right:
                                                    if col in df.columns:
                                                        col_values = df[col].dropna().tolist()
                                                        if col_values:
                                                            col_min = float(min(col_values))
                                                            col_max = float(max(col_values))
                                                            if col_min > 0:
                                                                col_min = 0.0
                                                            if col_max > 0:
                                                                col_buffer = max(col_max * 0.05, col_max * 0.02)
                                                                col_max = col_max + col_buffer
                                                            else:
                                                                col_max = 1.0
                                                            saved_right_ranges[col] = (col_min, col_max)
                            
                                            colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
                                            color_idx = 0
                                            color_map = {}
                            
                                            # Add traces for left Y-axis parameters (use validated columns)
                                            if plot_saved_left and plot_saved_x:
                                                for trace_idx, y_col in enumerate(plot_saved_left):
                                                    plot_data = df_filtered_saved[[plot_saved_x, y_col]].dropna()
                                                    if not plot_data.empty:
                                                        trace_color = colors[color_idx % len(colors)]
                                                        if trace_idx == 0:
                                                            saved_fig.add_trace(
                                                                go.Scatter(
                                                                    x=plot_data[plot_saved_x],
                                                                    y=plot_data[y_col],
                                                                    mode="lines+markers",
                                                                    name=get_display_name(y_col),
                                                                    line=dict(color=trace_color, width=2),
                                                                    marker=dict(size=4, color=trace_color),
                                                                ),
                                                                secondary_y=False,
                                                            )
                                                        else:
                                                            saved_fig.add_trace(
                                                                go.Scatter(
                                                                    x=plot_data[plot_saved_x],
                                                                    y=plot_data[y_col],
                                                                    mode="lines+markers",
                                                                    name=get_display_name(y_col),
                                                                    line=dict(color=trace_color, width=2),
                                                                    marker=dict(size=4, color=trace_color),
                                                                    yaxis='y3',
                                                                ),
                                                            )
                                                        color_map[y_col] = trace_color
                                                        color_idx += 1
                            
                                            # Add traces for right Y-axis parameters (use validated columns)
                                            if plot_saved_right and plot_saved_x:
                                                for trace_idx, y_col in enumerate(plot_saved_right):
                                                    plot_data = df_filtered_saved[[plot_saved_x, y_col]].dropna()
                                                    if not plot_data.empty:
                                                        trace_color = colors[color_idx % len(colors)]
                                                        if trace_idx == 0:
                                                            # Primary right‑axis parameter: keep markers for clarity
                                                            saved_fig.add_trace(
                                                                go.Scatter(
                                                                    x=plot_data[plot_saved_x],
                                                                    y=plot_data[y_col],
                                                                    mode="lines+markers",
                                                                    name=get_display_name(y_col),
                                                                    line=dict(color=trace_color, width=2, dash="solid"),
                                                                    marker=dict(size=4, color=trace_color),
                                                                ),
                                                                secondary_y=True,
                                                            )
                                                        else:
                                                            # Second right‑axis parameter: show as a clean solid line
                                                            # on its own offset right Y‑axis (y4).
                                                            saved_fig.add_trace(
                                                                go.Scatter(
                                                                    x=plot_data[plot_saved_x],
                                                                    y=plot_data[y_col],
                                                                    mode="lines",
                                                                    name=get_display_name(y_col),
                                                                    line=dict(color=trace_color, width=2, dash="solid"),
                                                                    yaxis="y4",
                                                                ),
                                                            )
                                                        color_map[y_col] = trace_color
                                                        color_idx += 1
                            
                                            # Set axes (use validated columns)
                                            if plot_saved_x:
                                                saved_fig.update_xaxes(
                                                    title_text=get_axis_title(plot_saved_x),
                                                    title_font=dict(size=22, color="black"),
                                                    tickfont=dict(size=18, color="black"),
                                                )
                            
                                            # Decide layout mode:
                                            # - multi_axis_mode: more than one parameter on either side
                                            # - dual_axis: at least one parameter on both left and right
                                            multi_axis_mode = (
                                                (plot_saved_left and len(plot_saved_left) > 1)
                                                or (plot_saved_right and len(plot_saved_right) > 1)
                                            )
                                            dual_axis = bool(plot_saved_left and plot_saved_right)

                                            # Mark layout meta so the report export helper can detect
                                            # multi‑parameter Y‑axis figures and adjust fonts/titles.
                                            try:
                                                current_meta = saved_fig.layout.meta or {}
                                                if not isinstance(current_meta, dict):
                                                    current_meta = {}
                                                current_meta.update({
                                                    # Mark multi_y_params when any side has 2+ params
                                                    # so export helpers adjust margins/titles.
                                                    "multi_y_params": bool(multi_axis_mode),
                                                    "multi_y_left_count": len(plot_saved_left or []),
                                                    "multi_y_right_count": len(plot_saved_right or []),
                                                })
                                                saved_fig.update_layout(meta=current_meta)
                                            except Exception:
                                                pass

                                            # Set Y-axis titles and ranges (use validated columns).
                                            # Build all axes in a single layout dict so that combinations
                                            # of 1–2 left and 1–2 right parameters are handled robustly.
                                            axis_layout = {}

                                            # Primary left axis (yaxis)
                                            if plot_saved_left and plot_saved_left[0] in saved_left_ranges:
                                                y_min, y_max = saved_left_ranges[plot_saved_left[0]]
                                                axis_layout["yaxis"] = dict(
                                                    # When any side has 2+ params (multi_axis_mode),
                                                    # remove side title; rely on top annotations.
                                                    # Classic dual-axis (1+1) and single-axis keep side titles.
                                                    title="" if multi_axis_mode else get_display_name(plot_saved_left[0]),
                                                    range=[y_min, y_max],
                                                    showgrid=True,
                                                    gridcolor="#999999",
                                                    gridwidth=1,
                                                    showline=True,
                                                    linecolor="#333333",
                                                    linewidth=1.5,
                                                    # Use this side as the single visible
                                                    # Y=0 baseline for the plot, in a lighter grey
                                                    # so only the true axis line is solid black.
                                                    zeroline=True,
                                                    zerolinecolor="#999999",
                                                    zerolinewidth=1.0,
                                                    title_font=dict(size=22, color="black"),
                                                    tickfont=dict(size=18, color="black"),
                                                    side="left",
                                                    position=0.0,
                                                )

                                            # Second left axis (yaxis3), if any
                                            if plot_saved_left and len(plot_saved_left) > 1 and plot_saved_left[1] in saved_left_ranges:
                                                y_min, y_max = saved_left_ranges[plot_saved_left[1]]
                                                axis_layout["yaxis3"] = dict(
                                                    title="",  # secondary left; top labels / legend carry meaning
                                                    title_font=dict(size=22, color="black"),
                                                    tickfont=dict(size=18, color="black"),
                                                    # Second left axis: free‑anchored and overlaid on the
                                                    # primary left axis, slightly inside so both scales are visible.
                                                    anchor="free",
                                                    overlaying="y",
                                                    side="left",
                                                    position=0.05,
                                                    range=[y_min, y_max],
                                                    showgrid=False,
                                                    showline=True,
                                                    linecolor="#333333",
                                                    linewidth=1.5,
                                                    showticklabels=True,
                                                )

                                            # Primary right axis (yaxis2)
                                            if plot_saved_right and plot_saved_right[0] in saved_right_ranges:
                                                y_min, y_max = saved_right_ranges[plot_saved_right[0]]
                                                axis_layout["yaxis2"] = dict(
                                                    # When any side has 2+ params (multi_axis_mode),
                                                    # remove side title; rely on top annotations.
                                                    title="" if multi_axis_mode else get_display_name(plot_saved_right[0]),
                                                    range=[y_min, y_max],
                                                    showgrid=False,
                                                    # Keep only the left side as the visual baseline:
                                                    # right axis line is drawn but its zeroline is hidden.
                                                    showline=True,
                                                    linecolor="#333333",
                                                    linewidth=1.5,
                                                    zeroline=False,
                                                    title_font=dict(size=22, color="black"),
                                                    tickfont=dict(size=18, color="black"),
                                                    overlaying="y",
                                                    side="right",
                                                    position=1.0,
                                                )

                                            # Second right axis (yaxis4), if any
                                            if plot_saved_right and len(plot_saved_right) > 1 and plot_saved_right[1] in saved_right_ranges:
                                                y_min, y_max = saved_right_ranges[plot_saved_right[1]]
                                                axis_layout["yaxis4"] = dict(
                                                    title="",  # secondary right; top labels / legend carry meaning
                                                    title_font=dict(size=22, color="black"),
                                                    tickfont=dict(size=18, color="black"),
                                                    # Second right axis: free‑anchored and overlaid on the
                                                    # primary right axis, slightly inside so both right
                                                    # scales are distinct.
                                                    anchor="free",
                                                    overlaying="y",
                                                    side="right",
                                                    position=0.90,
                                                    range=[y_min, y_max],
                                                    showgrid=False,
                                                    showline=True,
                                                    linecolor="#333333",
                                                    linewidth=1.5,
                                                    showticklabels=True,
                                                )

                                            if axis_layout:
                                                saved_fig.update_layout(**axis_layout)
                            
                                            saved_fig.update_xaxes(
                                                showgrid=True,
                                                gridcolor='#999999',
                                                gridwidth=1,
                                                showline=True,
                                                linecolor='#333333',
                                                linewidth=1.5,
                                                nticks=16,
                                                title_font=dict(size=22, color="black"),
                                                tickfont=dict(size=18, color="black"),
                                            )
                            
                                            # No internal title; heading above the graph (Plot tab / report) shows the name
                                            _has_dual_axis = bool(plot_saved_left and plot_saved_right)
                                            saved_fig.update_layout(
                                                showlegend=True,
                                                legend=dict(
                                                    orientation="h",
                                                    yanchor="bottom",
                                                    y=1.02,
                                                    # Place legend at the top‑center of the plot
                                                    xanchor="center",
                                                    x=0.5,
                                                    title_font=dict(size=16, color="black"),
                                                    font=dict(size=16, color="black"),
                                                ),
                                                template="plotly_white",
                                                # Use larger top margin when top labels are present
                                                # (multi_axis_mode) so annotations are visible.
                                                margin=dict(
                                                    l=40 if multi_axis_mode else 60,
                                                    r=40 if multi_axis_mode else 60,
                                                    t=90 if multi_axis_mode else 60,
                                                    b=50,
                                                ),
                                                hovermode='x unified',
                                            )

                                            # Add axis labels at the top for dual-axis (1+1 or multi-param).
                                            add_top_param_labels(saved_fig, plot_saved_left, plot_saved_right, color_map)
                            
                                            _register_plot_shown()
                                            st.plotly_chart(saved_fig, use_container_width=True)

                                            # Store saved graph for report using the same heading as in the Plot tab (e.g. "Speed-Based Performance Trends")
                                            store_graph_for_report(f"Graph {idx}", saved_fig, table=None, heading=graph_title)
                
                                # Force X-axis defaults for first three graphs so they don't
                                # all collapse to throttle when older template/session values exist.
                                _g1_x_pref = find_column_by_name(
                                    numeric_cols,
                                    ["RPM", "RPM1 - RPM", "Motor Electrical Speed (RPM)", "Motor Optical Speed (RPM)"],
                                )
                                _g2_x_pref = find_column_by_name(
                                    numeric_cols,
                                    ["Torque (N·m)", "Torque (N*m)", "Torque (Nm)", "Torque - N*m", "Torque", "torque"],
                                )
                                _g3_x_pref = find_column_by_name(
                                    numeric_cols,
                                    ["Thrust (gf)", "Thrust - gf", "Thrust (kgf)", "Thrust"],
                                )
                                if is_rotrix_data and time_x_plot:
                                    _forced_x_by_graph = {1: time_x_plot, 2: time_x_plot, 3: time_x_plot}
                                else:
                                    _forced_x_by_graph = {1: _g1_x_pref, 2: _g2_x_pref, 3: _g3_x_pref}
                                for _gi, _x_pref in _forced_x_by_graph.items():
                                    if not _x_pref:
                                        continue
                                    _gkey = f"multi_param_graph_{_gi}"
                                    _current = st.session_state.get(_gkey)
                                    if isinstance(_current, dict):
                                        _current["x_axis"] = _x_pref
                                        st.session_state[_gkey] = _current
                                    # Keep the corresponding widget key in sync so selectbox
                                    # reflects the enforced X-axis choice on rerun.
                                    st.session_state[f"multi_param_saved_{_gi}_x_axis"] = _x_pref

                                # Graph 1: Performance & Efficiency vs RPM – X=RPM, Left=Thrust (gf), Right=SysEffect (gf/W)
                                rpm_col = find_column_by_name(numeric_cols, ["RPM", "RPM1 - RPM", "Motor Electrical Speed (RPM)", "Time (s)"])
                                thrust_col = find_column_by_name(numeric_cols, ["Thrust (gf)", "Thrust - gf", "Thrust"])
                                syseffect_col = find_column_by_name(
                                    numeric_cols,
                                    ["SysEffect (gf/W)", "SysEffect - gf/W", "SysEffect", "Overall Efficiency", "efficiency", "Efficiency"],
                                )
                                graph1_defaults = {
                                    "x_axis": (
                                        time_x_plot
                                        if is_rotrix_data and time_x_plot
                                        else (rpm_col or (numeric_cols[0] if numeric_cols else None))
                                    ),
                                    "left_y_axes": [thrust_col] if thrust_col else [],
                                    "right_y_axes": [syseffect_col] if syseffect_col else [],
                                    "smoothing_enabled": False,
                                    "smoothing_method": "savgol",
                                    "smoothing_window": 5,
                                }
                                render_saved_graph(1, graph1_defaults)

                                # Graph 2 (from saved_graphs[0] or fallback with same robust names)
                                graph2_saved = st.session_state.multi_param_saved_graphs[0] if len(st.session_state.multi_param_saved_graphs) > 0 else None
                                if graph2_saved:
                                    render_saved_graph(2, graph2_saved)
                                else:
                                    torque_col = find_column_by_name(numeric_cols, ["Torque (N·m)", "Torque (N*m)", "Torque", "torque"])
                                    thrust_col = find_column_by_name(numeric_cols, ["Thrust (gf)", "Thrust - gf", "Thrust"])
                                    syseffect_col = find_column_by_name(numeric_cols, ["SysEffect (gf/W)", "SysEffect", "Overall Efficiency", "efficiency", "Efficiency"])
                                    graph2_defaults = {
                                        "x_axis": (
                                            time_x_plot
                                            if is_rotrix_data and time_x_plot
                                            else (torque_col or (numeric_cols[0] if numeric_cols else None))
                                        ),
                                        "left_y_axes": [thrust_col] if thrust_col else [],
                                        "right_y_axes": [syseffect_col] if syseffect_col else [],
                                        "smoothing_enabled": False,
                                        "smoothing_method": "savgol",
                                        "smoothing_window": 5,
                                    }
                                    render_saved_graph(2, graph2_defaults)
                
                                # Always show Graph 3 as an empty template
                                render_saved_graph(3, is_empty=True)
                
                                # Show Graph 4+ from saved_graphs[1:] if they exist
                                for idx_offset, saved in enumerate(st.session_state.multi_param_saved_graphs[1:], start=4):
                                    render_saved_graph(idx_offset, saved)

                                # Add Graph and Remove Last Graph buttons below all graphs
                                btn_col_clear1, btn_col_add, btn_col_remove, btn_col_clear = st.columns([0.34, 0.15, 0.25, 0.34])
                                with btn_col_add:
                                    # Always allow adding graph (Graph 4+) – Graph 4 uses preset: Time (s), Thrust (gf), Vibration (g)
                                    if st.button("➕ Add Graph", key="multi_param_add_graph"):
                                        is_graph4 = len(st.session_state.multi_param_saved_graphs) == 1
                                        if is_graph4:
                                            # Graph 4 preset (same as initial): Time (s), Thrust (gf) & Overall Efficiency (gf/W), Vibration (g)
                                            time_col = find_column_by_name(numeric_cols, ["Time (s)", "Time", "timestamp_seconds", "Time (secs)"])
                                            thrust_col = find_column_by_name(numeric_cols, ["Thrust (gf)", "Thrust - gf", "Thrust"])
                                            eff_col = find_column_by_name(
                                                numeric_cols,
                                                ["Overall Efficiency", "SysEffect (gf/W)", "SysEffect - gf/W", "SysEffect", "efficiency", "Efficiency"],
                                            )
                                            vibration_col = find_column_by_name(numeric_cols, ["Vibration RMS (g)", "Vibration (g)", "Vibration - g", "Vibration"])
                                            save_x_axis = time_col or (numeric_cols[0] if numeric_cols else None)
                                            save_left = [c for c in [thrust_col, eff_col] if c]
                                            save_right = [vibration_col] if vibration_col else []
                                        else:
                                            save_x_axis = numeric_cols[0] if numeric_cols else None
                                            save_left = []
                                            save_right = []
                                        if save_x_axis is not None:
                                            st.session_state.multi_param_saved_graphs.append(
                                                {
                                                    "x_axis": save_x_axis,
                                                    "left_y_axes": save_left,
                                                    "right_y_axes": save_right,
                                                    "smoothing_enabled": False,
                                                    "smoothing_method": "savgol",
                                                    "smoothing_window": 5,
                                                    "fig": None
                                                }
                                            )
                                            st.rerun()
                                with btn_col_remove:
                                    if st.button("➖ Remove Last Graph", key="multi_param_remove_graph"):
                                        # Removal rules depend on whether vibration data exists:
                                        # - With vibration columns: keep preset graphs 2, 4, 5 (saved_graphs[0..2]) -> only remove Graph 6+
                                        # - Without vibration columns: only Graph 2 is preset -> allow removing Graph 4+ (saved_graphs[1+])
                                        has_vibration_anywhere = any("vibration" in str(col).lower() for col in df.columns)
                                        min_preset_count = 3 if has_vibration_anywhere else 1
                                        if len(st.session_state.multi_param_saved_graphs) > min_preset_count:
                                            # Map the last saved-graph index to its actual Graph N number
                                            # used in the UI and in report_graph_entries.
                                            saved_count = len(st.session_state.multi_param_saved_graphs)
                                            removed_saved_idx = saved_count - 1  # 0‑based in multi_param_saved_graphs
                                            removed_entry = st.session_state.multi_param_saved_graphs[removed_saved_idx]
                                            explicit_num = removed_entry.get("graph_number") if isinstance(removed_entry, dict) else None
                                            if isinstance(explicit_num, int):
                                                graph_number = explicit_num
                                            else:
                                                if removed_saved_idx == 0:
                                                    graph_number = 2  # first saved graph is always Graph 2
                                                else:
                                                    # saved_idx 1 -> Graph 4, 2 -> Graph 5, 3 -> Graph 6, ...
                                                    graph_number = removed_saved_idx + 3

                                            graph_key = f"multi_param_graph_{graph_number}"
                                            # Remove widget / parameter state for this saved graph
                                            if graph_key in st.session_state:
                                                del st.session_state[graph_key]
                                            # Also remove this graph from the report cache so it no longer
                                            # appears in the Report tab's graph selection after rerun.
                                            try:
                                                if "report_graph_entries" in st.session_state:
                                                    st.session_state.report_graph_entries.pop(f"Graph {graph_number}", None)
                                            except Exception:
                                                pass
                                            # Finally remove the last saved-graph definition itself
                                            st.session_state.multi_param_saved_graphs.pop()
                                            st.rerun()
                                        else:
                                            suffix = "Graph 6 and above" if has_vibration_anywhere else "Graph 4 and above"
                                            st.info(f"Preset graphs cannot be removed. Only additional graphs ({suffix}) can be removed.")

                                # Report always uses Sorted performance table when available; sync only Sorted so report preview shows correct source
                                _has_sorted = getattr(st.session_state, "report_sorted_table_df", None) is not None
                                st.session_state.report_graphs_data_source = (
                                    "Sorted performance table" if _has_sorted else st.session_state.get("multi_param_plot_data_source", default_plot_source)
                                )
                    except Exception as e:
                        # Show a concise error without full traceback to avoid noisy logs.
                        st.error(f"An error occurred while updating the report settings: {e}")
                else:
                    # Report always uses Sorted performance table when available; keep report data source correct when viewing template list
                    if getattr(st.session_state, "report_sorted_table_df", None) is not None:
                        st.session_state.report_graphs_data_source = "Sorted performance table"
                    # Row: space (left) + Add New button (right); "Report template" heading is shown above
                    _rt_left, _rt_right = st.columns([3, 1])
                    with _rt_left:
                        pass
                    with _rt_right:
                        if st.button("➕ Add New", key="report_multi_add_new_btn", type="primary", use_container_width=True):
                            st.session_state.report_multi_add_new_view = True
                            st.rerun()
                    # Table-style list: header Template Name | Date | Created by | actions
                    profiles = _load_sorted_report_profiles()
                    if profiles:
                        # Header row (like the picture: Template Name, Date, Created by, icons)
                        _h_cols = st.columns([0.5, 3.5, 2, 2, 1, 1, 1])
                        with _h_cols[0]:
                            st.markdown("")
                        with _h_cols[1]:
                            st.markdown("**Template Name**")
                        with _h_cols[2]:
                            st.markdown("**Date**")
                        with _h_cols[3]:
                            st.markdown("**Created by**")
                        with _h_cols[4]:
                            st.markdown("")
                        with _h_cols[5]:
                            st.markdown("")
                        with _h_cols[6]:
                            st.markdown("")
                        _selected_profile_idx = st.session_state.get("report_profile_selected_idx", None)
                        for _idx, _profile in enumerate(profiles):
                            _name = (_profile.get("name") or "").strip()
                            _date = (_profile.get("created_at") or "—").strip() if _profile.get("created_at") else "—"
                            _by = (_profile.get("created_by") or "—").strip() if _profile.get("created_by") else "—"
                            _p_cols = st.columns([0.6, 3.5, 2, 2, 1, 1, 1])
                            with _p_cols[0]:
                                _is_selected = (_selected_profile_idx == _idx)
                                _radio_icon = "🔘" if _is_selected else "⚪"
                                if st.button(_radio_icon, key=f"report_profile_radio_{_idx}", use_container_width=True):
                                    st.session_state.report_profile_selected_idx = _idx
                                    # Apply the selected profile immediately so the right-hand settings
                                    # and preview use the saved configuration.
                                    apply_profile_to_session_state(_profile)
                                    # After applying profile, refresh report graphs so that the preview
                                    # column reflects the newly selected template.
                                    invalidate_report_after_data_change()
                                    ensure_report_graphs_for_current_config()
                                    # Clear any existing preview images so the next Update report
                                    # preview action generates a fresh PDF.
                                    st.session_state.report_preview_images = []
                                    st.rerun()
                            with _p_cols[1]:
                                st.markdown(f"**{_name or '—'}**")
                                _desc = (_profile.get("description") or "").strip()
                                if _desc:
                                    st.caption(_desc)
                            with _p_cols[2]:
                                st.caption(_date)
                            with _p_cols[3]:
                                st.caption(_by)
                            with _p_cols[4]:
                                if st.button("🔍", key=f"report_profile_preview_{_idx}", use_container_width=True, help="Quick Preview"):
                                    _current_idx = st.session_state.get("report_quick_preview_idx")
                                    _is_open = st.session_state.get("report_quick_preview_open", False)
                                    if _is_open and _current_idx == _idx:
                                        st.session_state["report_quick_preview_open"] = False
                                        st.session_state["report_quick_preview_idx"] = None
                                    else:
                                        st.session_state["report_quick_preview_idx"] = _idx
                                        st.session_state["report_quick_preview_open"] = True
                                    st.rerun()
                            with _p_cols[5]:
                                if st.button("✏️", key=f"report_profile_edit_{_idx}", use_container_width=True, help="Edit"):
                                    st.session_state["report_profile_edit_idx"] = _idx
                                    st.session_state["report_edit_mode"] = True
                                    st.session_state["report_multi_add_new_view"] = True
                                    # Pre-fill name/description for the edit form
                                    st.session_state["report_save_profile_name"] = _name or ""
                                    st.session_state["report_save_profile_desc"] = _desc or ""
                                    # Apply settings from this profile into the current session state
                                    apply_profile_to_session_state(_profile)
                                    # Land on Add New page (tabs); user clicks Save to open save form
                                    st.session_state["report_show_save_profile_form"] = False
                                    # Close any quick-preview sidebar when going into edit
                                    st.session_state["report_quick_preview_open"] = False
                                    st.session_state["report_quick_preview_idx"] = None
                                    st.rerun()
                            with _p_cols[6]:
                                if st.button("🗑️", key=f"report_profile_delete_{_idx}", use_container_width=True, help="Delete"):
                                    _qp_idx = st.session_state.get("report_quick_preview_idx", None)
                                    updated_profiles = [p for i, p in enumerate(profiles) if i != _idx]
                                    save_profiles(updated_profiles)
                                    _n = len(updated_profiles)
                                    if _selected_profile_idx >= _n:
                                        st.session_state.report_profile_selected_idx = max(0, _n - 1)
                                    elif _idx == _selected_profile_idx:
                                        st.session_state.report_profile_selected_idx = min(_idx, _n - 1) if _n else None
                                    elif _idx < _selected_profile_idx:
                                        st.session_state.report_profile_selected_idx = _selected_profile_idx - 1
                                    if _qp_idx is not None:
                                        if _qp_idx == _idx:
                                            st.session_state["report_quick_preview_open"] = False
                                            st.session_state["report_quick_preview_idx"] = None
                                        elif _idx < _qp_idx:
                                            st.session_state["report_quick_preview_idx"] = _qp_idx - 1
                                    st.rerun()
                        # Right-side quick preview sidebar (throttle + plot settings)
                        _qp_open = st.session_state.get("report_quick_preview_open", False)
                        _qp_idx = st.session_state.get("report_quick_preview_idx", None)
                        if _qp_open and isinstance(_qp_idx, int) and 0 <= _qp_idx < len(profiles):
                            _qp_profile = profiles[_qp_idx]
                            _qp_name = (_qp_profile.get("name") or "").strip()
                            _qp_desc = (_qp_profile.get("description") or "").strip()
                            _qp_date = (_qp_profile.get("created_at") or "—").strip() if _qp_profile.get("created_at") else "—"
                            _qp_by = (_qp_profile.get("created_by") or "—").strip() if _qp_profile.get("created_by") else "—"
                            _qp_throttle = _qp_profile.get("throttle_aggregation") or {}
                            _qp_graphs = _qp_profile.get("saved_graphs") or []
                            _qp_file = (st.session_state.get("multi_param_selected_file") or st.session_state.get("multi_param_file_selector") or "—").strip() or "—"
                            # Escape all user-provided values for safe HTML rendering
                            _qp_name_safe = html.escape(_qp_name) if _qp_name else "Untitled template"
                            _qp_desc_safe = html.escape(_qp_desc) if _qp_desc else ""
                            _qp_date_safe = html.escape(_qp_date) if _qp_date else "—"
                            _qp_by_safe = html.escape(_qp_by) if _qp_by else "—"
                            _qp_file_safe = html.escape(_qp_file) if _qp_file else "—"

                            _th_start = _qp_throttle.get("start_throttle")
                            _th_end = _qp_throttle.get("end_throttle")
                            _th_step = _qp_throttle.get("throttle_interval")
                            _th_mode = _qp_throttle.get("ramp_mode")

                            _th_lines = []
                            if _th_start is not None:
                                _th_lines.append(f"<div class='report-quick-preview-item'>Start throttle: <strong>{html.escape(str(_th_start))}</strong></div>")
                            if _th_end is not None:
                                _th_lines.append(f"<div class='report-quick-preview-item'>End throttle: <strong>{html.escape(str(_th_end))}</strong></div>")
                            if _th_step is not None:
                                _th_lines.append(f"<div class='report-quick-preview-item'>Interval: <strong>{html.escape(str(_th_step))}</strong></div>")
                            if isinstance(_th_mode, str) and _th_mode:
                                _th_lines.append(f"<div class='report-quick-preview-item'>Ramp mode: <strong>{html.escape(_th_mode)}</strong></div>")
                            if not _th_lines:
                                _th_lines.append("<div class='report-quick-preview-item'>No throttle aggregation settings saved.</div>")

                            _graphs_html_parts = []
                            if isinstance(_qp_graphs, list) and _qp_graphs:
                                for gi, g in enumerate(_qp_graphs[:6]):
                                    _gx = html.escape(str(g.get("x_axis") or "—"))
                                    _gleft = g.get("left_y_axes") or []
                                    _gright = g.get("right_y_axes") or []
                                    _graphs_html_parts.append(
                                        f"<div class='report-quick-preview-item'><strong>Graph {gi + 1}</strong>: X = <code>{_gx}</code></div>"
                                    )
                                    if _gleft:
                                        _left_pills = "".join(f"<span class='report-quick-preview-pill'>{html.escape(str(v))}</span>" for v in _gleft)
                                        _graphs_html_parts.append(f"<div class='report-quick-preview-item'>Left Y: {_left_pills}</div>")
                                    if _gright:
                                        _right_pills = "".join(f"<span class='report-quick-preview-pill'>{html.escape(str(v))}</span>" for v in _gright)
                                        _graphs_html_parts.append(f"<div class='report-quick-preview-item'>Right Y: {_right_pills}</div>")
                                    _graphs_html_parts.append("<div style='height:0.35rem;'></div>")
                            else:
                                _graphs_html_parts.append("<div class='report-quick-preview-item'>No saved plot settings for this template.</div>")

                            _desc_row = f"<div class='report-quick-preview-row'><span class='report-quick-preview-label'>Description:</span><span class='report-quick-preview-value'>{_qp_desc_safe}</span></div>" if _qp_desc_safe else ""

                            _qp_sidebar_html = (
                                "<style>"
                                ".report-quick-preview-sidebar{position:fixed;top:4.5rem;right:0;width:420px;max-width:100%;height:calc(100vh - 4.5rem - 70px);background-color:#fff;box-shadow:-2px 0 12px rgba(15,23,42,.18);border-left:1px solid rgba(148,163,184,.5);padding:1rem 1.5rem;overflow-y:auto;z-index:998;display:flex;flex-direction:column;margin-top:40px;}"
                                ".report-quick-preview-header-wrapper{position:relative;margin-bottom:.5rem;}"
                                ".report-quick-preview-close-x-wrapper{position:absolute;top:0;right:0;z-index:1000;}"
                                ".report-quick-preview-content{flex:1;overflow-y:auto;padding-bottom:.5rem;}"
                                ".report-quick-preview-title{font-size:1.5rem;font-weight:700;margin-bottom:.75rem;color:#1f2937;}"
                                ".report-quick-preview-row{display:flex;align-items:baseline;margin-bottom:.5rem;font-size:1rem;position:relative;}"
                                ".report-quick-preview-row-header{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:.5rem;font-size:1rem;position:relative;}"
                                ".report-quick-preview-label{font-size:1rem;font-weight:600;color:#4b5563;margin-right:.5rem;flex-shrink:0;}"
                                ".report-quick-preview-value{font-size:1rem;color:#111827;flex:1;}"
                                ".report-quick-preview-close-x-inline{width:24px;height:24px;display:inline-flex;align-items:center;justify-content:center;cursor:pointer;border-radius:4px;background-color:#f3f4f6;color:#6b7280;font-size:1.1rem;font-weight:600;line-height:1;transition:background-color .2s,color .2s;flex-shrink:0;margin-left:auto;}"
                                ".report-quick-preview-close-x-inline:hover{background-color:#e5e7eb;color:#111827;}"
                                ".report-quick-preview-meta{font-size:1rem;color:#6b7280;margin-bottom:1rem;}"
                                ".report-quick-preview-section{margin-bottom:1.25rem;}"
                                ".report-quick-preview-section-title{font-size:1.2rem;font-weight:700;margin-bottom:.5rem;color:#1f2937;}"
                                ".report-quick-preview-item{font-size:1rem;margin-bottom:.35rem;color:#374151;}"
                                ".report-quick-preview-pill{display:inline-block;padding:.2rem .6rem;margin:.15rem .15rem .15rem 0;border-radius:999px;background-color:#f3f4f6;font-size:.95rem;color:#111827;}"
                                "</style>"
                                "<div class='report-quick-preview-sidebar'>"
                                "<div class='report-quick-preview-content'>"
                                "<div class='report-quick-preview-row-header'>"
                                "<div style='display:flex;align-items:baseline;flex:1;'>"
                                f"<span class='report-quick-preview-label'>Name:</span>"
                                f"<span class='report-quick-preview-value'>{_qp_name_safe}</span>"
                                "</div></div>"
                                f"<div class='report-quick-preview-row'><span class='report-quick-preview-label'>File:</span><span class='report-quick-preview-value'>{_qp_file_safe}</span></div>"
                                f"<div class='report-quick-preview-row'><span class='report-quick-preview-label'>Date:</span><span class='report-quick-preview-value'>{_qp_date_safe}</span></div>"
                                f"<div class='report-quick-preview-row'><span class='report-quick-preview-label'>Created by:</span><span class='report-quick-preview-value'>{_qp_by_safe}</span></div>"
                                + _desc_row +
                                "<div class='report-quick-preview-row'><span class='report-quick-preview-label'>Data source:</span><span class='report-quick-preview-value'>Sorted performance table</span></div>"
                                "<div class='report-quick-preview-section'>"
                                "<div class='report-quick-preview-section-title'>Throttle aggregation</div>"
                                + "".join(_th_lines) +
                                "</div>"
                                "<div class='report-quick-preview-section'>"
                                "<div class='report-quick-preview-section-title'>Plot settings</div>"
                                + "".join(_graphs_html_parts) +
                                "</div>"
                                "</div></div>"
                            )
                            st.markdown(_qp_sidebar_html, unsafe_allow_html=True)
                            
                            # X button positioned inline with name row using absolute positioning
                            if st.button("×", key="report_quick_preview_close_x_btn", help="Close sidebar"):
                                st.session_state["report_quick_preview_open"] = False
                                st.session_state["report_quick_preview_idx"] = None
                                st.rerun()
                            
                            # Style the X button to position it inline with the name row
                            st.markdown(
                                """
                                <style>
                                button[data-testid="baseButton-secondary"][aria-label*="Close sidebar"] {
                                    position: fixed !important;
                                    top: calc(4.5rem + 40px + 1rem + 0.125rem) !important;
                                    right: calc(420px - 1.5rem - 24px) !important;
                                    width: 24px !important;
                                    height: 24px !important;
                                    min-width: 24px !important;
                                    padding: 0 !important;
                                    border-radius: 4px !important;
                                    background-color: #f3f4f6 !important;
                                    color: #6b7280 !important;
                                    font-size: 1.1rem !important;
                                    font-weight: 600 !important;
                                    z-index: 1000 !important;
                                    border: none !important;
                                    box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
                                }
                                button[data-testid="baseButton-secondary"][aria-label*="Close sidebar"]:hover {
                                    background-color: #e5e7eb !important;
                                    color: #111827 !important;
                                }
                                </style>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No saved profiles yet for report templates.")
                # Bottom row: Update report preview | Proceed (when template selected) | Cancel + Save (Add New only)
                if not show_save_form:
                    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                    _preview_btn_col, _proceed_btn_col, _cancel_btn_col, _save_btn_col = st.columns([1, 1, 0.5, 0.5])
                    with _preview_btn_col:
                        if st.button("Update report preview", key="report_preview_btn"):
                            # Guard: require a file; if report data is missing, build it from the selected file.
                            if selected_file == "None":
                                st.warning("Please select a file before updating the report preview.")
                            else:
                                _has_sorted = getattr(st.session_state, "report_sorted_table_df", None) is not None
                                _loaded_report_file = st.session_state.get("report_loaded_file_name")
                                _report_raw_df = getattr(st.session_state, "report_raw_data_df", None)
                                _has_summary_cols = (
                                    _report_raw_df is not None
                                    and not getattr(_report_raw_df, "empty", True)
                                    and "Throttle Range (10%)" in _report_raw_df.columns
                                    and "Time range" in _report_raw_df.columns
                                )
                                _needs_rebuild = (
                                    (not _has_sorted)
                                    or (_loaded_report_file != selected_file)
                                    or (not _has_summary_cols)
                                )
                                if _needs_rebuild:
                                    # Auto-build raw + sorted data for the selected file using current template settings.
                                    _raw_df, _sorted_df = _load_file_and_build_report_data(
                                        selected_file,
                                        st.session_state.uploaded_files,
                                    )
                                    if _raw_df is None or _sorted_df is None or _sorted_df.empty:
                                        st.error("Could not build the Sorted Performance Table for this file. Please check the data and template settings.")
                                        st.stop()
                                    st.session_state["multi_param_raw_df"] = _raw_df
                                    st.session_state["report_sorted_table_df"] = _sorted_df
                                    st.session_state["multi_param_selected_file"] = selected_file
                                    st.session_state["report_loaded_file_name"] = selected_file
                                    _report_raw = _build_report_raw_data_df(_raw_df)
                                    st.session_state["report_raw_data_df"] = _report_raw
                                # Always clear cached report figures/preview so Update report preview
                                # regenerates Section 2.1 graphs with the latest dwell-time logic.
                                invalidate_report_after_data_change()
                                # Same inputs as batch PDF: authoritative raw df → summary columns → graphs
                                _refresh_report_raw_data_df()
                                st.session_state.report_graphs_data_source = "Sorted performance table"
                                ensure_summary_graphs_for_current_file()
                                ensure_report_graphs_for_current_config()
                                _graph_entries = getattr(st.session_state, "report_graph_entries", {}) or {}
                                _graph_keys = list(_graph_entries.keys())
                                _table_keys = []
                                _sorted_now = getattr(st.session_state, "report_sorted_table_df", None)
                                if _sorted_now is not None and not getattr(_sorted_now, "empty", True):
                                    _table_keys.append("Sorted Performance Table")
                                _company = (st.session_state.get("author_company") or "").strip()
                                _user = (st.session_state.get("user_name") or st.session_state.get("author_name") or "").strip()
                                _org_id = st.session_state.get("organization_id")
                                _logo_path = get_org_logo_path(_org_id) if _org_id else None
                                # Prefer auto-detected file info; otherwise, fall back to manual text if provided.
                                _auto_info = (st.session_state.get("report_file_info_text") or "").strip()
                                _manual_info = (st.session_state.get("manual_file_info_text") or "").strip()
                                if _manual_info and not _auto_info:
                                    st.session_state.report_file_info_text = _manual_info
                                _pdf_bytes = build_report_pdf(
                                    include_info=True,
                                    selected_graph_keys=_graph_keys,
                                    selected_table_keys=_table_keys,
                                    include_cover_page=True,
                                    include_table_of_contents=True,
                                    cover_company_name=_company,
                                    cover_user_name=_user,
                                    cover_logo_path=_logo_path,
                                )
                                if _pdf_bytes and len(_pdf_bytes) > 0:
                                    doc = None
                                    try:
                                        import fitz
                                        doc = fitz.open(stream=BytesIO(_pdf_bytes), filetype="pdf")
                                        _pages = []
                                        # Cap rasterized preview pages (0.75×); 12 pages balances completeness vs RAM
                                        _max_preview_pages = 12
                                        for _p in range(min(_max_preview_pages, len(doc))):
                                            page = doc.load_page(_p)
                                            mat = fitz.Matrix(0.75, 0.75)
                                            pix = page.get_pixmap(matrix=mat, alpha=False)
                                            _pages.append(pix.tobytes("png"))
                                            del pix
                                        st.session_state.report_preview_images = _pages
                                        st.session_state.pop("report_preview_pdf_bytes", None)
                                    except Exception:
                                        st.session_state.report_preview_images = []
                                        st.session_state.report_preview_pdf_bytes = _pdf_bytes
                                    finally:
                                        if doc is not None:
                                            try:
                                                doc.close()
                                            except Exception:
                                                pass
                                            del doc
                                        gc.collect()
                                    st.rerun()
                                else:
                                    st.session_state.report_preview_images = []
                                    st.rerun()
                    with _proceed_btn_col:
                        _profiles = _load_sorted_report_profiles()
                        _sel_idx = st.session_state.get("report_profile_selected_idx", None)
                        _template_selected = (
                            not add_new_view
                            and isinstance(_sel_idx, int)
                            and 0 <= _sel_idx < len(_profiles)
                        )
                        if _template_selected and st.button("Proceed", key="report_proceed_btn", type="primary", use_container_width=True):
                            # Ensure the selected template's settings (including throttle aggregation)
                            # are applied to the current session before entering the report generation page.
                            _selected_profile = _profiles[_sel_idx]
                            apply_profile_to_session_state(_selected_profile)
                            invalidate_report_after_data_change()
                            _refresh_report_raw_data_df()
                            ensure_summary_graphs_for_current_file()
                            ensure_report_graphs_for_current_config()
                            st.session_state["report_generation_page"] = True
                            st.rerun()
                    with _cancel_btn_col:
                        if add_new_view and st.button("Cancel", key="report_cancel_btn", use_container_width=True):
                            st.session_state.report_multi_add_new_view = False
                            st.session_state["report_show_save_profile_form"] = False
                            st.session_state["report_edit_mode"] = False
                            st.session_state["report_profile_edit_idx"] = None
                            st.rerun()
                    with _save_btn_col:
                        if add_new_view:
                            _current_plot_source = st.session_state.get("multi_param_plot_data_source", "Sorted performance table")
                            if _current_plot_source == "Raw data":
                                if st.button("Save", key="report_save_btn", use_container_width=True):
                                    st.warning("⚠️ Please switch to **Sorted performance table** mode in the Plot tab before saving. Reports require sorted data.")
                            else:
                                if st.button("Save", key="report_save_btn", use_container_width=True):
                                    st.session_state["report_show_save_profile_form"] = True
                                    st.rerun()

        # Section 7 extracted to multi_file_view.py
        elif analysis_type == "Multi-File Comparison":
            from multi_file_view import render_multi_file_comparison
            render_multi_file_comparison(render_footer)

    # Universal footer for all pages
    render_footer()

if __name__ == "__main__":
    main()

