#type: ignore
"""
Multi-File Comparison view.

Lets users compare one benchmark file against up to 7 target files.
Renders:
  - File selector (benchmark + targets)
  - Plot tab: overlaid line charts with dual Y-axis
  - Data tab: side-by-side raw data tables with summary stats

Optimised for Streamlit session-state stability:
  - Widget keys are never pre-set before the widget renders.
  - File data is cached in session state via a fingerprint key
    so files are only re-loaded when the selection actually changes.
  - All helper functions live at module level to avoid re-creation on rerun.
"""
import hashlib
import os
import re

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import (
    ensure_seconds_column, fix_duplicate_columns,
    cached_load_file, cached_load_ulog, content_hash,
    find_column_by_pattern,
)
from data_processing import (
    sanitize_table_for_streamlit,
    process_throttle_aggregation,
)
from config import COLUMN_DISPLAY_NAMES, get_display_name


# ─── Module-level helpers (created once, not per rerun) ──────────────


def _get_axis_title(col_name: str) -> str:
    """Return a formatted axis title for a column."""
    return COLUMN_DISPLAY_NAMES.get(col_name, col_name)


def _register_plot_shown():
    """Count at most one plot-view event per Streamlit run for analytics."""
    if getattr(st.session_state, '_plots_counted_this_run', False):
        return
    st.session_state._plots_counted_this_run = True
    try:
        from usage_tracking import auto_track_plot
        auto_track_plot()
    except Exception:
        pass


def _detect_throttle_columns(df):
    """Detect standard motor-test columns required for throttle aggregation.
    Returns a dict of column names (or None when not found).

    Supports wingflyingtech / VT-100KG bench column names alongside the
    existing LY-30KGF and Rotrix formats.
    """
    return {
        "throttle": find_column_by_pattern(df, [
            "Throttle - %", "Throttle Input (%)", "Throttle (%)",
            "Throttle", "throttle",
        ]),
        "current": find_column_by_pattern(df, [
            "Cur - A", "Current (A)", "Current [A]",
            "Current", "current", "Cur",
        ]),
        "voltage": find_column_by_pattern(df, [
            "Vol - V", "Voltage (V)", "Voltage [V]",
            "Voltage", "voltage", "Vol",
        ]),
        "rpm1": find_column_by_pattern(df, [
            "RPM", "RPM1 - RPM", "RPM1",
            "Motor Electrical Speed (RPM)", "Electrical Speed (RPM)",
            "Rotational Speed (RPM)", "Rotational Speed",
        ]),
        "rpm2": find_column_by_pattern(df, [
            "RPM2 - RPM", "RPM2",
            "Motor Optical Speed (RPM)", "Optical Speed (RPM)",
        ]),
        "thrust": find_column_by_pattern(df, [
            "Thrust - gf", "Thrust (gf)", "Thrust (kgf)", "Thrust [g]",
            "Thrust", "thrust",
        ]),
        "torque": find_column_by_pattern(df, [
            "Torque - N*m", "Torque (N*m)", "Torque (N·m)",
            "Torque (Nm)", "Torque (N.m)", "Torque [N·m]", "Torque [N*m]",
            "Torque", "torque",
        ]),
        "motorpower": find_column_by_pattern(df, [
            "MotorPower - W", "MotorPower",
            "Mechanical Power (W)", "Mechanical (W)",
            "Electrical Power (W)", "Electrical (W)",
            "InPower - W", "InPower", "Power",
        ]),
    }


def _build_sorted_df(df, cols, throttle_min, throttle_max, throttle_interval, ramp_mode):
    """Run throttle aggregation on *df* using detected *cols*.
    Returns the sorted DataFrame or None on failure.
    """
    required = ("throttle", "current", "voltage", "thrust", "torque")
    if any(cols.get(k) is None for k in required):
        return None
    return process_throttle_aggregation(
        df,
        cols["throttle"], cols["current"], cols["voltage"],
        cols["rpm1"], cols["rpm2"], cols["thrust"], cols["torque"],
        cols["motorpower"],
        mode=ramp_mode,
        throttle_min=throttle_min,
        throttle_max=throttle_max,
        throttle_interval=throttle_interval,
    )


def _compute_file_set_fingerprint(selected_files):
    """Compute a stable hash of the selected filename list for cache keying."""
    key = "|".join(sorted(selected_files))
    return hashlib.md5(key.encode()).hexdigest()


def _short_name(filename):
    """Build a compact legend label from a filename."""
    name = filename.replace('.csv', '').replace('.ulg', '').replace('.xlsx', '')
    voltage_match = re.search(r'@\s*(\d+V)', name, re.IGNORECASE)
    if voltage_match:
        motor_match = re.search(r'(AXI\s*\d+)', name, re.IGNORECASE)
        if motor_match:
            return f"{motor_match.group(1).strip()} @ {voltage_match.group(1)}"
    if len(name) > 18:
        name = name[:15] + "…"
    return name


def _short_header(filename):
    """Build a compact column-header label from a filename."""
    name = filename.replace('.csv', '').replace('.ulg', '').replace('.xlsx', '')
    vm = re.search(r'@\s*(\d+V)', name, re.IGNORECASE)
    if vm:
        mm = re.search(r'(AXI\s*\d+)', name, re.IGNORECASE)
        if mm:
            return f"{mm.group(1).strip()} @ {vm.group(1)}"
    return name[:18] + "…" if len(name) > 18 else name


def _col_range(col_name, active_file_data):
    """Compute global min/max for one or more columns across all files."""
    if col_name is None:
        return None, None
    if isinstance(col_name, (list, tuple, set)):
        cols = [c for c in col_name if c is not None]
    else:
        cols = [col_name]
    if not cols:
        return None, None

    mins, maxs = [], []
    for df in active_file_data.values():
        for c in cols:
            if c in df.columns:
                series = df[c].dropna()
                if not series.empty:
                    mins.append(float(series.min()))
                    maxs.append(float(series.max()))
    if mins:
        return min(mins), max(maxs)
    return None, None


def _build_trace_kwargs(plot_style, color, benchmark_color, dash="solid", is_right=False):
    """Build Scatter trace kwargs from the selected plot_style."""
    dash_style = "dot" if is_right else dash
    if plot_style == "Line":
        return dict(
            mode="lines+markers",
            line=dict(color=color, width=2.5 if color == benchmark_color else 2, dash=dash_style),
            marker=dict(size=4, color=color, line=dict(width=0.5, color="white")),
        )
    if plot_style == "Scatter":
        return dict(
            mode="markers",
            marker=dict(size=7, color=color, line=dict(width=0.5, color="white")),
            line=dict(width=0),
        )
    if plot_style == "Line + Markers":
        return dict(
            mode="lines+markers",
            line=dict(color=color, width=2, dash=dash_style),
            marker=dict(size=5, color=color, line=dict(width=0.5, color="white")),
        )
    if plot_style == "Step (horizontal)":
        return dict(
            mode="lines",
            line=dict(color=color, width=2, dash=dash_style, shape="hv"),
        )
    if plot_style == "Step (vertical)":
        return dict(
            mode="lines",
            line=dict(color=color, width=2, dash=dash_style, shape="vh"),
        )
    return dict(mode="lines", line=dict(color=color, width=2, dash=dash_style))


def _safe_selectbox_index(options, value):
    """Return the index of *value* in *options*, or 0 if not found."""
    try:
        idx = list(options).index(value)
        return min(idx, len(options) - 1)
    except (ValueError, IndexError):
        return 0


def _load_files_into_cache(selected_files):
    """Load all selected files, caching the result in session state.

    Returns (file_data, file_extensions, all_numeric_cols) or
    raises an early-return flag via the returned sentinel.
    """
    fingerprint = _compute_file_set_fingerprint(selected_files)

    # Check if the cached data is still valid
    cached_fp = st.session_state.get("_multi_file_set_fingerprint", "")
    cached_data = st.session_state.get("_multi_file_cached_file_data")
    if cached_fp == fingerprint and cached_data is not None:
        return (
            cached_data,
            st.session_state.get("_multi_file_cached_file_extensions", {}),
            st.session_state.get("_multi_file_cached_numeric_cols", set()),
            st.session_state.get("_multi_file_cached_throttle_cols", {}),
        )

    # Cache miss → load files
    file_data = {}
    file_extensions = {}
    all_numeric_cols = set()
    file_throttle_cols = {}

    for filename in selected_files:
        try:
            file_obj = next(f for f in st.session_state.uploaded_files if f.name == filename)
            file_ext = os.path.splitext(filename)[-1].lower()
            file_extensions[filename] = file_ext

            file_obj.seek(0)
            raw_bytes = file_obj.read()
            file_obj.seek(0)
            if isinstance(raw_bytes, str):
                raw_bytes = raw_bytes.encode("utf-8")
            fhash = content_hash(raw_bytes)

            if file_ext == ".ulg":
                dfs_dict, topics = cached_load_ulog(fhash, raw_bytes)
                if not dfs_dict:
                    st.warning(f"⚠️ No usable topics found in {filename}")
                    continue

                topic_keys = list(dfs_dict.keys())
                selected_topic = st.session_state.get(
                    f"multi_file_multi_param_topic_{filename}",
                    topic_keys[0] if topic_keys else None,
                )

                if selected_topic and selected_topic in dfs_dict:
                    df = dfs_dict[selected_topic].copy()
                else:
                    df = dfs_dict[topic_keys[0]].copy() if topic_keys else None

                if df is not None:
                    df = ensure_seconds_column(df)
                    file_data[filename] = df
                    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != 'timestamp_seconds']
                    all_numeric_cols.update(numeric_cols)
            else:
                result = cached_load_file(fhash, raw_bytes, filename, file_ext)
                df = result[0] if result else None
                if df is not None and not df.empty:
                    df = ensure_seconds_column(df)
                    if 'Index' not in df.columns:
                        df.insert(0, 'Index', range(1, len(df) + 1))
                    file_data[filename] = df
                    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != 'timestamp_seconds']
                    all_numeric_cols.update(numeric_cols)
                else:
                    st.warning(f"⚠️ No data found in {filename}")
        except StopIteration:
            st.warning(f"⚠️ File '{filename}' not found in uploaded files.")
            continue
        except Exception as e:
            st.error(f"Error loading file {filename}: {str(e)}")
            continue

    # Detect throttle columns per file
    for filename, df in file_data.items():
        file_throttle_cols[filename] = _detect_throttle_columns(df)

    # Save to session state cache
    st.session_state["_multi_file_set_fingerprint"] = fingerprint
    st.session_state["_multi_file_cached_file_data"] = file_data
    st.session_state["_multi_file_cached_file_extensions"] = file_extensions
    st.session_state["_multi_file_cached_numeric_cols"] = all_numeric_cols
    st.session_state["_multi_file_cached_throttle_cols"] = file_throttle_cols

    return file_data, file_extensions, all_numeric_cols, file_throttle_cols


# ─── Main render function ────────────────────────────────────────────

def render_multi_file_comparison(render_footer_fn):
    """Render the Multi-File Comparison page.

    Parameters
    ----------
    render_footer_fn : callable
        The footer rendering function from app.py (passed to avoid circular imports).
    """
    st.markdown(
        f"<p style='color: #666; font-size: 0.9rem;'>"
        f"Compare benchmark file with target files. Select one benchmark file and multiple target files."
        f"</p>",
        unsafe_allow_html=True,
    )

    # ── File selection ────────────────────────────────────────────
    available_files = [f.name for f in st.session_state.uploaded_files]

    bench_col, target_col = st.columns([1, 2])

    # ── Benchmark file selector ──
    with bench_col:
        options_bench = ["None"] + available_files
        # Determine correct initial index from persisted selection
        prev_bench = st.session_state.get("multi_file_selected_benchmark", "None")
        bench_index = _safe_selectbox_index(options_bench, prev_bench)

        benchmark_file = st.selectbox(
            "Select Benchmark File",
            options=options_bench,
            index=bench_index,
            key="multi_file_benchmark_selector",
            help="Select the benchmark/reference file to compare against."
        )
        st.session_state.multi_file_selected_benchmark = benchmark_file

    if benchmark_file == "None":
        st.info("📋 Please select a benchmark file to begin Multi-File Comparison.")
        render_footer_fn()
        return

    # ── Target files selector (exclude benchmark) ──
    target_file_options = [f for f in available_files if f != benchmark_file]
    with target_col:
        prev_targets = st.session_state.get("multi_file_selected_targets", [])
        # Filter out stale targets that are no longer available
        valid_prev_targets = [t for t in prev_targets if t in target_file_options]

        selected_target_files = st.multiselect(
            "Select Target Files (up to 5)",
            options=target_file_options,
            default=valid_prev_targets,
            key="multi_file_target_selector",
            help="Select target files to compare with the benchmark file."
        )
    st.session_state.multi_file_selected_targets = selected_target_files

    if len(selected_target_files) > 5:
        st.warning(f"⚠️ Maximum 5 target files allowed. Only the first 5 files will be used.")
        selected_target_files = selected_target_files[:5]

    if not selected_target_files or len(selected_target_files) < 1:
        st.info("📋 Please select at least one target file to compare with the benchmark.")
        render_footer_fn()
        return

    selected_files = [benchmark_file] + selected_target_files

    # ── Load all selected files (with session-state caching) ─────
    file_data, file_extensions, all_numeric_cols, file_throttle_cols = _load_files_into_cache(selected_files)

    # Guard against excessive total rows across all selected files
    MAX_TOTAL_ROWS = 1_000_000
    total_rows = sum(len(df) for df in file_data.values() if df is not None)
    if total_rows > MAX_TOTAL_ROWS:
        st.error(
            f"Total rows across files ({total_rows:,}) exceeds limit "
            f"({MAX_TOTAL_ROWS:,}). Please use smaller files or fewer files."
        )
        render_footer_fn()
        return

    if not file_data:
        st.error("❌ No files could be loaded. Please check your file selections.")
        render_footer_fn()
        return

    # ── Common numeric columns for raw data ───────────────────────
    common_numeric_cols = list(all_numeric_cols)
    for filename, df in file_data.items():
        df_numeric = {
            col
            for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and col != "timestamp_seconds"
        }
        common_numeric_cols = [col for col in common_numeric_cols if col in df_numeric]

    if not common_numeric_cols:
        st.error("❌ No common numeric columns found across selected files.")
        render_footer_fn()
        return

    # Determine a global throttle range from the first file that has one
    _global_thr_min, _global_thr_max = 0.0, 100.0
    for filename, df in file_data.items():
        tc = file_throttle_cols.get(filename, {}).get("throttle")
        if tc and tc in df.columns:
            vals = pd.to_numeric(df[tc], errors="coerce").dropna()
            if not vals.empty:
                _global_thr_min = float(vals.min())
                _global_thr_max = float(vals.max())
                break

    # ── Data mode selector (shared across Plot & Data tabs) ───────
    data_mode = st.radio(
        "Data View",
        ["Raw Data", "Sorted Data"],
        horizontal=True,
        key="multi_file_data_view_mode",
        help="Raw Data shows the original measurements. "
             "Sorted Data applies throttle aggregation to produce averaged performance tables.",
    )
    show_sorted = data_mode == "Sorted Data"

    # Throttle aggregation settings (only visible in Sorted Data mode)
    if show_sorted:
        with st.expander("⚙️ Throttle Aggregation Settings", expanded=True):
            tc1, tc2, tc3, tc4 = st.columns(4)
            with tc1:
                mf_start_throttle = st.number_input(
                    "Start Throttle (%)",
                    min_value=0.0, max_value=100.0,
                    value=float(_global_thr_min),
                    step=1.0,
                    key="multi_file_throttle_min_input",
                    help="Starting throttle percentage for the range",
                )
            with tc2:
                mf_end_throttle = st.number_input(
                    "End Throttle (%)",
                    min_value=0.0, max_value=100.0,
                    value=float(_global_thr_max),
                    step=1.0,
                    key="multi_file_throttle_max_input",
                    help="Ending throttle percentage for the range",
                )
            with tc3:
                mf_throttle_interval = st.number_input(
                    "Throttle Interval (%)",
                    min_value=0.1, max_value=10.0,
                    value=5.0,
                    step=0.5,
                    key="multi_file_throttle_interval_input",
                    help="Bin size for throttle grouping (e.g., 5% = 0-5, 5-10, etc.)",
                )
            with tc4:
                mf_ramp_mode = st.selectbox(
                    "Ramp Mode",
                    ["ramp_up", "ramp_down", "bi_directional"],
                    index=0,
                    key="multi_file_ramp_mode_select",
                    help="ramp_up: only increasing throttle, ramp_down: only decreasing, bi_directional: all data",
                )
        mf_thr_min = min(mf_start_throttle, mf_end_throttle)
        mf_thr_max = max(mf_start_throttle, mf_end_throttle)
    else:
        mf_thr_min = min(
            st.session_state.get("multi_file_throttle_min_input", _global_thr_min),
            st.session_state.get("multi_file_throttle_max_input", _global_thr_max),
        )
        mf_thr_max = max(
            st.session_state.get("multi_file_throttle_min_input", _global_thr_min),
            st.session_state.get("multi_file_throttle_max_input", _global_thr_max),
        )
        mf_throttle_interval = st.session_state.get("multi_file_throttle_interval_input", 5.0)
        mf_ramp_mode = st.session_state.get("multi_file_ramp_mode_select", "ramp_up")

    # Build sorted DataFrames when the user picks "Sorted Data"
    sorted_file_data = {}
    if show_sorted:
        for filename, df in file_data.items():
            cols_map = file_throttle_cols.get(filename, {})
            sdf = _build_sorted_df(
                df, cols_map,
                mf_thr_min, mf_thr_max,
                mf_throttle_interval, mf_ramp_mode,
            )
            if sdf is not None and not sdf.empty:
                sorted_file_data[filename] = sdf

        if not sorted_file_data:
            st.warning(
                "⚠️ Could not produce sorted data for any file. "
                "Ensure files have Throttle, Current, Voltage, Thrust, and Torque columns."
            )

    # Pick which DataFrame dict to display
    active_file_data = sorted_file_data if show_sorted else file_data

    # Numeric columns for axis selectors (respect current data mode)
    axis_numeric_cols_set = set()
    for df in active_file_data.values():
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and col != "timestamp_seconds":
                axis_numeric_cols_set.add(col)
    axis_numeric_cols = sorted(axis_numeric_cols_set)
    if not axis_numeric_cols:
        st.error("❌ No numeric columns available for the current data view.")
        render_footer_fn()
        return

    # ── Tabs: Plot & Data ─────────────────────────────────────────
    tab_plot, tab_data = st.tabs(["📊 Plot", "📋 Data"])

    # ── DATA TAB ──────────────────────────────────────────────────
    with tab_data:
        _render_data_tab(selected_files, active_file_data, show_sorted)

    # ── PLOT TAB ──────────────────────────────────────────────────
    with tab_plot:
        _render_plot_tab(
            selected_files, benchmark_file, active_file_data,
            axis_numeric_cols, render_footer_fn,
        )

    render_footer_fn()


# ─── Data tab rendering ──────────────────────────────────────────────

def _render_data_tab(selected_files, active_file_data, show_sorted):
    """Render the Data tab with side-by-side tables."""
    num_files = len(selected_files)
    if num_files == 0:
        st.warning("No files selected for comparison.")
        return

    if not active_file_data:
        st.warning("No data available to display.")
        return

    all_available_cols = list(set(
        col for df in active_file_data.values()
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) or col == "Index" or "timestamp" in col.lower()
    ))
    if "Index" not in all_available_cols:
        all_available_cols = ["Index"] + all_available_cols

    enable_column_selection = st.checkbox(
        "Select columns to display",
        value=False,
        key="multi_file_enable_col_selection",
        help="Check this to select specific columns. By default, all columns are displayed.",
    )

    if enable_column_selection:
        default_selected = ["Index"] + (all_available_cols[1:16] if len(all_available_cols) > 16 else all_available_cols[1:])
        selected_cols = st.multiselect(
            "Columns",
            all_available_cols,
            default=default_selected,
            key="multi_file_data_column_selector",
            help="Select columns to display in the data tables",
        )
        if not selected_cols:
            selected_cols = ["Index"] + all_available_cols[1:16]
        selected_cols = list(dict.fromkeys(selected_cols))
        if "Index" in selected_cols:
            selected_cols.remove("Index")
        selected_cols = ["Index"] + selected_cols
    else:
        selected_cols = all_available_cols.copy()
        if "Index" in selected_cols:
            selected_cols.remove("Index")
        selected_cols = ["Index"] + selected_cols

    cols_per_row = min(num_files, 4)
    num_rows_display = (num_files + cols_per_row - 1) // cols_per_row

    for row in range(num_rows_display):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            file_idx = row * cols_per_row + col_idx
            if file_idx < num_files:
                filename = selected_files[file_idx]
                df = active_file_data.get(filename)

                with cols[col_idx]:
                    st.markdown(f"<h4 style='font-size: 16px;'>{filename}</h4>", unsafe_allow_html=True)
                    if df is not None and not df.empty:
                        df_display = fix_duplicate_columns(df)
                        if "Index" not in df_display.columns:
                            df_display.insert(0, "Index", range(1, len(df_display) + 1))

                        display_cols = [c for c in selected_cols if c in df_display.columns]
                        display_cols = list(dict.fromkeys(display_cols))

                        table_df = df_display[display_cols].rename(columns=COLUMN_DISPLAY_NAMES)
                        table_df = sanitize_table_for_streamlit(table_df)
                        html_table = table_df.to_html(index=False, border=0, classes="data-table")
                        st.markdown(
                            f'<div style="max-height: 400px; overflow-y: auto; overflow-x: auto; '
                            f'border: 1px solid #dee2e6; border-radius: 5px; padding: 8px;">'
                            f'{html_table}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        if show_sorted:
                            st.warning(f"Sorted data not available for {filename}.")
                        else:
                            st.warning(f"No data loaded for {filename}.")


# ─── Per-column range helper ─────────────────────────────────────────

def _single_col_range(col_name, active_file_data):
    """Compute global min/max for a *single* column across all files."""
    if col_name is None:
        return None, None
    mins, maxs = [], []
    for df in active_file_data.values():
        if col_name in df.columns:
            series = df[col_name].dropna()
            if not series.empty:
                mins.append(float(series.min()))
                maxs.append(float(series.max()))
    if mins:
        raw_min, raw_max = min(mins), max(maxs)
        y_min = 0.0 if raw_min > 0 else raw_min
        y_max = raw_max + max(raw_max * 0.05, raw_max * 0.02) if raw_max > 0 else 1.0
        return y_min, y_max
    return None, None


# ─── Plot tab rendering ──────────────────────────────────────────────

_MAX_Y_PARAMS = 3  # Maximum parameters per Y-axis side

def _render_plot_tab(selected_files, benchmark_file, active_file_data, axis_numeric_cols, render_footer_fn):
    """Render the Plot tab with professional multi-axis charts and comparison table."""
    # Narrow parameters column, wide plot column
    param_col, plot_col = st.columns([0.22, 0.78])

    with param_col:
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
            <span style='font-size: 1.2rem;'>🧮</span>
            <span style='font-size: 1.1rem; font-weight: 600;'>Parameters</span>
        </div>
        """, unsafe_allow_html=True)

        # ── X-axis selection ──
        # Validate stored widget value against current options
        _x_key = "multi_file_x_axis_selector"
        if _x_key in st.session_state:
            if st.session_state[_x_key] not in axis_numeric_cols:
                st.session_state[_x_key] = axis_numeric_cols[0]

        x_axis = st.selectbox(
            "X-Axis",
            axis_numeric_cols,
            key=_x_key,
        )

        # ── Left & Right Y-axis (mutually exclusive option lists) ──
        y_candidates = [c for c in axis_numeric_cols if c != x_axis]
        if not y_candidates:
            st.error("Not enough common numeric columns to create Y-axis parameters.")
            left_y_axes = []
            right_y_axes = []
        else:
            _left_key = "multi_file_left_y_axes_selector"
            _right_key = "multi_file_right_y_axes_selector"

            _is_first_left = _left_key not in st.session_state
            _is_first_right = _right_key not in st.session_state

            # Read stored selections from the previous rerun
            _stored_left = list(st.session_state.get(_left_key, []))
            _stored_right = list(st.session_state.get(_right_key, []))

            # Purge values no longer in y_candidates (Raw↔Sorted, x-axis change)
            _stored_left = [c for c in _stored_left if c in y_candidates]
            _stored_right = [c for c in _stored_right if c in y_candidates]

            # Resolve any residual overlap (left side always wins)
            _dup = set(_stored_left) & set(_stored_right)
            if _dup:
                _stored_right = [c for c in _stored_right if c not in _dup]

            # Each side hides the other side's current selections
            left_options = [c for c in y_candidates if c not in _stored_right]
            right_options = [c for c in y_candidates if c not in _stored_left]

            # Write cleaned values back so widgets never encounter stale keys
            if _is_first_left:
                st.session_state[_left_key] = _stored_left or left_options[:1]
            else:
                st.session_state[_left_key] = [
                    c for c in _stored_left if c in left_options
                ] or left_options[:1]

            if _is_first_right:
                st.session_state[_right_key] = _stored_right
            else:
                st.session_state[_right_key] = [
                    c for c in _stored_right if c in right_options
                ]

            left_y_axes = st.multiselect(
                f"Left Y-Axis parameters (max {_MAX_Y_PARAMS})",
                options=left_options,
                key=_left_key,
                help=f"Select up to {_MAX_Y_PARAMS} parameters for the left Y-axis. "
                     f"Selections are hidden from the right axis.",
            )
            if len(left_y_axes) > _MAX_Y_PARAMS:
                left_y_axes = left_y_axes[:_MAX_Y_PARAMS]
                st.warning(f"Only the first {_MAX_Y_PARAMS} left-axis parameters are used.")
            if not left_y_axes:
                left_y_axes = left_options[:1] if left_options else y_candidates[:1]

            right_y_axes = st.multiselect(
                f"Right Y-Axis parameters (optional, max {_MAX_Y_PARAMS})",
                options=right_options,
                key=_right_key,
                help=f"Select up to {_MAX_Y_PARAMS} parameters for the right Y-axis. "
                     f"Selections are hidden from the left axis.",
            )
            if len(right_y_axes) > _MAX_Y_PARAMS:
                right_y_axes = right_y_axes[:_MAX_Y_PARAMS]
                st.warning(f"Only the first {_MAX_Y_PARAMS} right-axis parameters are used.")

            plot_style = st.selectbox(
                "Plot style",
                ["Line", "Scatter", "Line + Markers", "Step (horizontal)", "Step (vertical)"],
                index=0,
                key="multi_file_plot_style",
                help="Line: smooth lines; Scatter: points only; Line+Markers: both; Step: step lines.",
            )

    if not y_candidates:
        return

    left_params = left_y_axes[:_MAX_Y_PARAMS] if left_y_axes else []
    right_params = right_y_axes[:_MAX_Y_PARAMS] if right_y_axes else []
    has_right = len(right_params) > 0
    multi_axis_mode = len(left_params) > 1 or len(right_params) > 1

    # ── Per-param range computation ──
    left_ranges = {}
    for col in left_params:
        y_min, y_max = _single_col_range(col, active_file_data)
        if y_min is not None:
            left_ranges[col] = (y_min, y_max)
        else:
            left_ranges[col] = (0.0, 1.0)

    right_ranges = {}
    for col in right_params:
        y_min, y_max = _single_col_range(col, active_file_data)
        if y_min is not None:
            right_ranges[col] = (y_min, y_max)
        else:
            right_ranges[col] = (0.0, 1.0)

    # ── Plot column ──
    with plot_col:
        filtered_data = {}
        for filename, df in active_file_data.items():
            if x_axis in df.columns and not df.empty:
                filtered_data[filename] = df

        if not filtered_data:
            st.warning("No data available to display.")
        else:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            benchmark_color = '#1f77b4'
            target_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

            # Per-param colour palette (for colour map used in top labels)
            all_param_colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            color_map = {}
            color_idx = 0

            benchmark_data = None
            target_data_list = []

            for filename, df_filtered in filtered_data.items():
                if filename == benchmark_file:
                    benchmark_data = (filename, df_filtered)
                else:
                    target_data_list.append((filename, df_filtered))

            # ── Assign yaxis mapping for each param ──
            # left_params[0] → yaxis (primary left)  |  secondary_y=False
            # left_params[1] → yaxis3  (overlaid)     |  yaxis='y3'
            # left_params[2] → yaxis5  (overlaid)     |  yaxis='y5'
            # right_params[0] → yaxis2 (primary right) |  secondary_y=True
            # right_params[1] → yaxis4  (overlaid)     |  yaxis='y4'
            # right_params[2] → yaxis6  (overlaid)     |  yaxis='y6'
            left_yaxis_map = {}
            for i, col in enumerate(left_params):
                if i == 0:
                    left_yaxis_map[col] = None  # Use secondary_y=False (yaxis)
                else:
                    left_yaxis_map[col] = f"y{2*i + 1}"  # y3, y5

            right_yaxis_map = {}
            for i, col in enumerate(right_params):
                if i == 0:
                    right_yaxis_map[col] = None  # Use secondary_y=True (yaxis2)
                else:
                    right_yaxis_map[col] = f"y{2*i + 2}"  # y4, y6

            # ── Build color map (per-param colors for top labels) ──
            for col in left_params:
                color_map[col] = all_param_colors[color_idx % len(all_param_colors)]
                color_idx += 1
            for col in right_params:
                color_map[col] = all_param_colors[color_idx % len(all_param_colors)]
                color_idx += 1

            # ────────────────────────────────────────────────
            # Add traces
            # ────────────────────────────────────────────────

            def _add_traces_for_file(filename, df_filtered, file_color, is_benchmark, file_idx=0):
                """Add left + right param traces for one file."""
                short = _short_name(filename)
                prefix = "B" if is_benchmark else f"T{file_idx+1}"
                line_styles = ['solid', 'dash', 'dot', 'dashdot']
                base_style_idx = 0 if is_benchmark else ((file_idx // len(target_colors)) % len(line_styles))

                # Left-axis parameters
                for p_idx, col in enumerate(left_params):
                    if col not in df_filtered.columns:
                        continue
                    plot_data = df_filtered[[x_axis, col]].dropna()
                    if plot_data.empty:
                        continue
                    style = line_styles[(base_style_idx + p_idx) % len(line_styles)]
                    kw = _build_trace_kwargs(plot_style, file_color, benchmark_color, dash=style, is_right=False)

                    yaxis_ref = left_yaxis_map[col]
                    if yaxis_ref is None:
                        # Primary left (yaxis) – use secondary_y=False
                        fig.add_trace(
                            go.Scatter(
                                x=plot_data[x_axis], y=plot_data[col],
                                name=f"{prefix}: {short} — {get_display_name(col)}",
                                **kw,
                            ),
                            secondary_y=False,
                        )
                    else:
                        # Additional left axis (y3, y5)
                        fig.add_trace(
                            go.Scatter(
                                x=plot_data[x_axis], y=plot_data[col],
                                name=f"{prefix}: {short} — {get_display_name(col)}",
                                yaxis=yaxis_ref,
                                **kw,
                            ),
                        )

                # Right-axis parameters
                for p_idx, col in enumerate(right_params):
                    if col not in df_filtered.columns:
                        continue
                    plot_data = df_filtered[[x_axis, col]].dropna()
                    if plot_data.empty:
                        continue
                    style = line_styles[(base_style_idx + p_idx + 1) % len(line_styles)]
                    kw = _build_trace_kwargs(plot_style, file_color, benchmark_color, dash=style, is_right=True)

                    yaxis_ref = right_yaxis_map[col]
                    if yaxis_ref is None:
                        # Primary right (yaxis2) – use secondary_y=True
                        fig.add_trace(
                            go.Scatter(
                                x=plot_data[x_axis], y=plot_data[col],
                                name=f"{prefix}: {short} — {get_display_name(col)}",
                                **kw,
                            ),
                            secondary_y=True,
                        )
                    else:
                        # Additional right axis (y4, y6)
                        fig.add_trace(
                            go.Scatter(
                                x=plot_data[x_axis], y=plot_data[col],
                                name=f"{prefix}: {short} — {get_display_name(col)}",
                                yaxis=yaxis_ref,
                                **kw,
                            ),
                        )

            # Benchmark traces
            if benchmark_data:
                _add_traces_for_file(
                    benchmark_data[0], benchmark_data[1],
                    benchmark_color, is_benchmark=True,
                )

            # Target file traces
            for idx, (filename, df_filtered) in enumerate(target_data_list):
                file_color = target_colors[idx % len(target_colors)]
                _add_traces_for_file(
                    filename, df_filtered,
                    file_color, is_benchmark=False, file_idx=idx,
                )

            # ────────────────────────────────────────────────
            # Build axis layout (multi-axis support)
            # ────────────────────────────────────────────────
            axis_layout = {}

            # Reserve horizontal space for additional axis tick labels.
            # Each extra left/right axis uses 10% of the paper width.
            n_extra_left = max(0, len(left_params) - 1)   # 0, 1, or 2
            n_extra_right = max(0, len(right_params) - 1) # 0, 1, or 2
            _AXIS_GAP = 0.05  # 5% paper width per extra axis
            domain_left = _AXIS_GAP * n_extra_left      # 0, 0.10, or 0.20
            domain_right = 1.0 - _AXIS_GAP * n_extra_right  # 1.0, 0.90, or 0.80

            # X-axis with constrained domain
            fig.update_xaxes(
                title_text=_get_axis_title(x_axis),
                domain=[domain_left, domain_right],
                showgrid=True,
                gridcolor="#e0e0e0",
                gridwidth=1,
                nticks=12,
                showline=True,
                linecolor="#222",
                linewidth=2,
                title_font=dict(size=18, color="#111"),
                tickfont=dict(size=16, color="#333"),
            )

            # ── Left-side axes ──
            # automargin=True lets Plotly auto-adjust spacing for tick labels
            # and titles across different display resolutions.

            # Primary left axis (yaxis)
            if left_params and left_params[0] in left_ranges:
                y_min, y_max = left_ranges[left_params[0]]
                axis_layout["yaxis"] = dict(
                    title="" if multi_axis_mode else get_display_name(left_params[0]),
                    range=[y_min, y_max],
                    showgrid=True,
                    gridcolor="#e0e0e0",
                    gridwidth=1,
                    showline=True,
                    linecolor="#222",
                    linewidth=2,
                    zeroline=True,
                    zerolinecolor="#ccc",
                    zerolinewidth=1,
                    title_font=dict(size=17, color="#111"),
                    tickfont=dict(size=15, color="#333"),
                    side="left",
                    automargin=True,
                )

            # Second left axis (yaxis3)
            if len(left_params) > 1 and left_params[1] in left_ranges:
                y_min, y_max = left_ranges[left_params[1]]
                axis_layout["yaxis3"] = dict(
                    title="",
                    title_font=dict(size=16, color="#111"),
                    tickfont=dict(size=14, color="#333"),
                    anchor="free",
                    overlaying="y",
                    side="left",
                    position=0.0 if n_extra_left == 1 else domain_left / 2,
                    range=[y_min, y_max],
                    showgrid=False,
                    showline=True,
                    linecolor="#333",
                    linewidth=1.5,
                    showticklabels=True,
                    automargin=True,
                )

            # Third left axis (yaxis5)
            if len(left_params) > 2 and left_params[2] in left_ranges:
                y_min, y_max = left_ranges[left_params[2]]
                axis_layout["yaxis5"] = dict(
                    title="",
                    title_font=dict(size=16, color="#111"),
                    tickfont=dict(size=14, color="#333"),
                    anchor="free",
                    overlaying="y",
                    side="left",
                    position=0.0,
                    range=[y_min, y_max],
                    showgrid=False,
                    showline=True,
                    linecolor="#444",
                    linewidth=1.5,
                    showticklabels=True,
                    automargin=True,
                )

            # ── Right-side axes ──

            # Primary right axis (yaxis2) — only if right params exist
            if has_right and right_params[0] in right_ranges:
                y_min, y_max = right_ranges[right_params[0]]
                axis_layout["yaxis2"] = dict(
                    title="" if multi_axis_mode else get_display_name(right_params[0]),
                    range=[y_min, y_max],
                    showgrid=False,
                    showline=True,
                    linecolor="#222",
                    linewidth=2,
                    zeroline=False,
                    title_font=dict(size=17, color="#111"),
                    tickfont=dict(size=15, color="#333"),
                    overlaying="y",
                    side="right",
                    automargin=True,
                )
            elif not has_right:
                # Explicitly hide the secondary (right) Y-axis when no right params
                axis_layout["yaxis2"] = dict(
                    visible=False,
                    showticklabels=False,
                    showline=False,
                    showgrid=False,
                )

            # Second right axis (yaxis4)
            if len(right_params) > 1 and right_params[1] in right_ranges:
                y_min, y_max = right_ranges[right_params[1]]
                axis_layout["yaxis4"] = dict(
                    title="",
                    title_font=dict(size=16, color="#111"),
                    tickfont=dict(size=14, color="#333"),
                    anchor="free",
                    overlaying="y",
                    side="right",
                    position=1.0 if n_extra_right == 1 else domain_right + (1.0 - domain_right) / 2,
                    range=[y_min, y_max],
                    showgrid=False,
                    showline=True,
                    linecolor="#333",
                    linewidth=1.5,
                    showticklabels=True,
                    automargin=True,
                )

            # Third right axis (yaxis6)
            if len(right_params) > 2 and right_params[2] in right_ranges:
                y_min, y_max = right_ranges[right_params[2]]
                axis_layout["yaxis6"] = dict(
                    title="",
                    title_font=dict(size=16, color="#111"),
                    tickfont=dict(size=14, color="#333"),
                    anchor="free",
                    overlaying="y",
                    side="right",
                    position=1.0,
                    range=[y_min, y_max],
                    showgrid=False,
                    showline=True,
                    linecolor="#444",
                    linewidth=1.5,
                    showticklabels=True,
                    automargin=True,
                )

            if axis_layout:
                fig.update_layout(**axis_layout)

            # ── Title (reflects all selected params, not just the first) ──
            left_names = [get_display_name(c) for c in left_params]
            right_names = [get_display_name(c) for c in right_params]
            if right_names:
                title_text = f"{' · '.join(left_names)} & {' · '.join(right_names)} Vs {_get_axis_title(x_axis)}"
            else:
                title_text = f"{' · '.join(left_names)} Vs {_get_axis_title(x_axis)}"

            # Count total legend entries to size the bottom space.
            # Aim for ~3 items per row with a compact but readable font.
            n_traces = len(fig.data)
            legend_rows = max(1, (n_traces + 2) // 3)
            legend_bottom_margin = 80 + legend_rows * 20

            fig.update_layout(
                title=dict(
                    text=f"<b>{title_text}</b>",
                    x=0.5,
                    xanchor="center",
                    font=dict(size=14 if multi_axis_mode else 16, color="#111"),
                ),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.18,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=11, color="#222"),
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="rgba(0,0,0,0.15)",
                    borderwidth=1,
                    tracegroupgap=4,
                    itemwidth=260,
                ),
                template="plotly_white",
                height=750,
                margin=dict(l=20, r=20, t=100 if multi_axis_mode else 70, b=legend_bottom_margin),
                hovermode="x unified",
                plot_bgcolor="#fafafa",
            )

            # ── Top colour-coded param labels aligned above each axis ──
            if multi_axis_mode:
                from config import get_short_param_label
                _LABEL_Y = 1.08
                _LABEL_SIZE = 14

                # Compute the x position of each left axis line
                # left_params[0] → primary left at domain_left edge
                # left_params[1] → yaxis3 position
                # left_params[2] → yaxis5 at 0.0
                left_label_positions = []
                for i in range(len(left_params)):
                    if i == 0:
                        left_label_positions.append(domain_left)
                    elif i == 1:
                        left_label_positions.append(
                            0.0 if n_extra_left == 1
                            else domain_left / 2
                        )
                    elif i == 2:
                        left_label_positions.append(0.0)

                for i, col in enumerate(left_params[:3]):
                    label = get_short_param_label(col)
                    if not label:
                        continue
                    fig.add_annotation(
                        xref="paper", yref="paper",
                        x=left_label_positions[i], y=_LABEL_Y,
                        text=label, showarrow=False,
                        xanchor="center", yanchor="bottom",
                        font=dict(size=_LABEL_SIZE,
                                  color=color_map.get(col, "#333")),
                    )

                # Compute the x position of each right axis line
                # right_params[0] → primary right at domain_right edge
                # right_params[1] → yaxis4 position
                # right_params[2] → yaxis6 at 1.0
                right_label_positions = []
                for i in range(len(right_params)):
                    if i == 0:
                        right_label_positions.append(domain_right)
                    elif i == 1:
                        right_label_positions.append(
                            1.0 if n_extra_right == 1
                            else domain_right + (1.0 - domain_right) / 2
                        )
                    elif i == 2:
                        right_label_positions.append(1.0)

                for i, col in enumerate(right_params[:3]):
                    label = get_short_param_label(col)
                    if not label:
                        continue
                    fig.add_annotation(
                        xref="paper", yref="paper",
                        x=right_label_positions[i], y=_LABEL_Y,
                        text=label, showarrow=False,
                        xanchor="center", yanchor="bottom",
                        font=dict(size=_LABEL_SIZE,
                                  color=color_map.get(col, "#333")),
                    )

            _register_plot_shown()
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": True, "displaylogo": False},
            )

    # ── Per-parameter comparison tables (full width below plot) ──
    all_y_params = list(left_params) + list(right_params)
    if not all_y_params:
        return

    st.markdown("---")
    st.markdown(
        "<h3 style='margin-bottom:0.2rem;'>📋 Parameter Comparison</h3>"
        "<p style='color:#666;font-size:0.9rem;margin-top:0;'>Side-by-side comparison across selected files with summary statistics</p>",
        unsafe_allow_html=True,
    )

    _TABLE_CSS = (
        "<style>"
        ".cmp-tbl{font-size:13px;border-collapse:collapse;width:100%;font-family:inherit}"
        ".cmp-tbl th{background:linear-gradient(180deg,#f8f9fb,#eef0f4);position:sticky;top:0;"
        "padding:8px 12px;text-align:center;font-weight:700;border-bottom:2px solid #c0c4cc;"
        "white-space:pre-line;font-size:13px;color:#222}"
        ".cmp-tbl td{padding:6px 12px;text-align:right;border-bottom:1px solid #e8eaed;color:#333}"
        ".cmp-tbl tr:nth-child(even) td{background:#f9fafb}"
        ".cmp-tbl tr:hover td{background:#e8f0fe}"
        ".cmp-tbl .summary-row td{background:#fff3cd!important;font-weight:600;border-top:2px solid #c0c4cc}"
        ".stat-card{background:linear-gradient(135deg,#f8f9fb,#fff);border:1px solid #dee2e6;"
        "border-radius:8px;padding:12px 16px;text-align:center}"
        ".stat-val{font-size:1.3rem;font-weight:700;color:#1a73e8}"
        ".stat-label{font-size:0.75rem;color:#666;text-transform:uppercase;letter-spacing:0.5px}"
        "</style>"
    )
    st.markdown(_TABLE_CSS, unsafe_allow_html=True)

    # Build per-parameter DataFrames + stats
    param_tables = {}
    param_stats = {}
    x_display = COLUMN_DISPLAY_NAMES.get(x_axis, x_axis)

    for y_col in all_y_params:
        param_df = None
        file_stats = {}

        for filename in selected_files:
            df = active_file_data.get(filename)
            if df is None or df.empty or x_axis not in df.columns or y_col not in df.columns:
                continue

            part = df[[x_axis, y_col]].copy().dropna(subset=[x_axis, y_col])
            if part.empty:
                continue

            part = part.groupby(x_axis, as_index=False)[y_col].mean()
            short = _short_header(filename)
            part = part.rename(columns={y_col: short, x_axis: x_display})

            # Compute stats for this file
            vals = part[short].dropna()
            if len(vals) > 0:
                file_stats[short] = {
                    "Min": round(float(vals.min()), 2),
                    "Max": round(float(vals.max()), 2),
                    "Mean": round(float(vals.mean()), 2),
                    "Std Dev": round(float(vals.std()), 2) if len(vals) > 1 else 0.0,
                }

            if param_df is None:
                param_df = part
            else:
                param_df = pd.merge(param_df, part, on=x_display, how="outer")

        if param_df is not None and not param_df.empty:
            param_df = param_df.sort_values(by=x_display)
            for col in param_df.columns:
                if pd.api.types.is_numeric_dtype(param_df[col]):
                    param_df[col] = param_df[col].round(4)

            # Add Δ% column when exactly 2 files
            file_cols = [c for c in param_df.columns if c != x_display]
            if len(file_cols) == 2:
                a_col, b_col = file_cols
                a_vals = pd.to_numeric(param_df[a_col], errors="coerce")
                b_vals = pd.to_numeric(param_df[b_col], errors="coerce")
                denom = a_vals.replace(0, np.nan)
                delta_pct = ((b_vals - a_vals) / denom.abs() * 100).round(2)
                param_df["Δ %"] = delta_pct

            param_tables[y_col] = param_df
            param_stats[y_col] = file_stats

    if not param_tables:
        st.info("No data available to display.")
        return

    # Download All as Excel (multiple sheets)
    try:
        import io
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            for y_col, tbl in param_tables.items():
                sheet = get_display_name(y_col)[:31]
                tbl.to_excel(writer, sheet_name=sheet, index=False)
            # Summary stats sheet
            if param_stats:
                summary_rows = []
                for y_col, fstats in param_stats.items():
                    for fname, stats in fstats.items():
                        row = {"Parameter": get_display_name(y_col), "File": fname}
                        row.update(stats)
                        summary_rows.append(row)
                if summary_rows:
                    pd.DataFrame(summary_rows).to_excel(
                        writer, sheet_name="Summary Statistics", index=False
                    )
        excel_buf.seek(0)
        _dl_spacer, _dl_btn = st.columns([0.75, 0.25])
        with _dl_btn:
            st.download_button(
                label="📥 Download All Parameters (Excel)",
                data=excel_buf,
                file_name="parameter_comparison.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    except Exception:
        pass

    # Render each parameter table
    for y_col, tbl in param_tables.items():
        display_y = get_display_name(y_col)
        fstats = param_stats.get(y_col, {})

        with st.expander(f"📊 **{display_y}**", expanded=True):
            # ── Summary stat cards ──
            if fstats:
                stat_cols = st.columns(len(fstats))
                for idx, (fname, stats) in enumerate(fstats.items()):
                    with stat_cols[idx]:
                        st.markdown(
                            f"<div class='stat-card'>"
                            f"<div style='font-weight:600;font-size:0.85rem;color:#444;margin-bottom:6px'>{fname}</div>"
                            f"<div style='display:flex;justify-content:space-around;gap:4px'>"
                            f"<div><div class='stat-val' style='color:#1a73e8'>{stats['Mean']}</div>"
                            f"<div class='stat-label'>Mean</div></div>"
                            f"<div><div class='stat-val' style='color:#34a853'>{stats['Max']}</div>"
                            f"<div class='stat-label'>Max</div></div>"
                            f"<div><div class='stat-val' style='color:#ea4335'>{stats['Min']}</div>"
                            f"<div class='stat-label'>Min</div></div>"
                            f"</div></div>",
                            unsafe_allow_html=True,
                        )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # ── CSV download (right-aligned) ──
            csv_data = tbl.to_csv(index=False).encode("utf-8")
            _csv_spacer, _csv_btn = st.columns([0.75, 0.25])
            with _csv_btn:
                st.download_button(
                    label=f"📥 Download {display_y} (CSV)",
                    data=csv_data,
                    file_name=f"{display_y.replace(' ', '_')}_comparison.csv",
                    mime="text/csv",
                    key=f"dl_{y_col}",
                )

            # ── Data table ──
            display_df = tbl.fillna("–")
            display_df = sanitize_table_for_streamlit(display_df)
            html_table = display_df.to_html(index=False, border=0, classes="cmp-tbl")
            st.markdown(
                f'<div style="max-height:380px;overflow:auto;'
                f'border:1px solid #dee2e6;border-radius:8px;padding:0;'
                f'box-shadow:0 1px 3px rgba(0,0,0,0.08);">'
                f'{html_table}</div>',
                unsafe_allow_html=True,
            )

