#type: ignore
"""
Report graph storage, auto-graph generation for reports, and Plotly
figure-to-image export helpers.
"""
import sys
import hashlib
import base64
import threading

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import get_display_name, get_axis_title
from data_processing import smooth_data_for_plotting, add_top_param_labels


# ---------------------------------------------------------------------------
# Graph storage for reports
# ---------------------------------------------------------------------------

def store_graph_for_report(graph_name: str, fig, table=None, heading: str | None = None):
    """
    Idempotently store a Plotly figure (and optional table) for report generation.
    """
    try:
        if 'report_graph_entries' not in st.session_state:
            st.session_state.report_graph_entries = {}

        existing = st.session_state.report_graph_entries.get(graph_name, {})

        final_fig = fig if fig is not None else existing.get("fig")
        final_table = table if table is not None else existing.get("table")
        final_heading = heading if heading is not None else existing.get("heading")

        if final_fig is None and final_table is None:
            return False

        st.session_state.report_graph_entries[graph_name] = {
            "fig": final_fig,
            "table": final_table,
            "heading": final_heading,
        }
        return True
    except Exception:
        return False


def invalidate_report_after_data_change():
    """
    Clear report graph entries, preview cache, and summary figures.
    """
    if "report_graph_entries" in st.session_state:
        st.session_state.report_graph_entries = {}
    for key in (
        "report_preview_pdf_bytes",
        "report_preview_file",
        "report_preview_signature",
        "report_download_zip_bytes",
        "report_download_filename",
    ):
        st.session_state.pop(key, None)
    for key in ("report_throttle_line_fig", "report_throttle_bar_fig",
                "report_throttle_area_fig", "report_throttle_area_is_vibration"):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Auto-generation of performance graphs for the report
# ---------------------------------------------------------------------------

def ensure_report_performance_graphs_for_current_file():
    """
    Auto-build Graph 1–5 for the currently selected file when the user
    has not opened the Plot tab.
    """
    try:
        existing_entries = getattr(st.session_state, "report_graph_entries", {}) or {}
        if existing_entries:
            return False

        # Report always uses sorted performance table when available.
        df_sorted = getattr(st.session_state, "report_sorted_table_df", None)
        has_sorted = df_sorted is not None and not getattr(df_sorted, "empty", True)
        if has_sorted:
            df = df_sorted
            plot_data_source = "Sorted performance table"
        else:
            _raw = st.session_state.get("multi_param_raw_df")
            if _raw is not None and not getattr(_raw, "empty", True):
                df = _raw
            else:
                df = getattr(st.session_state, "report_raw_data_df", None)
            plot_data_source = "Raw data"

        if df is None or df.empty:
            return False

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            return False

        def _find_col(cols, targets):
            names = [targets] if isinstance(targets, str) else targets
            for name in names:
                low = str(name).lower()
                for col in cols:
                    if str(col).lower() == low:
                        return col
                for col in cols:
                    if low in str(col).lower():
                        return col
            return None

        graphs_built = 0

        def _safe_add_graph(graph_idx: int, title: str, x_col: str | None, left_cols, right_cols):
            nonlocal graphs_built
            if not x_col:
                return
            left_cols = [c for c in (left_cols or []) if c and c in df.columns]
            right_cols = [c for c in (right_cols or []) if c and c in df.columns]
            if not left_cols and not right_cols:
                return

            has_right = bool(right_cols)
            fig = make_subplots(specs=[[{"secondary_y": has_right}]])

            for col in left_cols:
                sub = df[[x_col, col]].dropna(subset=[x_col, col])
                if sub.empty:
                    continue
                fig.add_trace(
                    go.Scatter(x=sub[x_col], y=sub[col], mode="lines+markers", name=str(col)),
                    secondary_y=False,
                )

            for col in right_cols:
                sub = df[[x_col, col]].dropna(subset=[x_col, col])
                if sub.empty:
                    continue
                fig.add_trace(
                    go.Scatter(x=sub[x_col], y=sub[col], mode="lines", name=str(col)),
                    secondary_y=True,
                )

            if not fig.data:
                return

            # Decide layout mode for auto graphs:
            # - multi_axis_mode: more than one parameter on either side
            # - dual_axis: at least one parameter on both left and right
            multi_axis_mode = (
                (left_cols and len(left_cols) > 1)
                or (right_cols and len(right_cols) > 1)
            )
            dual_axis = bool(left_cols and right_cols)

            fig.update_layout(
                template="plotly_white",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            )
            fig.update_xaxes(title=x_col)

            # Side axis titles:
            # - When each side has at most 1 param, show side titles.
            # - When any side has 2+ params (multi_axis_mode), remove side
            #   titles and rely on top labels + legend.
            left_title = ", ".join(left_cols) or None
            right_title = ", ".join(right_cols) or None

            if multi_axis_mode:
                fig.update_yaxes(title="", secondary_y=False)
                if has_right:
                    _has_accz = any("accz" in str(c).lower() for c in right_cols)
                    if _has_accz:
                        fig.update_yaxes(title="", secondary_y=True)
                    else:
                        fig.update_yaxes(title="", secondary_y=True, rangemode="tozero")
            else:
                fig.update_yaxes(title=left_title, secondary_y=False)
                if has_right:
                    _has_accz = any("accz" in str(c).lower() for c in right_cols)
                    if _has_accz:
                        fig.update_yaxes(title=right_title, secondary_y=True)
                    else:
                        fig.update_yaxes(title=right_title, secondary_y=True, rangemode="tozero")

            # Mark layout meta for export helpers (dual‑axis / multi‑Y)
            try:
                current_meta = fig.layout.meta or {}
                if not isinstance(current_meta, dict):
                    current_meta = {}
                current_meta.update(
                    {
                        # Mark multi_y_params when any side has 2+ params so
                        # export helpers adjust margins/titles accordingly.
                        "multi_y_params": bool(multi_axis_mode),
                        "multi_y_left_count": len(left_cols or []),
                        "multi_y_right_count": len(right_cols or []),
                    }
                )
                fig.update_layout(meta=current_meta)
            except Exception:
                pass

            # Add top labels for dual‑axis / multi‑parameter graphs
            try:
                add_top_param_labels(fig, left_cols, right_cols, color_map=None)
            except Exception:
                pass

            store_graph_for_report(f"Graph {graph_idx}", fig, table=None, heading=title)
            graphs_built += 1

        # Graph 1: Speed-Based Performance Trends
        rpm_col = _find_col(
            numeric_cols,
            ["Motor Electrical Speed (RPM)", "Motor Optical Speed (RPM)", "RPM1 - RPM", "RPM",
             "Rotational Speed (RPM)", "Rotational Speed"],
        )
        thrust_col = _find_col(numeric_cols, ["Thrust (gf)", "Thrust - gf", "Thrust (kgf)", "Thrust"])
        eff_col = _find_col(
            numeric_cols,
            [
                "Overall Efficiency (gf/W)",
                "SysEffect (gf/W)",
                "SysEffect - gf/W",
                "SysEffect",
                "Motor Efficiency (%)",
                "Motor Efficiency",
                "efficiency",
                "Efficiency",
            ],
        )
        g1_title = "Speed-Based Performance Trends"
        # If we cannot locate the typical X/Y parameters at all, fall back to a
        # neutral heading so the report does not imply specific metrics.
        if rpm_col is None or thrust_col is None or eff_col is None:
            g1_title = "Graph 1"
        _safe_add_graph(
            1,
            g1_title,
            rpm_col or (numeric_cols[0] if numeric_cols else None),
            [thrust_col],
            [eff_col] if eff_col else [],
        )

        # Graph 2: Load-Based Performance Trends
        torque_col = _find_col(
            numeric_cols,
            ["Torque (N·m)", "Torque (N*m)", "Torque (Nm)", "Torque (N.m)",
             "|Torque| (N·m)", "Torque - N*m", "Torque", "torque"],
        )
        g2_title = "Load-Based Performance Trends"
        if torque_col is None or thrust_col is None or eff_col is None:
            g2_title = "Graph 2"
        _safe_add_graph(
            2,
            g2_title,
            torque_col or rpm_col,
            [thrust_col],
            [eff_col] if eff_col else [],
        )

        # Graph 3: Power and Efficiency Trends
        power_col = _find_col(
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
        motor_eff_col = _find_col(
            numeric_cols,
            ["Motor Efficiency (%)", "Motor Efficiency", "MotorRate - %"],
        )
        g3_title = "Power and Efficiency Trends"
        if power_col is None or motor_eff_col is None or thrust_col is None:
            g3_title = "Graph 3"
        _safe_add_graph(
            3,
            g3_title,
            thrust_col or rpm_col,
            [power_col] if power_col else [],
            [motor_eff_col] if motor_eff_col else [],
        )

        # Graph 4: Thrust and Vibration Response
        time_col = _find_col(numeric_cols, ["Time (s)", "Time", "timestamp_seconds", "Time (secs)"])
        vibration_col = _find_col(numeric_cols, ["Vibration RMS (g)", "Vibration (g)", "Vibration - g", "Vibration"])
        left_g4 = [c for c in [thrust_col, eff_col] if c]
        right_g4 = [vibration_col] if vibration_col else []
        g4_title = None
        if time_col and (left_g4 or right_g4):
            g4_title = "Thrust and Vibration Response" if vibration_col else "Graph 4"
            _safe_add_graph(4, g4_title, time_col, left_g4, right_g4)

        # Graph 5: Acceleration Response
        accx = _find_col(numeric_cols, ["AccX (g)", "AccX", "accx"])
        accy = _find_col(numeric_cols, ["AccY (g)", "AccY", "accy"])
        accz = _find_col(numeric_cols, ["AccZ (g)", "AccZ corrected", "AccZ", "accz"])
        left_g5 = [c for c in [accx, accy] if c]
        right_g5 = [accz] if accz else []
        g5_title = None
        if time_col and (left_g5 or right_g5):
            g5_title = "Acceleration Response"
            if not left_g5 and not right_g5:
                g5_title = "Graph 5"
            _safe_add_graph(5, g5_title, time_col, left_g5, right_g5)

        # #region agent log
        try:
            import json as _json
            import time as _time
            import os as _os
            _log_path = _os.path.join(_os.path.dirname(__file__), "..", "debug-4d38f6.log")
            with open(_log_path, "a", encoding="utf-8") as _f:
                _f.write(
                    _json.dumps(
                        {
                            "sessionId": "4d38f6",
                            "id": f"log_{int(_time.time()*1000)}_ensure_report_performance_graphs_for_current_file",
                            "timestamp": int(_time.time() * 1000),
                            "location": "plotting.py:ensure_report_performance_graphs_for_current_file",
                            "message": "auto_performance_graphs_state",
                            "data": {
                                "plot_data_source": plot_data_source,
                                "graphs_built": graphs_built,
                                "rpm_col": rpm_col,
                                "thrust_col": thrust_col,
                                "eff_col": eff_col,
                                "torque_col": torque_col,
                                "power_col": power_col,
                                "motor_eff_col": motor_eff_col,
                                "time_col": time_col,
                                "vibration_col": vibration_col,
                                "accx": accx,
                                "accy": accy,
                                "accz": accz,
                                "g1_title": g1_title,
                                "g2_title": g2_title,
                                "g3_title": g3_title,
                                "g4_title": g4_title,
                                "g5_title": g5_title,
                            },
                            "runId": "run1",
                            "hypothesisId": "H2",
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion

        if graphs_built > 0:
            st.session_state.report_graphs_data_source = plot_data_source
        return graphs_built > 0
    except Exception:
        return False


def _get_report_dataframe_for_current_source():
    """
    Resolve the dataframe used for report graphs and PDF. Report generation and
    preview always use the sorted performance table when available; raw data
    is only used when no sorted table exists.
    """
    df_sorted = getattr(st.session_state, "report_sorted_table_df", None)
    has_sorted = df_sorted is not None and not getattr(df_sorted, "empty", True)

    if has_sorted:
        return df_sorted, "Sorted performance table"
    # If there is no sorted performance table, do not fall back to raw for
    # report graphs/preview; callers should ensure the sorted table exists.
    return None, "Sorted performance table"


def ensure_report_graphs_from_session_graphs() -> bool:
    """
    Build report_graph_entries using the explicit graph configurations stored in
    multi_param_graph_{idx} session keys. This respects the Plot tab settings
    (X/Y axes, smoothing) instead of using heuristics.
    """
    try:
        # #region agent log
        try:
            import json as _json
            import time as _time
            import os as _os
            _log_path = _os.path.join(_os.path.dirname(__file__), "..", "debug-4d38f6.log")
            _graphs_meta = []
            for k, v in st.session_state.items():
                if isinstance(k, str) and k.startswith("multi_param_graph_") and isinstance(v, dict):
                    _graphs_meta.append(
                        {
                            "key": k,
                            "x_axis": v.get("x_axis"),
                            "left_y_axes": v.get("left_y_axes"),
                            "right_y_axes": v.get("right_y_axes"),
                        }
                    )
            with open(_log_path, "a", encoding="utf-8") as _f:
                _f.write(
                    _json.dumps(
                        {
                            "sessionId": "4d38f6",
                            "id": f"log_{int(_time.time()*1000)}_ensure_report_graphs_from_session_graphs_enter",
                            "timestamp": int(_time.time() * 1000),
                            "location": "plotting.py:ensure_report_graphs_from_session_graphs",
                            "message": "enter_ensure_report_graphs_from_session_graphs",
                            "data": {"graph_state_count": len(_graphs_meta), "graphs": _graphs_meta},
                            "runId": "run1",
                            "hypothesisId": "H1",
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        # Collect all configured graphs from session state.
        graph_configs: list[dict] = []
        for key, value in st.session_state.items():
            if not isinstance(key, str) or not key.startswith("multi_param_graph_"):
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

        if not graph_configs:
            return False

        df, plot_data_source = _get_report_dataframe_for_current_source()
        if df is None or getattr(df, "empty", True):
            return False

        numeric_cols = [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_cols:
            return False

        def _find_col(cols, targets):
            names = [targets] if isinstance(targets, str) else targets
            for name in names:
                low = str(name).lower()
                for col in cols:
                    if str(col).lower() == low:
                        return col
                for col in cols:
                    if low in str(col).lower():
                        return col
            return None

        # Clear existing entries so that the report reflects the current graphs only.
        st.session_state.report_graph_entries = {}

        graph_configs.sort(key=lambda g: g.get("graph_number", 0))

        # Titles aligned with Plot tab defaults.
        graph_titles = {
            1: "Speed-Based Performance Trends",
            2: "Load-Based Performance Trends",
            3: "Power and Efficiency Trends",
            4: "Thrust and Vibration Response",
            5: "Acceleration Response",
        }

        # Detect whether the dataset has any vibration channels at all, so we
        # can fall back to generic titles for Graphs 4 and 5 when vibration is
        # completely unavailable (mirrors Plot tab behaviour).
        has_vibration_anywhere = any(
            "vibration" in str(col).lower() for col in df.columns
        )

        graphs_built = 0

        _df_col_lowers = {str(c).strip().lower() for c in df.columns}
        is_rotrix_data = any(
            key in _df_col_lowers
            for key in {"testrecordid", "timestamp (hh:mm:ss)", "test type", "time type"}
        )
        time_x_rotrix = _find_col(
            numeric_cols,
            ["timestamp_seconds", "Time (s)", "Time", "Time range", "Timestamp (hh:mm:ss)"],
        )

        for cfg in graph_configs:
            graph_idx = cfg.get("graph_number")
            if not isinstance(graph_idx, int) or graph_idx <= 0:
                continue

            # Keep default X-axis intent for the first three graphs regardless
            # of stale template/session values.
            if is_rotrix_data and time_x_rotrix and graph_idx in (1, 2, 3):
                cfg["x_axis"] = time_x_rotrix
            elif graph_idx == 1:
                cfg["x_axis"] = _find_col(
                    numeric_cols,
                    ["RPM", "RPM1 - RPM", "Motor Electrical Speed (RPM)", "Motor Optical Speed (RPM)"],
                ) or cfg.get("x_axis")
            elif graph_idx == 2:
                cfg["x_axis"] = _find_col(
                    numeric_cols,
                    ["Torque (N·m)", "Torque (N*m)", "Torque (Nm)", "Torque - N*m", "Torque", "torque"],
                ) or cfg.get("x_axis")
            elif graph_idx == 3:
                cfg["x_axis"] = _find_col(
                    numeric_cols,
                    ["Thrust (gf)", "Thrust - gf", "Thrust (kgf)", "Thrust"],
                ) or cfg.get("x_axis")

            x_col = cfg.get("x_axis")
            if not x_col or x_col not in df.columns:
                continue

            left_cols = [c for c in (cfg.get("left_y_axes") or []) if c in df.columns]
            right_cols = [c for c in (cfg.get("right_y_axes") or []) if c in df.columns]
            if not left_cols and not right_cols:
                continue

            has_right = bool(right_cols)
            fig = make_subplots(specs=[[{"secondary_y": has_right}]])

            # Apply smoothing if enabled.
            df_plot = df.copy()
            if cfg.get("smoothing_enabled") and x_col:
                all_y = left_cols + right_cols
                if all_y:
                    try:
                        df_plot = smooth_data_for_plotting(
                            df_plot,
                            x_col,
                            all_y,
                            method=cfg.get("smoothing_method", "savgol"),
                            smoothing_window=cfg.get("smoothing_window", 5),
                        )
                    except Exception:
                        df_plot = df.copy()

            # ---- Compute per-param ranges (mirroring the Plot tab) ----
            def _param_range(col_name):
                vals = df_plot[col_name].dropna().tolist() if col_name in df_plot.columns else []
                if not vals:
                    return (0.0, 1.0)
                c_min = float(min(vals))
                c_max = float(max(vals))
                if c_min > 0:
                    c_min = 0.0
                if c_max > 0:
                    c_max = c_max + max(c_max * 0.05, c_max * 0.02)
                else:
                    c_max = 1.0
                return (c_min, c_max)

            left_ranges = {c: _param_range(c) for c in left_cols}
            right_ranges = {c: _param_range(c) for c in right_cols}

            colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728',
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            color_idx = 0

            for ti, col in enumerate(left_cols):
                sub = df_plot[[x_col, col]].dropna(subset=[x_col, col])
                if sub.empty:
                    continue
                trace_color = colors[color_idx % len(colors)]
                color_idx += 1
                if ti == 0:
                    fig.add_trace(
                        go.Scatter(
                            x=sub[x_col], y=sub[col],
                            mode="lines+markers",
                            name=get_display_name(col),
                            line=dict(color=trace_color, width=2),
                            marker=dict(size=4, color=trace_color),
                        ),
                        secondary_y=False,
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=sub[x_col], y=sub[col],
                            mode="lines+markers",
                            name=get_display_name(col),
                            line=dict(color=trace_color, width=2),
                            marker=dict(size=4, color=trace_color),
                            yaxis='y3',
                        ),
                    )

            for ti, col in enumerate(right_cols):
                sub = df_plot[[x_col, col]].dropna(subset=[x_col, col])
                if sub.empty:
                    continue
                trace_color = colors[color_idx % len(colors)]
                color_idx += 1
                if ti == 0:
                    fig.add_trace(
                        go.Scatter(
                            x=sub[x_col], y=sub[col],
                            mode="lines+markers",
                            name=get_display_name(col),
                            line=dict(color=trace_color, width=2, dash="solid"),
                            marker=dict(size=4, color=trace_color),
                        ),
                        secondary_y=True,
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=sub[x_col], y=sub[col],
                            mode="lines",
                            name=get_display_name(col),
                            line=dict(color=trace_color, width=2, dash="solid"),
                            yaxis="y4",
                        ),
                    )

            if not fig.data:
                continue

            multi_axis_mode = (
                (left_cols and len(left_cols) > 1)
                or (right_cols and len(right_cols) > 1)
            )
            dual_axis = bool(left_cols and right_cols)

            try:
                current_meta = fig.layout.meta or {}
                if not isinstance(current_meta, dict):
                    current_meta = {}
                current_meta.update(
                    {
                        # Mark multi_y_params when any side has 2+ params so
                        # export helpers adjust margins/titles accordingly.
                        "multi_y_params": bool(multi_axis_mode),
                        "multi_y_left_count": len(left_cols or []),
                        "multi_y_right_count": len(right_cols or []),
                    }
                )
                fig.update_layout(meta=current_meta)
            except Exception:
                pass

            axis_layout: dict = {}

            # Primary left axis (yaxis)
            if left_cols and left_cols[0] in left_ranges:
                y_min, y_max = left_ranges[left_cols[0]]
                axis_layout["yaxis"] = dict(
                    # When any side has 2+ params, remove side title
                    # and rely on top annotations.  Classic dual-axis
                    # (1 left + 1 right) and single-axis keep side titles.
                    title="" if multi_axis_mode else get_display_name(left_cols[0]),
                    range=[y_min, y_max],
                    showgrid=True,
                    gridcolor="#999999",
                    gridwidth=1,
                    showline=True,
                    linecolor="#333333",
                    linewidth=1.5,
                    zeroline=True,
                    zerolinecolor="#999999",
                    zerolinewidth=1.0,
                    title_font=dict(size=22, color="black"),
                    tickfont=dict(size=18, color="black"),
                    side="left",
                    position=0.0,
                )

            # Second left axis (yaxis3) — free-anchored, own scale
            if left_cols and len(left_cols) > 1 and left_cols[1] in left_ranges:
                y_min, y_max = left_ranges[left_cols[1]]
                axis_layout["yaxis3"] = dict(
                    title="",  # secondary left; top labels / legend carry meaning
                    title_font=dict(size=22, color="black"),
                    tickfont=dict(size=18, color="black"),
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
            if has_right and right_cols and right_cols[0] in right_ranges:
                y_min, y_max = right_ranges[right_cols[0]]
                _right_title = "" if multi_axis_mode else get_display_name(right_cols[0])
                _has_accz = any("accz" in str(c).lower() for c in right_cols)
                yaxis2_cfg = dict(
                    title=_right_title,
                    range=[y_min, y_max],
                    showgrid=False,
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
                axis_layout["yaxis2"] = yaxis2_cfg

            # Second right axis (yaxis4) — free-anchored, own scale
            if right_cols and len(right_cols) > 1 and right_cols[1] in right_ranges:
                y_min, y_max = right_ranges[right_cols[1]]
                axis_layout["yaxis4"] = dict(
                    title="",  # secondary right; top labels / legend carry meaning
                    title_font=dict(size=22, color="black"),
                    tickfont=dict(size=18, color="black"),
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
                fig.update_layout(**axis_layout)

            fig.update_xaxes(
                title=get_axis_title(x_col),
                showgrid=True,
                gridcolor="#999999",
                gridwidth=1,
                showline=True,
                linecolor="#333333",
                linewidth=1.5,
                nticks=16,
                title_font=dict(size=22, color="black"),
                tickfont=dict(size=18, color="black"),
            )

            _has_dual_axis = bool(left_cols and right_cols)
            fig.update_layout(
                template="plotly_white",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                ),
                margin=dict(
                    l=40 if multi_axis_mode else 60,
                    r=40 if multi_axis_mode else 60,
                    t=90 if multi_axis_mode else 60,
                    b=50,
                ),
            )
            color_map = {c: colors[i % len(colors)] for i, c in enumerate((left_cols or []) + (right_cols or []))}
            add_top_param_labels(fig, left_cols, right_cols, color_map)

            # Titles aligned with Plot tab, with vibration‑aware fallback for
            # Graphs 4 and 5 when there is no vibration channel at all.
            title = graph_titles.get(graph_idx, f"Graph {graph_idx}")
            if graph_idx in (4, 5) and not has_vibration_anywhere:
                title = f"Graph {graph_idx}"
            store_graph_for_report(f"Graph {graph_idx}", fig, table=None, heading=title)
            graphs_built += 1

            # #region agent log
            try:
                import json as _json
                import time as _time
                import os as _os
                _log_path = _os.path.join(_os.path.dirname(__file__), "..", "debug-4d38f6.log")
                with open(_log_path, "a", encoding="utf-8") as _f:
                    _f.write(
                        _json.dumps(
                            {
                                "sessionId": "4d38f6",
                                "id": f"log_{int(_time.time()*1000)}_ensure_report_graphs_from_session_graphs_graph",
                                "timestamp": int(_time.time() * 1000),
                                "location": "plotting.py:ensure_report_graphs_from_session_graphs",
                                "message": "built_graph_from_session_config",
                                "data": {
                                    "graph_idx": graph_idx,
                                    "title": title,
                                    "x_col": x_col,
                                    "left_cols": left_cols,
                                    "right_cols": right_cols,
                                    "multi_axis_mode": multi_axis_mode,
                                    "has_vibration_anywhere": has_vibration_anywhere,
                                },
                                "runId": "run1",
                                "hypothesisId": "H2",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion

        if graphs_built > 0:
            st.session_state.report_graphs_data_source = plot_data_source
            # #region agent log
            try:
                import json as _json
                import time as _time
                import os as _os
                _log_path = _os.path.join(_os.path.dirname(__file__), "..", "debug-4d38f6.log")
                with open(_log_path, "a", encoding="utf-8") as _f:
                    _f.write(
                        _json.dumps(
                            {
                                "sessionId": "4d38f6",
                                "id": f"log_{int(_time.time()*1000)}_ensure_report_graphs_from_session_graphs_built",
                                "timestamp": int(_time.time() * 1000),
                                "location": "plotting.py:ensure_report_graphs_from_session_graphs",
                                "message": "graphs_built_from_session_graphs",
                                "data": {"graphs_built": graphs_built, "plot_data_source": plot_data_source},
                                "runId": "run1",
                                "hypothesisId": "H1",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            return True
        return False
    except Exception as _e:
        # #region agent log
        try:
            import json as _json
            import time as _time
            import os as _os
            _log_path = _os.path.join(_os.path.dirname(__file__), "..", "debug-4d38f6.log")
            with open(_log_path, "a", encoding="utf-8") as _f:
                _f.write(
                    _json.dumps(
                        {
                            "sessionId": "4d38f6",
                            "id": f"log_{int(_time.time()*1000)}_ensure_report_graphs_from_session_graphs_error",
                            "timestamp": int(_time.time() * 1000),
                            "location": "plotting.py:ensure_report_graphs_from_session_graphs",
                            "message": "error_in_ensure_report_graphs_from_session_graphs",
                            "data": {"exception_type": type(_e).__name__, "exception_msg": str(_e)},
                            "runId": "run1",
                            "hypothesisId": "H1",
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        return False


def ensure_report_graphs_for_current_config() -> bool:
    """
    Entry point used by the app to ensure report graphs exist for the current
    configuration. Prefer graphs derived from Plot-tab settings; fall back to
    heuristic generation when no explicit graph configs exist.
    """
    # If any multi_param_graph_* entries exist, prefer using them so that
    # template settings and manual Plot-tab edits are respected.
    has_explicit_graphs = any(
        isinstance(k, str) and k.startswith("multi_param_graph_")
        for k in st.session_state.keys()
    )

    # #region agent log
    try:
        import json as _json
        import time as _time
        import os as _os
        _log_path = _os.path.join(_os.path.dirname(__file__), "..", "debug-4d38f6.log")
        with open(_log_path, "a", encoding="utf-8") as _f:
            _f.write(
                _json.dumps(
                    {
                        "sessionId": "4d38f6",
                        "id": f"log_{int(_time.time()*1000)}_ensure_report_graphs_for_current_config_branch",
                        "timestamp": int(_time.time() * 1000),
                        "location": "plotting.py:ensure_report_graphs_for_current_config",
                        "message": "ensure_report_graphs_for_current_config_branch",
                        "data": {"has_explicit_graphs": has_explicit_graphs},
                        "runId": "run1",
                        "hypothesisId": "H1",
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion

    if has_explicit_graphs:
        return ensure_report_graphs_from_session_graphs()

    # Otherwise, keep the existing behaviour: auto-generate graphs from the
    # currently selected file when needed.
    return ensure_report_performance_graphs_for_current_file()


# ---------------------------------------------------------------------------
# Throttle summary graphs
# ---------------------------------------------------------------------------

THROTTLE_BAND_LABELS = [f"{i}-{i+10}" for i in range(0, 100, 10)]

_THROTTLE_BAR_COLORS = [
    "rgba(52, 152, 219, 0.85)", "rgba(230, 126, 34, 0.85)", "rgba(46, 204, 113, 0.85)",
    "rgba(26, 188, 156, 0.85)", "rgba(155, 89, 182, 0.85)", "rgba(231, 76, 60, 0.85)",
    "rgba(241, 196, 15, 0.85)", "rgba(149, 165, 166, 0.85)", "rgba(192, 57, 43, 0.85)",
    "rgba(41, 128, 185, 0.85)",
]


def find_monotonic_time_column_for_dwell(df):
    """Column with absolute elapsed time (seconds), sorted ascending for dwell sums."""
    if df is None or getattr(df, "empty", True):
        return None
    for c in df.columns:
        if str(c).lower() == "timestamp_seconds":
            return c
    for c in df.columns:
        s = str(c).strip().lower()
        if s in ("time (s)", "time (secs)"):
            return c
    for c in df.columns:
        cl = str(c).lower()
        if "time" in cl and "range" not in cl:
            return c
    return None


def compute_dwell_seconds_by_band(df):
    """
    Sum sample-interval durations (Δt) to the throttle band active at the start of each interval.

    Returns (labels, seconds_per_label, total_seconds) or None if not computable.
    """
    tr_col = "Throttle Range (10%)"
    if df is None or tr_col not in df.columns:
        return None
    tc = find_monotonic_time_column_for_dwell(df)
    if tc is None:
        return None
    d = df[[tr_col, tc]].copy()
    d[tc] = pd.to_numeric(d[tc], errors="coerce")
    d = d.dropna(subset=[tc])
    d[tr_col] = d[tr_col].apply(lambda x: str(x).strip() if pd.notna(x) else "nan")
    # Keep original sample order. Sorting by time can mis-assign dwell when
    # logs contain repeated/reset timestamps across stitched segments.
    if len(d) < 2:
        return None
    t = d[tc].to_numpy(dtype=float)
    bands = d[tr_col].to_numpy()
    dts = np.diff(t)
    dwell = {lb: 0.0 for lb in THROTTLE_BAND_LABELS}
    pos = dts[dts > 0]
    med_dt = float(np.median(pos)) if len(pos) > 0 else 1.0
    max_allowed = max(med_dt * 500.0, 120.0)
    for j in range(len(dts)):
        dt = float(dts[j])
        if dt <= 0 or dt > max_allowed:
            continue
        b = str(bands[j]).strip()
        if b.lower() in ("nan", "none", ""):
            continue
        if b not in dwell:
            dwell[b] = 0.0
        dwell[b] += dt
    labels = list(THROTTLE_BAND_LABELS)
    secs = [float(dwell.get(lb, 0.0)) for lb in labels]
    total = float(sum(secs))
    if total <= 0:
        return None
    return labels, secs, total


def build_throttle_dwell_bar_figure(bar_df_raw):
    """
    Bar chart: total elapsed time (s) per 10% throttle band (section 2.1.1).
    Returns None if dwell cannot be computed.
    """
    res = compute_dwell_seconds_by_band(bar_df_raw)
    if res is None:
        return None
    labels, secs, _total = res
    colors = [_THROTTLE_BAR_COLORS[i % len(_THROTTLE_BAR_COLORS)] for i in range(len(labels))]

    fig_sec = go.Figure()
    fig_sec.add_trace(
        go.Bar(
            x=labels,
            y=secs,
            name="Total time (s)",
            marker=dict(color=colors),
            text=[f"{s:.1f}" if s >= 0.05 else "" for s in secs],
            textposition="outside",
        )
    )
    fig_sec.update_layout(
        xaxis=dict(
            title="Throttle range (%)",
            title_font=dict(size=22, color="black"),
            tickfont=dict(size=16, color="black"),
            showgrid=False,
            showline=True,
            linecolor="#333333",
            linewidth=1.5,
        ),
        yaxis=dict(
            title="Total time (s)",
            title_font=dict(size=22, color="black"),
            tickfont=dict(size=18, color="black"),
            nticks=12,
            showgrid=True,
            gridcolor="#e0e0e0",
            showline=True,
            linecolor="#333333",
            linewidth=1.5,
        ),
        margin=dict(l=60, r=60, t=48, b=60),
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
    )
    return fig_sec


def build_stacked_thrust_area_figure(bar_df_raw):
    """
    Stacked area chart: thrust (or vibration) vs absolute time, one layer per throttle band (section 2.1.3).
    Matches the bench report: X = Time (s), Y = Thrust - gf, legend = throttle ranges.
    Returns (figure, area_is_vibration) or (None, False) if not buildable.
    """
    try:
        area_df = bar_df_raw.copy()
        area_df["Throttle Range (10%)"] = area_df["Throttle Range (10%)"].astype(str)

        time_axis_col = None
        for c in area_df.columns:
            if str(c).lower() == "timestamp_seconds":
                time_axis_col = c
                break
        if time_axis_col is None:
            for c in area_df.columns:
                if str(c).strip() == "Time (s)" or c == "Time (secs)" or str(c).lower() == "time (s)":
                    time_axis_col = c
                    break
        if time_axis_col is None:
            for c in area_df.columns:
                if "time" in str(c).lower() and "range" not in str(c).lower():
                    time_axis_col = c
                    break
        if time_axis_col is None and "Time range" in area_df.columns:
            time_axis_col = "Time range"

        area_y_col = None
        area_is_vibration = False
        vib_col = None
        for c in area_df.columns:
            if "vibration" in str(c).lower() and (
                "(g)" in str(c) or " - g" in str(c).lower() or str(c).lower().endswith("g")
            ):
                vib_col = c
                break
        if vib_col is None:
            for c in area_df.columns:
                if "vibration" in str(c).lower():
                    vib_col = c
                    break
        if vib_col is not None:
            area_y_col = vib_col
            area_is_vibration = True
        else:
            thrust_col_area = None
            for c in area_df.columns:
                if "thrust" in str(c).lower() and ("gf" in str(c).lower() or "g)" in str(c).lower()):
                    thrust_col_area = c
                    break
            if thrust_col_area is None:
                for c in area_df.columns:
                    if "thrust" in str(c).lower():
                        thrust_col_area = c
                        break
            if thrust_col_area is not None:
                area_y_col = thrust_col_area
                area_is_vibration = False

        if not time_axis_col or area_y_col is None or area_y_col not in area_df.columns:
            return None, False

        area_df[time_axis_col] = pd.to_numeric(area_df[time_axis_col], errors="coerce")
        area_df[area_y_col] = pd.to_numeric(area_df[area_y_col], errors="coerce")
        area_df = area_df.dropna(subset=[time_axis_col, "Throttle Range (10%)", area_y_col])
        if area_df.empty:
            return None, False

        non_zero_mask = area_df[area_y_col].abs() > 0
        if non_zero_mask.any():
            first_active_time = area_df.loc[non_zero_mask, time_axis_col].min()
        else:
            first_active_time = area_df[time_axis_col].min()
        area_df[time_axis_col] = area_df[time_axis_col] - first_active_time
        area_df = area_df[area_df[time_axis_col] >= 0].copy()

        pivot = area_df.pivot_table(
            index=time_axis_col,
            columns="Throttle Range (10%)",
            values=area_y_col,
            aggfunc="mean",
            fill_value=0,
        )
        order_tr = [f"{i}-{i+10}" for i in range(0, 100, 10)]
        pivot = pivot.reindex(columns=[c for c in order_tr if c in pivot.columns])
        pivot = pivot.sort_index()
        if pivot.empty or len(pivot.columns) == 0:
            return None, False

        for col in pivot.columns:
            mask = pivot[col] != 0
            if mask.any():
                first_active = mask.idxmax()
                last_active = mask[::-1].idxmax()
                rep_value = pivot.loc[mask, col].mean()
                pivot.loc[first_active:last_active, col] = pivot.loc[
                    first_active:last_active, col
                ].replace(0, rep_value)
                pivot.loc[first_active:, col] = pivot.loc[first_active:, col].replace(
                    0, rep_value
                )

        colors_area = [
            "rgba(52, 152, 219, 0.7)", "rgba(230, 126, 34, 0.7)", "rgba(46, 204, 113, 0.7)",
            "rgba(26, 188, 156, 0.7)", "rgba(155, 89, 182, 0.7)", "rgba(231, 76, 60, 0.7)",
            "rgba(241, 196, 15, 0.7)", "rgba(149, 165, 166, 0.7)", "rgba(192, 57, 43, 0.7)",
            "rgba(41, 128, 185, 0.7)",
        ]
        fig_area = go.Figure()
        for i, tr_label in enumerate(pivot.columns):
            fig_area.add_trace(
                go.Scatter(
                    x=pivot.index.values,
                    y=pivot[tr_label].values,
                    name=str(tr_label),
                    stackgroup="one",
                    fill="tonexty" if i > 0 else "tozeroy",
                    line=dict(width=0.5, color=colors_area[i % len(colors_area)]),
                )
            )
        if area_is_vibration:
            y_axis_title = "Vibration (g)"
        else:
            _ac = str(area_y_col).lower()
            if "thrust" in _ac and "gf" in _ac:
                y_axis_title = "Thrust - gf"
            elif "thrust" in _ac and "kgf" in _ac:
                y_axis_title = "Thrust - kgf"
            else:
                y_axis_title = get_display_name(area_y_col)

        fig_area.update_layout(
            xaxis=dict(
                title="Time (s)",
                title_font=dict(size=22, color="black"),
                tickfont=dict(size=18, color="black"),
                nticks=16,
                showgrid=False,
                showline=True,
                linecolor="#333333",
                linewidth=1.5,
            ),
            yaxis=dict(
                title=y_axis_title,
                title_font=dict(size=22, color="black"),
                tickfont=dict(size=18, color="black"),
                nticks=16,
                showgrid=False,
                showline=True,
                linecolor="#333333",
                linewidth=1.5,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=16, color="black"),
                title=dict(text="Throttle Range"),
                title_font=dict(size=16, color="black"),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
            ),
            margin=dict(l=60, r=60, t=72, b=60),
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
        )
        return fig_area, area_is_vibration
    except Exception:
        return None, False


def ensure_summary_graphs_for_current_file():
    """
    Build throttle summary graphs (line, bar, area) for report Section 2.1.
    """
    try:
        bar_df_raw = getattr(st.session_state, "report_raw_data_df", None)
        has_tr_cols = (
            bar_df_raw is not None
            and not getattr(bar_df_raw, "empty", True)
            and "Throttle Range (10%)" in bar_df_raw.columns
            and "Time range" in bar_df_raw.columns
        )
        if not has_tr_cols:
            return False

        built_any = False

        # ---- Line summary ----
        try:
            line_df = bar_df_raw.copy()
            line_df["Time range"] = pd.to_numeric(line_df["Time range"], errors="coerce")
            line_df["Throttle Range (10%)"] = line_df["Throttle Range (10%)"].astype(str)
            line_df = line_df[line_df["Throttle Range (10%)"] != "0-10"]
            thrust_col_line = None
            for c in line_df.columns:
                if "thrust" in str(c).lower() and ("gf" in str(c).lower() or "g)" in str(c).lower()):
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
                colors = [
                    "rgba(231, 76, 60, 0.9)", "rgba(52, 152, 219, 0.9)",
                    "rgba(46, 204, 113, 0.9)", "rgba(241, 196, 15, 0.9)",
                    "rgba(155, 89, 182, 0.9)", "rgba(26, 188, 156, 0.9)",
                    "rgba(230, 126, 34, 0.9)", "rgba(149, 165, 166, 0.9)",
                    "rgba(192, 57, 43, 0.9)", "rgba(41, 128, 185, 0.9)",
                ]
                for color_idx, (tr_label, grp) in enumerate(
                    line_df.groupby("Throttle Range (10%)", observed=True, sort=False)
                ):
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
                            line=dict(color=colors[color_idx % len(colors)], width=2),
                        )
                    )
            fig_line.update_layout(
                xaxis=dict(title="Time range (s)", title_font=dict(size=22, color="black"),
                           tickfont=dict(size=18, color="black"), nticks=10,
                           showgrid=True, gridcolor="#e0e0e0", gridwidth=1,
                           showline=True, linecolor="#333333", linewidth=1.5),
                yaxis=dict(title=thrust_col_line if thrust_col_line else "Thrust",
                           title_font=dict(size=22, color="black"),
                           tickfont=dict(size=18, color="black"), nticks=10,
                           showgrid=True, gridcolor="#e0e0e0", gridwidth=1,
                           showline=True, linecolor="#333333", linewidth=1.5),
                margin=dict(l=60, r=60, t=40, b=60),
                template="plotly_white",
                hovermode="x unified",
            )
            if len(fig_line.data) > 0:
                st.session_state["report_throttle_line_fig"] = fig_line
                built_any = True
        except Exception:
            pass

        # ---- Bar summary (2.1.1): total elapsed time per throttle band (Δt sums) ----
        try:
            fig_dwell = build_throttle_dwell_bar_figure(bar_df_raw)
            if fig_dwell is not None:
                st.session_state["report_throttle_bar_fig"] = fig_dwell
                built_any = True
        except Exception:
            pass

        # ---- Third chart (2.1.3) is hidden for now ----
        st.session_state.pop("report_throttle_area_fig", None)
        st.session_state.pop("report_throttle_area_is_vibration", None)

        return built_any
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Figure export helpers
# ---------------------------------------------------------------------------

def _prepare_fig_for_export(fig):
    """
    Return a copy of the figure with font sizes suitable for PDF embedding.
    """
    if fig is None:
        return None
    try:
        if not isinstance(fig, go.Figure):
            return fig
        copy_fn = getattr(fig, "full_copy", None) or getattr(fig, "copy", None)
        if callable(copy_fn):
            try:
                export_fig = copy_fn()
            except Exception:
                export_fig = go.Figure(fig.to_dict())
        else:
            export_fig = go.Figure(fig.to_dict())

        tick_pt = 26
        axis_title_pt = 28
        legend_pt = 20

        layout_meta = getattr(export_fig.layout, "meta", {}) or {}
        multi_y_params = False
        try:
            if isinstance(layout_meta, dict):
                multi_y_params = bool(layout_meta.get("multi_y_params"))
        except Exception:
            multi_y_params = False

        export_fig.update_xaxes(
            tickfont=dict(size=tick_pt, color="black"),
            title_font=dict(size=axis_title_pt, color="black"),
        )

        if multi_y_params:
            # Dual-axis graphs: preserve the (empty) side titles that were
            # intentionally cleared so they don't get re-introduced by
            # update_yaxes.  Non-dual-axis graphs skip this block entirely
            # so their original axis labels (e.g. "Thrust (gf)") are
            # never overwritten.
            def _get_title(layout_axis):
                try:
                    t = getattr(layout_axis, "title", None)
                    return getattr(t, "text", None) if t is not None else None
                except Exception:
                    return None
            y1_title = _get_title(getattr(export_fig.layout, "yaxis", None))
            y2_title = _get_title(getattr(export_fig.layout, "yaxis2", None))
            export_fig.update_yaxes(
                tickfont=dict(size=tick_pt, color="black"),
                title_font=dict(size=axis_title_pt, color="black"),
            )
            if y1_title is not None:
                export_fig.update_layout(yaxis=dict(title=y1_title, title_font=dict(size=axis_title_pt, color="black")))
            if y2_title is not None:
                export_fig.update_layout(yaxis2=dict(title=y2_title, title_font=dict(size=axis_title_pt, color="black")))
        else:
            # Non-dual-axis: just bump font sizes, titles stay as-is
            export_fig.update_yaxes(
                tickfont=dict(size=tick_pt, color="black"),
                title_font=dict(size=axis_title_pt, color="black"),
            )

        extra_tick_pt = max(tick_pt - 8, 12)
        for attr in ("yaxis3", "yaxis4"):
            if hasattr(export_fig.layout, attr) and getattr(export_fig.layout, attr) is not None:
                this_tick = extra_tick_pt
                export_fig.update_layout(**{
                    attr: dict(
                        tickfont=dict(size=this_tick, color="black"),
                        title_font=dict(size=axis_title_pt, color="black"),
                    )
                })

        export_fig.update_layout(
            legend=dict(
                font=dict(size=legend_pt, color="black"),
                title_font=dict(size=legend_pt, color="black"),
            )
        )

        if multi_y_params:
            try:
                margin = getattr(export_fig.layout, "margin", None)
                current_t = getattr(margin, "t", None) if margin is not None else None
                current_l = getattr(margin, "l", None) if margin is not None else None
                current_r = getattr(margin, "r", None) if margin is not None else None
                current_b = getattr(margin, "b", None) if margin is not None else None
                new_margin = dict(
                    l=max(current_l if isinstance(current_l, (int, float)) else 60, 70),
                    r=max(current_r if isinstance(current_r, (int, float)) else 60, 70),
                    b=current_b if isinstance(current_b, (int, float)) else 50,
                    t=max(current_t if isinstance(current_t, (int, float)) else 60, 95),
                )
                export_fig.update_layout(margin=new_margin)

                anns = list(getattr(export_fig.layout, "annotations", []) or [])
                new_anns = []
                for ann in anns:
                    try:
                        y_val = getattr(ann, "y", 0)
                        if isinstance(y_val, (int, float)) and y_val > 1.0:
                            font = getattr(ann, "font", None) or {}
                            size = getattr(font, "size", 14)
                            new_size = max(size, 18)
                            ann.font = dict(
                                **{k: getattr(font, k) for k in dir(font) if not k.startswith("_") and k in ("color",)},
                                size=new_size,
                            )
                    except Exception:
                        pass
                    new_anns.append(ann)
                export_fig.layout.annotations = tuple(new_anns)
            except Exception:
                pass

        return export_fig
    except Exception:
        return fig


# Module-level Kaleido scope singleton — avoids spawning a new Chromium process
# per image export. This saves ~200MB RAM per additional graph.
_KALEIDO_SCOPE = None
_KALEIDO_LOCK = threading.Lock()


def _get_kaleido_scope():
    """Lazily create a shared Kaleido PlotlyScope instance (thread-safe)."""
    global _KALEIDO_SCOPE
    if _KALEIDO_SCOPE is None:
        with _KALEIDO_LOCK:
            if _KALEIDO_SCOPE is None:
                try:
                    from kaleido.scopes.plotly import PlotlyScope  # type: ignore
                    _KALEIDO_SCOPE = PlotlyScope()
                except Exception:
                    _KALEIDO_SCOPE = None
    return _KALEIDO_SCOPE


def _fig_to_image_bytes(fig, quality=None):
    """Convert a Plotly figure to PNG bytes for embedding in reports.
    
    Adapts image resolution to available system RAM:
      - Low RAM (< 700MB):  900×450 @ 0.75x scale (~150KB/image)
      - Medium (700-2GB):   1200×600 @ 1.0x  (~300KB/image)
      - High (> 2GB):       1200×600 @ 1.0x  (~300KB/image)
    """
    if fig is None:
        return ""

    export_fig = _prepare_fig_for_export(fig)
    if export_fig is None:
        export_fig = fig

    try:
        if not isinstance(fig, go.Figure):
            return ""
    except Exception:
        pass

    # Detect quality settings from system resources
    if quality is None:
        try:
            from resource_manager import detect_quality
            quality = detect_quality()
        except Exception:
            quality = None

    if quality is not None:
        _width = quality.kaleido_width
        _height = quality.kaleido_height
        _scale = quality.kaleido_scale
        _do_gc = quality.aggressive_gc
    else:
        _width = 1200
        _height = 720
        _scale = 1.0
        _do_gc = False

    try:
        # Try reusable Kaleido scope (singleton, no new Chromium process)
        scope = _get_kaleido_scope()
        if scope is not None:
            try:
                with _KALEIDO_LOCK:
                    img_bytes = scope.transform(
                        export_fig,
                        format="png",
                        width=_width,
                        height=_height,
                        scale=_scale,
                    )
                if img_bytes and len(img_bytes) > 0:
                    if _do_gc:
                        import gc; gc.collect()
                    return img_bytes
            except Exception:
                pass

        # Try kaleido engine via plotly
        try:
            import kaleido  # noqa: F401
            img_bytes = export_fig.to_image(format="png", width=_width, height=_height, scale=_scale, engine="kaleido")
            if img_bytes and len(img_bytes) > 0:
                if _do_gc:
                    import gc; gc.collect()
                return img_bytes
        except (ImportError, ModuleNotFoundError):
            pass
        except Exception:
            pass

        # Fallback: default engine
        try:
            img_bytes = export_fig.to_image(format="png", width=_width, height=_height, scale=_scale)
            if img_bytes and len(img_bytes) > 0:
                if _do_gc:
                    import gc; gc.collect()
                return img_bytes
        except Exception:
            pass

        return ""
    except Exception:
        return ""


def _fig_to_base64(fig, quality=None):
    """Convert a Plotly figure to base64 PNG for embedding in HTML reports."""
    img_bytes = _fig_to_image_bytes(fig, quality=quality)
    if not img_bytes:
        return ""
    return base64.b64encode(img_bytes).decode("utf-8")


def _fig_content_hash(fig) -> str:
    """Compute a hash of a Plotly figure's JSON for caching."""
    try:
        fig_json = fig.to_json()
        return hashlib.md5(fig_json.encode()).hexdigest()
    except Exception:
        return ""


@st.cache_data(show_spinner=False, max_entries=50)
def cached_fig_to_png(fig_hash: str, _fig, quality=None) -> bytes:
    """Cached wrapper around _fig_to_image_bytes.

    Avoids re-exporting the same figure to PNG on repeated calls (e.g.
    during PDF report generation or preview updates).
    """
    return _fig_to_image_bytes(_fig, quality=quality)

