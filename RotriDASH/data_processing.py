#type: ignore
"""
Throttle aggregation, data smoothing, abnormality detection,
resampling, and table sanitization utilities.
"""
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

from config import get_short_param_label, get_display_name
from data_loader import find_column_by_pattern


def _df_hash(df: pd.DataFrame) -> str:
    """Fast hash of DataFrame shape + sample values for cache keying."""
    sig = f"{df.shape}|{list(df.columns)}|{df.iloc[0].tolist() if len(df) > 0 else []}|{df.iloc[-1].tolist() if len(df) > 0 else []}"
    return hashlib.md5(sig.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Plotting-related column helpers
# ---------------------------------------------------------------------------

def add_top_param_labels(fig, left_cols, right_cols, color_map):
    """
    Add compact, colour-coded axis labels at the top of the plot.

    Labels are placed in two groups:
    - Left Y-axis labels span from x = 0.0  to x = 0.40  (left-aligned)
    - Right Y-axis labels span from x = 0.60 to x = 1.0  (right-aligned)

    Each group distributes its items evenly across its half so that
    labels never overlap regardless of how many axes are shown.
    """
    try:
        left_cols = left_cols or []
        right_cols = right_cols or []
        if not (len(left_cols) > 1 or len(right_cols) > 1):
            return

        base_y = 1.06
        color_map = color_map or {}
        _FONT_SIZE = 12

        n_left = min(len(left_cols), 4)
        n_right = min(len(right_cols), 4)

        if n_left > 0:
            left_zone = (-0.04, 0.05)
            span = left_zone[1] - left_zone[0]
            if n_left == 1:
                positions = [left_zone[0]]
            else:
                step = span / (n_left - 1)
                positions = [left_zone[0] + i * step for i in range(n_left)]
            for i, col in enumerate(left_cols[:n_left]):
                label = get_short_param_label(col)
                if not label:
                    continue
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=positions[i], y=base_y,
                    text=label, showarrow=False,
                    xanchor="center" if n_left > 1 else "left",
                    yanchor="bottom",
                    font=dict(size=_FONT_SIZE, color=color_map.get(col, "black")),
                )

        if n_right > 0:
            right_zone = (0.90, 0.97)
            span = right_zone[1] - right_zone[0]
            if n_right == 1:
                positions = [right_zone[1]]
            else:
                step = span / (n_right - 1)
                positions = [right_zone[0] + i * step for i in range(n_right)]
            for i, col in enumerate(right_cols[:n_right]):
                label = get_short_param_label(col)
                if not label:
                    continue
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=positions[i], y=base_y,
                    text=label, showarrow=False,
                    xanchor="center" if n_right > 1 else "right",
                    yanchor="bottom",
                    font=dict(size=_FONT_SIZE, color=color_map.get(col, "black")),
                )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Numeric column helpers
# ---------------------------------------------------------------------------

def get_numeric_columns(df):
    """Get numeric columns from dataframe."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def is_column_empty(df, col):
    """Check if a column is empty (all null/NaN values)."""
    if col not in df.columns:
        return True
    return df[col].isna().all() or (df[col] == 0).all() or len(df[col].dropna()) == 0


def get_non_empty_columns(df1, df2, columns):
    """Get columns that are not empty in both dataframes."""
    non_empty_cols = []
    for col in columns:
        if col in df1.columns and col in df2.columns:
            if not is_column_empty(df1, col) and not is_column_empty(df2, col):
                non_empty_cols.append(col)
    return non_empty_cols


# ---------------------------------------------------------------------------
# Smoothing / interpolation
# ---------------------------------------------------------------------------

def smooth_data_for_plotting(df, x_col, y_cols, method='linear', num_points=None, smoothing_window=5):
    """
    Smooth/interpolate data to create smooth lines while preserving all original data points.
    """
    if df is None or df.empty or x_col not in df.columns:
        return df

    df_sorted = df.sort_values(by=x_col).copy()
    df_sorted = df_sorted.drop_duplicates(subset=[x_col], keep='first')

    if len(df_sorted) < 2:
        return df_sorted

    x_min = df_sorted[x_col].min()
    x_max = df_sorted[x_col].max()
    x_range = x_max - x_min

    if num_points is None:
        original_points = len(df_sorted)
        if x_range > 0:
            num_points = max(200, min(original_points * 10, 2000))
        else:
            num_points = original_points

    x_interp = np.linspace(x_min, x_max, num_points)
    result_df = pd.DataFrame({x_col: x_interp})

    for y_col in y_cols:
        if y_col not in df_sorted.columns:
            continue

        valid_mask = df_sorted[y_col].notna()
        if valid_mask.sum() < 2:
            result_df[y_col] = np.nan
            continue

        x_valid = df_sorted.loc[valid_mask, x_col].values
        y_valid = df_sorted.loc[valid_mask, y_col].values

        unique_mask = np.concatenate(([True], np.diff(x_valid) != 0))
        x_unique = x_valid[unique_mask]
        y_unique = y_valid[unique_mask]

        if len(x_unique) < 2:
            result_df[y_col] = np.nan
            continue

        try:
            if method == 'moving_average' and len(y_unique) > smoothing_window:
                window = min(smoothing_window, len(y_unique) // 3)
                if window >= 3:
                    temp_series = pd.Series(y_unique, index=x_unique)
                    y_smoothed = temp_series.rolling(window=window, center=True, min_periods=1).mean().values
                    y_unique = y_smoothed
            elif method == 'savgol' and len(y_unique) > smoothing_window:
                try:
                    from scipy.signal import savgol_filter
                    window_length = min(smoothing_window, len(y_unique))
                    if window_length % 2 == 0:
                        window_length += 1
                    if window_length >= 3 and window_length <= len(y_unique):
                        polyorder = min(3, window_length - 1)
                        y_unique = savgol_filter(y_unique, window_length, polyorder)
                except ImportError:
                    window = min(smoothing_window, len(y_unique) // 3)
                    if window >= 3:
                        temp_series = pd.Series(y_unique, index=x_unique)
                        y_unique = temp_series.rolling(window=window, center=True, min_periods=1).mean().values

            if method == 'cubic' and len(x_unique) >= 4:
                try:
                    from scipy.interpolate import interp1d
                    f = interp1d(x_unique, y_unique, kind='cubic', bounds_error=False, fill_value='extrapolate')
                    y_interp = f(x_interp)
                except (ImportError, ValueError):
                    y_interp = np.interp(x_interp, x_unique, y_unique)
            else:
                y_interp = np.interp(x_interp, x_unique, y_unique)

            result_df[y_col] = y_interp
        except Exception:
            try:
                y_interp = np.interp(x_interp, x_unique, y_valid[unique_mask])
                result_df[y_col] = y_interp
            except Exception:
                result_df[y_col] = np.nan

    result_df = result_df.sort_values(by=x_col).reset_index(drop=True)
    return result_df


# ---------------------------------------------------------------------------
# Abnormality detection
# ---------------------------------------------------------------------------

def detect_abnormalities(series, threshold=3.0):
    """Detect abnormal points in a series using z-score threshold."""
    if len(series) < 2:
        return pd.Series(False, index=series.index), pd.Series(0, index=series.index)
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold, z_scores


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_to_common_time(df1, df2, freq=1.0):
    """Resample two dataframes to a common time base that spans the union of their ranges."""
    if 'timestamp_seconds' not in df1.columns or 'timestamp_seconds' not in df2.columns:
        st.error("timestamp_seconds column missing in one or both dataframes")
        return df1.copy(), df2.copy(), []

    try:
        df1_c = df1.copy().set_index('timestamp_seconds').sort_index()
        df2_c = df2.copy().set_index('timestamp_seconds').sort_index()

        if df1_c.empty and df2_c.empty:
            return df1.copy(), df2.copy(), []

        start = min(df1_c.index.min() if not df1_c.empty else np.inf,
                    df2_c.index.min() if not df2_c.empty else np.inf)
        end = max(df1_c.index.max() if not df1_c.empty else -np.inf,
                  df2_c.index.max() if not df2_c.empty else -np.inf)

        if start >= end or start == np.inf or end == -np.inf:
            return df1, df2, []

        common_time_index = pd.Index(np.arange(start, end, freq), name='timestamp_seconds')

        if len(common_time_index) == 0:
            return df1, df2, []

        df1_resampled = df1_c.reindex(df1_c.index.union(common_time_index)).interpolate(method='index').reindex(common_time_index)
        df2_resampled = df2_c.reindex(df2_c.index.union(common_time_index)).interpolate(method='index').reindex(common_time_index)

        return df1_resampled.reset_index(), df2_resampled.reset_index(), common_time_index.to_numpy()

    except Exception as e:
        st.error(f"Error during resampling: {str(e)}")
        return df1.copy(), df2.copy(), []


# ---------------------------------------------------------------------------
# Throttle aggregation
# ---------------------------------------------------------------------------

def _dominant_ramp_slice(throttle_series, mode="ramp_up", throttle_interval=5):
    """
    Find the strongest monotonic throttle run directly from the data.
    Returns a positional (start_row, end_row) tuple or None.
    """
    throttle = pd.to_numeric(throttle_series, errors="coerce")
    throttle = throttle.dropna().reset_index(drop=True)
    if len(throttle) < 2 or mode not in {"ramp_up", "ramp_down"}:
        return None

    window = max(1, min(11, len(throttle) // 40 * 2 + 1))
    if window > len(throttle):
        window = len(throttle) if len(throttle) % 2 == 1 else max(1, len(throttle) - 1)
    smoothed = throttle.rolling(window=window, center=True, min_periods=1).median()

    quant_step = max(float(throttle_interval) / 2.0, 1.0)
    quantized = (smoothed / quant_step).round() * quant_step
    same_tol = max(quant_step * 0.1, 0.25)

    plateau_values = [float(quantized.iloc[0])]
    plateau_starts = [0]
    plateau_ends = [0]
    for pos in range(1, len(quantized)):
        current_value = float(quantized.iloc[pos])
        if np.isclose(current_value, plateau_values[-1], atol=same_tol):
            plateau_ends[-1] = pos
        else:
            plateau_values.append(current_value)
            plateau_starts.append(pos)
            plateau_ends.append(pos)

    if len(plateau_values) < 2:
        return None

    target_sign = 1 if mode == "ramp_up" else -1
    delta_tol = max(quant_step * 0.25, 0.5)
    best_slice = None
    best_score = None
    run_start = None

    for diff_idx in range(len(plateau_values) - 1):
        delta = plateau_values[diff_idx + 1] - plateau_values[diff_idx]
        if abs(delta) <= delta_tol:
            sign = 0
        elif delta > 0:
            sign = 1
        else:
            sign = -1

        if sign == target_sign:
            if run_start is None:
                run_start = diff_idx
        else:
            if run_start is not None:
                run_end = diff_idx - 1
                start_row = plateau_starts[run_start]
                end_row = plateau_ends[run_end + 1]
                amplitude = abs(plateau_values[run_end + 1] - plateau_values[run_start])
                span = end_row - start_row + 1
                score = (amplitude, span)
                if best_score is None or score > best_score:
                    best_score = score
                    best_slice = (start_row, end_row)
                run_start = None

    if run_start is not None:
        run_end = len(plateau_values) - 2
        start_row = plateau_starts[run_start]
        end_row = plateau_ends[run_end + 1]
        amplitude = abs(plateau_values[run_end + 1] - plateau_values[run_start])
        span = end_row - start_row + 1
        score = (amplitude, span)
        if best_score is None or score > best_score:
            best_slice = (start_row, end_row)

    return best_slice


def filter_df_by_ramp_mode(
    df,
    throttle_col,
    current_col,
    mode="ramp_up",
    throttle_min=0,
    throttle_max=100,
    throttle_interval=5,
):
    """
    Apply the same ramp-mode filtering used by throttle aggregation.
    """
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()
    if throttle_col not in df.columns or current_col not in df.columns:
        return pd.DataFrame()

    filtered_df = df.copy()
    filtered_df[throttle_col] = pd.to_numeric(filtered_df[throttle_col], errors="coerce")
    filtered_df[current_col] = pd.to_numeric(filtered_df[current_col], errors="coerce")
    filtered_df = filtered_df[
        (filtered_df[throttle_col] > 0) & (filtered_df[current_col] > 0)
    ].copy()

    if filtered_df.empty:
        return filtered_df

    if mode in {"ramp_up", "ramp_down"}:
        ramp_slice = _dominant_ramp_slice(
            filtered_df[throttle_col],
            mode=mode,
            throttle_interval=throttle_interval,
        )
        if ramp_slice is not None:
            start_row, end_row = ramp_slice
            filtered_df = filtered_df.iloc[start_row : end_row + 1].copy()

    if filtered_df.empty:
        return filtered_df

    return filtered_df[
        (filtered_df[throttle_col] >= throttle_min)
        & (filtered_df[throttle_col] <= throttle_max)
    ].copy()


def process_throttle_aggregation(df, throttle_col, current_col, voltage_col, rpm1_col, rpm2_col,
                                 thrust_col, torque_col, motorpower_col, mode="ramp_up",
                                 throttle_min=0, throttle_max=100, throttle_interval=5):
    """
    Process raw motor test data into sorted performance table.
    """
    df = filter_df_by_ramp_mode(
        df,
        throttle_col,
        current_col,
        mode=mode,
        throttle_min=throttle_min,
        throttle_max=throttle_max,
        throttle_interval=throttle_interval,
    )

    if df.empty:
        return None

    # Compute per-row RPM
    if rpm1_col in df.columns and rpm2_col in df.columns:
        r1 = pd.to_numeric(df[rpm1_col], errors="coerce")
        r2 = pd.to_numeric(df[rpm2_col], errors="coerce")
        valid1 = r1.notna() & (r1 > 0)
        valid2 = r2.notna() & (r2 > 0)
        rpm_series = pd.Series(0.0, index=df.index)
        both_valid = valid1 & valid2
        rpm_series[both_valid] = (r1[both_valid] + r2[both_valid]) / 2.0
        only1 = valid1 & ~valid2
        rpm_series[only1] = r1[only1]
        only2 = valid2 & ~valid1
        rpm_series[only2] = r2[only2]
        df["RPM"] = rpm_series
    elif rpm1_col in df.columns:
        df["RPM"] = pd.to_numeric(df[rpm1_col], errors="coerce")
    elif rpm2_col in df.columns:
        df["RPM"] = pd.to_numeric(df[rpm2_col], errors="coerce")
    else:
        df["RPM"] = 0

    df["Electrical Power"] = (
        pd.to_numeric(df[voltage_col], errors="coerce") *
        pd.to_numeric(df[current_col], errors="coerce")
    )

    throttle_points = []
    current_throttle = throttle_min
    while current_throttle <= throttle_max:
        throttle_points.append(current_throttle)
        current_throttle += throttle_interval

    accx_col = find_column_by_pattern(df, ["AccX (g)", "AccX", "accx", "Acc X", "Acceleration X"])
    accy_col = find_column_by_pattern(df, ["AccY (g)", "AccY", "accy", "Acc Y", "Acceleration Y"])
    accz_col = find_column_by_pattern(df, ["AccZ (g)", "AccZ", "accz", "Acc Z", "Acceleration Z"])
    vibration_col = find_column_by_pattern(df, ["Vibration (g)", "Vibration - g", "Vibration RMS (g)", "Vibration"])
    motor_eff_col = find_column_by_pattern(df, ["Motor Efficiency (%)", "Motor Efficiency", "MotorEff", "Motor Eff", "MotorEfficiency"])
    prop_eff_col = find_column_by_pattern(df, ["Propeller Mech. Efficiency (gf/W)", "Propeller Mech Efficiency (gf/W)", "Propeller Mech. Efficiency", "Propeller Efficiency", "Propeller Mech Efficiency", "Prop Mech Eff"])

    grouped_data = []
    throttle_series_num = pd.to_numeric(df[throttle_col], errors="coerce")
    # Use a tolerance window so non-exact throttle values (e.g. 39.9/40.1)
    # still map to the intended step and the sorted table can be generated.
    throttle_tol = max(float(throttle_interval) * 0.5, 0.25)
    for throttle_point in throttle_points:
        mask = (throttle_series_num - float(throttle_point)).abs() <= throttle_tol
        throttle_data = df[mask].copy()

        if not throttle_data.empty:
            avg_data = {
                throttle_col: throttle_point,
                voltage_col: throttle_data[voltage_col].mean(),
                current_col: throttle_data[current_col].mean(),
                'RPM': throttle_data['RPM'].mean(),
                thrust_col: throttle_data[thrust_col].mean(),
                torque_col: throttle_data[torque_col].mean(),
                'Electrical Power - W': throttle_data['Electrical Power'].mean(),
            }

            if motorpower_col and motorpower_col in throttle_data.columns:
                avg_data['MotorPower - W'] = throttle_data[motorpower_col].mean()
            if accx_col and accx_col in throttle_data.columns:
                avg_data['AccX (g)'] = throttle_data[accx_col].mean()
            if accy_col and accy_col in throttle_data.columns:
                avg_data['AccY (g)'] = throttle_data[accy_col].mean()
            if accz_col and accz_col in throttle_data.columns:
                avg_data['AccZ (g)'] = throttle_data[accz_col].mean()
            if vibration_col and vibration_col in throttle_data.columns:
                avg_data['Vibration (g)'] = throttle_data[vibration_col].mean()
            if motor_eff_col and motor_eff_col in throttle_data.columns:
                avg_data['Motor Efficiency (%)'] = throttle_data[motor_eff_col].mean()
            if prop_eff_col and prop_eff_col in throttle_data.columns:
                avg_data['Propeller Mech. Efficiency (gf/W)'] = throttle_data[prop_eff_col].mean()

            grouped_data.append(avg_data)

    if not grouped_data:
        return None

    grouped = pd.DataFrame(grouped_data)

    grouped["SysEffect - gf/W"] = grouped[thrust_col] / grouped['Electrical Power - W'].replace(0, np.nan)

    # Round values
    round_map = {
        throttle_col: (0, int), voltage_col: (2, None), current_col: (2, None),
        "RPM": (0, int), thrust_col: (0, int), torque_col: (3, None),
        "SysEffect - gf/W": (2, None), "MotorPower - W": (2, None),
        "Electrical Power - W": (2, None),
        "AccX (g)": (3, None), "AccY (g)": (3, None), "AccZ (g)": (3, None),
        "Vibration (g)": (3, None), "Motor Efficiency (%)": (2, None),
        "Propeller Mech. Efficiency (gf/W)": (2, None),
    }
    for col, (decimals, cast_type) in round_map.items():
        if col in grouped.columns:
            grouped[col] = grouped[col].round(decimals)
            if cast_type:
                grouped[col] = grouped[col].astype(cast_type)

    final_cols = [throttle_col, voltage_col, current_col, "RPM",
                  thrust_col, torque_col, "SysEffect - gf/W"]

    if "MotorPower - W" in grouped.columns:
        final_cols.append("MotorPower - W")
    final_cols.append("Electrical Power - W")

    for extra in ["AccX (g)", "AccY (g)", "AccZ (g)", "Vibration (g)",
                   "Motor Efficiency (%)", "Propeller Mech. Efficiency (gf/W)"]:
        if extra in grouped.columns:
            final_cols.append(extra)

    final_cols = [col for col in final_cols if col in grouped.columns]
    grouped = grouped[final_cols]

    if throttle_col in grouped.columns:
        grouped = grouped.sort_values(by=throttle_col)

    return grouped


# ---------------------------------------------------------------------------
# Throttle regime detection
# ---------------------------------------------------------------------------

def detect_throttle_regimes_from_raw(
    df_raw: pd.DataFrame,
    throttle_col: str,
    timestamp_col: str,
    grouped: pd.DataFrame,
    throttle_interval: float,
):
    """
    Data-driven operating regime detection based on dwell time in throttle bands.
    """
    if timestamp_col not in df_raw.columns or throttle_col not in df_raw.columns:
        return None

    df = df_raw.copy()
    df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors="coerce")
    df[throttle_col] = pd.to_numeric(df[throttle_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col, throttle_col]).sort_values(timestamp_col)
    if df.empty:
        return None

    dt = df[timestamp_col].diff().fillna(0.0)
    dt[dt < 0] = 0.0
    df["_dt"] = dt

    throttle_values = sorted(pd.to_numeric(grouped[throttle_col].dropna(), errors="coerce").unique().tolist())
    if not throttle_values:
        return None

    dwell = {}
    for t in throttle_values:
        mask = df[throttle_col] == t
        dwell[t] = float(df.loc[mask, "_dt"].sum())

    total_dwell = sum(dwell.values())
    if total_dwell <= 0:
        return None

    dwell_frac = {t: d / total_dwell for t, d in dwell.items()}
    max_frac = max(dwell_frac.values())

    labels = {t: "other" for t in throttle_values}

    dominant = [t for t, f in dwell_frac.items() if f >= 0.5 * max_frac]
    if dominant:
        cruise_t = min(dominant)
        labels[cruise_t] = "cruise"

    max_throttle = max(throttle_values)
    high_threshold = 0.8 * max_throttle
    for t in throttle_values:
        if t >= high_threshold and dwell_frac[t] <= 0.5 * max_frac:
            labels[t] = "high_load"

    return {
        "throttle_col": throttle_col,
        "timestamp_col": timestamp_col,
        "throttle_values": throttle_values,
        "throttle_interval": float(throttle_interval),
        "dwell": dwell,
        "dwell_frac": dwell_frac,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Streamlit table sanitization
# ---------------------------------------------------------------------------

def sanitize_table_for_streamlit(table_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Make a dataframe safe to send to st.dataframe.
    """
    if table_df is None:
        return None

    df_safe = table_df.copy()
    df_safe = df_safe.replace([np.inf, -np.inf], np.nan)

    def _coerce_cell(x):
        if x is None or isinstance(x, (int, float, bool, str)):
            return x
        return str(x)

    df_safe = df_safe.applymap(_coerce_cell)

    new_cols: list[str] = []
    seen: dict[str, int] = {}
    for col in df_safe.columns:
        name = str(col)
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        new_cols.append(name)
    df_safe.columns = new_cols

    return df_safe


# ---------------------------------------------------------------------------
# Cached wrappers for expensive functions
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=20)
def cached_smooth_data(df_hash: str, _df: pd.DataFrame, x_col: str,
                       y_cols: list, method: str = 'linear',
                       num_points: int | None = None,
                       smoothing_window: int = 5) -> pd.DataFrame:
    """Cached wrapper around smooth_data_for_plotting."""
    return smooth_data_for_plotting(_df, x_col, y_cols, method, num_points, smoothing_window)


@st.cache_data(show_spinner=False, max_entries=10)
def cached_throttle_aggregation(
    df_hash: str, _df: pd.DataFrame,
    throttle_col: str, current_col: str, voltage_col: str,
    rpm1_col: str, rpm2_col: str, thrust_col: str, torque_col: str,
    motorpower_col: str, mode: str = "ramp_up",
    throttle_min: int = 0, throttle_max: int = 100,
    throttle_interval: int = 5,
) -> pd.DataFrame | None:
    """Cached wrapper around process_throttle_aggregation."""
    return process_throttle_aggregation(
        _df, throttle_col, current_col, voltage_col, rpm1_col, rpm2_col,
        thrust_col, torque_col, motorpower_col, mode,
        throttle_min, throttle_max, throttle_interval,
    )
