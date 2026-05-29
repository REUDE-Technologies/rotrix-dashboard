#type: ignore
"""
CSV / ULog / Excel file loading, header detection, data cleaning, and
timestamp conversion utilities.
"""
import re
import os
import hashlib
import tempfile

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pyulog import ULog

from config import TOPIC_ASSESSMENT_PAIRS


# Hard limits to protect against extremely large uploads
MAX_ROWS = 500_000
MAX_COLS = 200
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

def clean_file_info_text(raw_text: str) -> str:
    """
    Clean pre-header / file info text for display in the UI and report.
    - Collapse runs of commas into a single space (removes ",,,,,,")
    - Optionally remove stray commas that are not between words/numbers
    - Collapse multiple spaces into one
    - Strip leading/trailing whitespace and drop empty lines
    """
    try:
        raw = raw_text or ""
        cleaned_lines = []
        for raw_line in raw.splitlines():
            line = (raw_line or "").rstrip()
            if not line:
                continue
            if "www.lyfh-tj.com" in line:
                continue
            line = line.replace("\uFF1A", " : ")
            for char in ("\u25A0", "\u220E", "\u25AA", "\u25AB"):
                line = line.replace(char, " : ")
            line = re.sub(r",+", " ", line)
            line = re.sub(r"\s,\s", " ", line)
            line = re.sub(r"\s{2,}", " ", line)
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)
    except Exception:
        return raw_text or ""


def parse_file_info_to_table(info_text: str):
    """
    Parse file info text into key-value rows for table display.
    Returns a list of (key, value) tuples.
    """
    rows = []
    for raw_line in (info_text or "").splitlines():
        line_raw = re.sub(r",", " ", (raw_line or "").strip())
        line = re.sub(r"\s{2,}", " ", line_raw).strip()
        if not line:
            continue
        match = re.match(r"^([A-Za-z][^:]{0,120}?):\s*(.*)$", line)
        if match:
            key, value = match.group(1).strip(), match.group(2).strip()
            if key or value:
                rows.append((key, value))
        else:
            # Excel header rows are often extracted as "Key  Value" (two+ spaces)
            # without a colon. Parse that form into key/value pairs.
            spaced_match = re.match(r"^(.+?)\s{2,}(.+)$", line_raw)
            if spaced_match:
                key = re.sub(r"\s{2,}", " ", spaced_match.group(1)).strip()
                value = re.sub(r"\s{2,}", " ", spaced_match.group(2)).strip()
                if key or value:
                    rows.append((key, value))
                continue
            value = line.strip("-").strip()
            if value.startswith(" : "):
                value = value[3:].strip()
            if not value:
                continue
            if re.match(r"^\d+$", value) and rows and rows[-1][0]:
                prev_key, prev_val = rows[-1]
                rows[-1] = (prev_key, (prev_val or "") + ":" + value)
            else:
                rows.append(("", value))
    return rows


def extract_test_type_from_info(info_text: str, filename: str = "") -> str:
    """
    Detect a RotriX test type token from file-info text and/or filename.

    Priority:
    1) Filename token like ``_UAT001`` / ``_UAT002``.
    2) Explicit file-info keys such as ``Auto Test Type`` or ``Test Type``.
    3) Combined fallback from ``Test Mode`` + ``Auto Test Type``.
    """
    def _norm_token(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (text or "").strip().lower())

    # Filename pattern support: RotriX_..._UAT001.xlsx -> UAT001
    if filename:
        m = re.search(r"_(UAT\d{3,})\b", str(filename), flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    rows = parse_file_info_to_table(info_text or "")
    if not rows:
        return ""

    kv: dict[str, str] = {}
    for key, value in rows:
        k = _norm_token(key)
        v = (value or "").strip()
        if k and v:
            kv[k] = v

    # Direct key-based extraction
    for k in ("autotesttype", "testtype"):
        val = kv.get(k, "")
        if val:
            # If value itself contains UATxxx, normalize to token.
            m = re.search(r"\b(UAT\d{3,})\b", val, flags=re.IGNORECASE)
            if m:
                return m.group(1).upper()
            return val

    # Combined fallback if explicit test type is missing.
    test_mode = kv.get("testmode", "")
    auto_type = kv.get("autotesttype", "")
    if test_mode or auto_type:
        combo = " ".join([x for x in [test_mode, auto_type] if x]).strip()
        return combo

    return ""


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def fix_duplicate_columns(df):
    """Fix duplicate column names in a DataFrame by appending suffixes."""
    if df.columns.duplicated().any():
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            dup_indices = cols[cols == dup].index.values.tolist()
            cols[dup_indices] = [dup if x == 0 else f"{dup}_{x}" for x in range(len(dup_indices))]
        df.columns = cols
    return df


def convert_to_numeric_safe(series):
    """
    Safely convert a series to numeric, handling various formats.
    Returns the converted series with non-numeric values as NaN.
    """
    if series.dtype in [np.int64, np.float64, np.int32, np.float32]:
        return series
    return pd.to_numeric(series, errors='coerce')


def filter_info_rows(df):
    """
    Filter out rows that contain mostly "None" values or string information rows.
    Also converts columns that should be numeric to proper numeric types.
    """
    if df is None or df.empty:
        return df

    df_filtered = df.copy()
    df_filtered = df_filtered.replace(['None', 'none', 'NONE', 'null', 'Null', 'NULL'], np.nan)

    numeric_patterns = [
        'throttle', 'voltage', 'current', 'rpm', 'thrust', 'torque',
        'power', 'effect', 'speed', 'time', 'timestamp', 'index',
        'vol v', 'cur a', 'thrust gf', 'torque nm', 'rpm1 rpm',
        'sys', 'motor', 'electrical', 'mechanical', 'propulsion',
        'propeller', 'efficiency', 'pressure', 'ambient', 'winding',
        'phase', 'baring',
    ]

    potential_numeric_cols = []
    for col in df_filtered.columns:
        col_lower = str(col).lower()
        if col in ['Index', 'index'] or 'timestamp' in col_lower or col_lower == 'time':
            continue
        if any(pattern in col_lower for pattern in numeric_patterns):
            if df_filtered[col].dtype == 'object':
                potential_numeric_cols.append(col)

    for col in potential_numeric_cols:
        df_filtered[col] = convert_to_numeric_safe(df_filtered[col])

    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        valid_numeric_count = df_filtered[numeric_cols].notna().sum(axis=1)
        total_numeric_cols = len(numeric_cols)
        threshold = max(1, total_numeric_cols * 0.5)
        df_filtered = df_filtered[valid_numeric_count >= threshold].copy()

    if len(df_filtered.columns) > 0:
        total_cols = len(df_filtered.columns)
        valid_count = df_filtered.notna().sum(axis=1)
        threshold = max(1, total_cols * 0.3)
        df_filtered = df_filtered[valid_count >= threshold].copy()

    for col in df_filtered.columns:
        col_lower = str(col).lower()
        if col in ['Index', 'index'] or 'timestamp' in col_lower or col_lower == 'time':
            continue
        if df_filtered[col].dtype == 'object':
            converted = pd.to_numeric(df_filtered[col], errors='coerce')
            if converted.notna().sum() > len(df_filtered) * 0.5:
                df_filtered[col] = converted

    df_filtered = df_filtered.reset_index(drop=True)
    return df_filtered


def _apply_size_guards(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Enforce global row/column limits on loaded dataframes to avoid
    unbounded memory growth when users upload very large files.
    """
    if df is None or df.empty:
        return df

    try:
        rows, cols = df.shape
    except Exception:
        return df

    if rows > MAX_ROWS:
        st.warning(
            f"File truncated to {MAX_ROWS:,} rows "
            f"(original: {rows:,}). Consider pre-filtering large datasets."
        )
        df = df.head(MAX_ROWS)

    if cols > MAX_COLS:
        df = df.iloc[:, :MAX_COLS]

    return df


# ---------------------------------------------------------------------------
# Timestamp utilities
# ---------------------------------------------------------------------------

def format_seconds_to_mmss(seconds):
    """Format seconds to MM:SS format."""
    try:
        minutes = int(float(seconds) // 60)
        remaining_seconds = int(float(seconds) % 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"
    except Exception:
        return "00:00"


def mmss_to_seconds(mmss_str):
    """Convert common time strings to seconds.

    Supported formats:
    - MM:SS
    - MM:SS.s
    - HH:MM:SS
    - HH:MM:SS AM/PM
    """
    try:
        if mmss_str is None:
            return 0.0

        raw = str(mmss_str).strip()
        if not raw:
            return 0.0

        # Prefer explicit datetime parsing first for 12-hour and 24-hour clock forms.
        for fmt in ("%I:%M:%S %p", "%H:%M:%S"):
            try:
                dt = datetime.strptime(raw, fmt)
                return float(dt.hour * 3600 + dt.minute * 60 + dt.second)
            except ValueError:
                pass

        if ":" in raw:
            parts = [p.strip() for p in raw.split(":")]
            if len(parts) == 2:
                minutes, seconds = parts
                return int(minutes) * 60 + float(seconds)
            if len(parts) == 3:
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

        return float(raw)
    except Exception:
        return 0.0


def seconds_to_mmss(seconds):
    """Convert seconds to MM:SS format."""
    try:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"
    except Exception:
        return "00:00"


def get_tick_spacing(data_range):
    """Get appropriate tick spacing based on data range."""
    if data_range <= 10:
        return 1
    elif data_range <= 60:
        return 10
    elif data_range <= 300:
        return 30
    elif data_range <= 600:
        return 60
    else:
        return 120


def get_timestamp_ticks(data):
    """Generate evenly spaced timestamp ticks."""
    if data is None or len(data) == 0:
        return [], []
    try:
        data_min = float(data.min())
        data_max = float(data.max())
        data_range = data_max - data_min
        spacing = get_tick_spacing(data_range)
        ticks = np.arange(data_min, data_max + spacing, spacing)
        return ticks, [format_seconds_to_mmss(float(t)) for t in ticks]
    except Exception:
        return [], []


def _hhmmss_ms_to_elapsed(series):
    """Convert HH:MM:SS or HH:MM:SS.mmm strings to elapsed seconds from the first value."""
    def _parse(ts):
        try:
            ts = str(ts).strip()
            parts = ts.split(":")
            if len(parts) == 3:
                h, m = int(parts[0]), int(parts[1])
                s = float(parts[2])
                return h * 3600 + m * 60 + s
            if len(parts) == 2:
                m = int(parts[0])
                s = float(parts[1])
                return m * 60 + s
        except Exception:
            pass
        return None

    parsed = series.apply(_parse)
    if parsed.notna().any():
        return _to_monotonic_elapsed(parsed)
    return None


def _to_monotonic_elapsed(series):
    """
    Convert a numeric second-series into monotonic elapsed seconds.

    Handles time-of-day wrap-around (e.g., 23:59:59 -> 00:00:02) and test-loop
    resets (e.g., 600 -> 0), preserving continuity for filtering/plotting.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        return s

    vals = s.to_numpy(dtype=float)
    out = vals.copy()
    offset = 0.0
    prev = None

    for i, cur in enumerate(vals):
        if np.isnan(cur):
            out[i] = np.nan
            continue
        if prev is not None and not np.isnan(prev):
            # Detect a reset/drop with tolerance for tiny jitter.
            drop = prev - cur
            reset_threshold = max(5.0, abs(prev) * 0.1)
            if drop > reset_threshold:
                # If previous value is hour-scale and current near zero,
                # this is likely a clock wrap at midnight.
                if prev >= 3600 and cur <= 3600:
                    offset += 24 * 3600
                else:
                    # Generic loop reset (e.g., run restarts at 0).
                    offset += prev
        out[i] = cur + offset
        prev = cur

    out_s = pd.Series(out, index=s.index)
    first_valid = out_s.dropna().iloc[0] if not out_s.dropna().empty else 0.0
    return out_s - first_valid


def ensure_seconds_column(df):
    """Ensure timestamp_seconds column exists in the dataframe."""
    if 'timestamp_seconds' not in df.columns:
        # Check for 'Timestamp' column with HH:MM:SS.mmm format (e.g. wingflyingtech bench)
        ts_col = None
        for col in df.columns:
            if str(col).strip().lower() == 'timestamp':
                ts_col = col
                break

        # Check for a numeric elapsed-seconds column (e.g. "Time (s)", "Time (secs)",
        # "time_s"). Many bench exports (RotriX, EY DELTA sample data, etc.) use this
        # form and we must use it directly so that summary runtime and dwell-time
        # calculations reflect the real test duration instead of a 0..N row index.
        seconds_col = None
        for col in df.columns:
            cl = str(col).strip().lower()
            if cl in ('time (s)', 'time (secs)', 'time_s', 'time_seconds', 'elapsed (s)', 'elapsed time (s)'):
                seconds_col = col
                break

        if seconds_col is not None:
            raw_ts = pd.to_numeric(df[seconds_col], errors='coerce')
            if raw_ts.notna().any():
                df['timestamp_seconds'] = _to_monotonic_elapsed(raw_ts)
            else:
                df['timestamp_seconds'] = range(len(df))
        elif ts_col is not None and not pd.api.types.is_numeric_dtype(df[ts_col]):
            elapsed = _hhmmss_ms_to_elapsed(df[ts_col])
            if elapsed is not None:
                df['timestamp_seconds'] = elapsed
            else:
                df['timestamp_seconds'] = range(len(df))
        elif 'timestamp' in df.columns:
            raw_ts = pd.to_numeric(df['timestamp'], errors='coerce') / 1e6
            df['timestamp_seconds'] = _to_monotonic_elapsed(raw_ts)
        elif 'Timestamp (hh:mm:ss)' in df.columns:
            raw_ts = df['Timestamp (hh:mm:ss)'].apply(lambda x: mmss_to_seconds(str(x)))
            df['timestamp_seconds'] = _to_monotonic_elapsed(raw_ts)
        elif 'Time' in df.columns:
            raw_ts = df['Time'].apply(lambda x: mmss_to_seconds(str(x)))
            df['timestamp_seconds'] = _to_monotonic_elapsed(raw_ts)
        else:
            df['timestamp_seconds'] = range(len(df))

    if 'timestamp_seconds' in df.columns:
        if df['timestamp_seconds'].isna().all() or df['timestamp_seconds'].isnull().all():
            df = df.drop('timestamp_seconds', axis=1)

    return df


def add_hhmmss_seconds_column(df, timestamp_col='Timestamp (hh:mm:ss)'):
    def hhmmss_to_seconds(ts):
        try:
            if pd.isnull(ts):
                return None
            ts = str(ts).strip()
            try:
                dt = datetime.strptime(ts, "%I:%M:%S %p")
            except ValueError:
                try:
                    dt = datetime.strptime(ts, "%H:%M:%S")
                except ValueError:
                    return None
            return dt.hour * 3600 + dt.minute * 60 + dt.second
        except Exception:
            return None

    abs_seconds = df[timestamp_col].apply(hhmmss_to_seconds)
    if not abs_seconds.isnull().all():
        elapsed_seconds = _to_monotonic_elapsed(abs_seconds)
        df['timestamp_seconds'] = elapsed_seconds
    else:
        df['timestamp_seconds'] = abs_seconds
    return df


def convert_timestamps_to_seconds(df):
    """Convert timestamp columns to seconds."""
    if df is None:
        return df
    if isinstance(df, pd.DataFrame) and len(df.index) > 0:
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
        for col in timestamp_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col] / 1000000
    return df


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv(file):
    """
    Load CSV data.
    - Detect Rotrix-style files by header content and use the original Rotrix logic.
    - For all other files, use the robust generic logic.
    """
    if isinstance(file, str):
        with open(file, "r", encoding="utf-8") as f:
            lines = [next(f, "") for _ in range(20)]
        file_obj = file
    else:
        file.seek(0)
        first = file.readline()
        file.seek(0)
        lines = []
        for _ in range(20):
            line = file.readline()
            if not line:
                break
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="ignore")
            lines.append(line)
        file.seek(0)
        file_obj = file

    rotrix_keywords = ["TestrecordId", "Timestamp (hh:mm:ss)", "Speedmms"]
    is_rotrix = any(
        any(key in (line or "") for key in rotrix_keywords) for line in lines
    )

    if is_rotrix:
        header_row = None
        for i, line in enumerate(lines):
            if any(key in (line or "") for key in rotrix_keywords):
                header_row = i
                break
        if header_row is None:
            header_row = 4

        try:
            raw_info = "".join(lines[:header_row])
            info_text = clean_file_info_text(raw_info)
            st.session_state.report_file_info_text = info_text
        except Exception:
            pass

        try:
            df = pd.read_csv(
                file_obj,
                encoding="utf-8",
                skiprows=header_row,
                header=0,
            )
            return _apply_size_guards(df)
        except Exception:
            if not isinstance(file, str):
                file.seek(0)
            df = pd.read_csv(
                file_obj,
                encoding="utf-8-sig",
                skiprows=header_row,
                header=0,
            )
            return _apply_size_guards(df)

    # Check for test bench format
    core_test_bench_keywords = ["throttle", "voltage", "current", "thrust", "torque"]

    prioritized_header_row = None
    for i, line in enumerate(lines):
        if not line:
            continue
        line_lower = (line or "").lower()
        keyword_count = sum(1 for keyword in core_test_bench_keywords if keyword in line_lower)
        if keyword_count >= 2 and line.count(",") >= 2:
            prioritized_header_row = i
            break

    test_bench_patterns = ["LY-30KGF", "UAV Power System", "Test Bench"]
    has_test_bench_title = any(
        any(pattern in (line or "") for pattern in test_bench_patterns) for line in lines[:5]
    )

    recordid_header_row = None
    for i, line in enumerate(lines):
        if line and "RecordID" in line and line.count(",") >= 2:
            recordid_header_row = i
            break

    if has_test_bench_title and recordid_header_row is not None:
        header_row = recordid_header_row
    elif prioritized_header_row is not None:
        header_row = prioritized_header_row
    else:
        header_row = None
        for i, line in enumerate(lines):
            line_lower = (line or "").lower()
            if any(
                keyword in line_lower
                for keyword in [
                    "testrecordid", "recordid", "timestamp", "time",
                    "throttle", "rpm", "thrust", "torque", "voltage",
                    "current", "speedmms", "vol v", "cur a",
                    "thrust gf", "torque nm", "rpm1 rpm",
                ]
            ):
                if line.count(",") >= 2:
                    header_row = i
                    break

        if header_row is None:
            for skip in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]:
                try:
                    if not isinstance(file, str):
                        file_obj.seek(0)
                    test_df = pd.read_csv(
                        file_obj, encoding="utf-8", skiprows=skip, header=0, nrows=1
                    )
                    if len(test_df.columns) >= 3:
                        header_row = skip
                        break
                except Exception:
                    continue

        if header_row is None:
            header_row = 0

    # Load with error handling
    try:
        if not isinstance(file, str):
            file_obj.seek(0)

        try:
            raw_info = "".join(lines[:header_row])
            info_text = clean_file_info_text(raw_info)
            st.session_state.report_file_info_text = info_text
        except Exception:
            pass

        df = pd.read_csv(
            file_obj,
            encoding="utf-8",
            skiprows=header_row,
            header=0,
            on_bad_lines="skip",
        )
        df = df.dropna(how="all").dropna(axis=1, how="all")
        df = fix_duplicate_columns(df)
        df = filter_info_rows(df)
        return _apply_size_guards(df)
    except Exception:
        try:
            if not isinstance(file, str):
                file_obj.seek(0)
            df = pd.read_csv(
                file_obj,
                encoding="utf-8-sig",
                skiprows=header_row,
                header=0,
                on_bad_lines="skip",
            )
            df = df.dropna(how="all").dropna(axis=1, how="all")
            df = fix_duplicate_columns(df)
            df = filter_info_rows(df)
            return _apply_size_guards(df)
        except Exception:
            try:
                if not isinstance(file, str):
                    file_obj.seek(0)
                df = pd.read_csv(
                    file_obj,
                    encoding="utf-8",
                    header=0,
                    on_bad_lines="skip",
                )
                df = df.dropna(how="all").dropna(axis=1, how="all")
                df = fix_duplicate_columns(df)
                df = filter_info_rows(df)
                return _apply_size_guards(df)
            except Exception as e3:
                raise Exception(f"Failed to parse CSV: {str(e3)}")


# ---------------------------------------------------------------------------
# ULog loading
# ---------------------------------------------------------------------------

def load_ulog(file, key_suffix=""):
    ALLOWED_TOPICS = set(t for t, _ in TOPIC_ASSESSMENT_PAIRS)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.ulg') as tmp_file:
        try:
            if isinstance(file, str):
                with open(file, 'rb') as f:
                    tmp_file.write(f.read())
            else:
                file.seek(0)
                tmp_file.write(file.read())
            tmp_file.flush()

            ulog = ULog(tmp_file.name)
            if not ulog.data_list:
                st.warning("⚠️ No data found in the ULog file")
                return {}, []

            extracted_dfs = {}
            for msg in ulog.data_list:
                if msg.data:
                    df = pd.DataFrame(msg.data)
                    if not df.empty:
                        extracted_dfs[msg.name] = df

            filtered_dfs = {topic: df for topic, df in extracted_dfs.items() if topic in ALLOWED_TOPICS}
            if not filtered_dfs:
                st.warning("⚠️ No extractable topics found in ULog file")
                return {}, []

            topic_names = ["None"] + list(filtered_dfs.keys())
            return filtered_dfs, topic_names

        except Exception as e:
            st.error(f"Error processing ULog file: {str(e)}")
            return {}, []
        finally:
            try:
                os.unlink(tmp_file.name)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Generic loader
# ---------------------------------------------------------------------------

def _load_excel_with_header_detection(file):
    """Load Excel files with automatic header detection.

    Handles workbooks where:
    - test data starts after metadata rows,
    - powertrain/file-info lives in a separate sheet,
    - metric/imperial variants may exist in different sheets or headers.
    """
    header_keywords = {
        "throttle", "voltage", "current", "rpm", "thrust", "torque",
        "timestamp", "power", "electrical", "mechanical", "propulsion",
        "propeller", "efficiency", "pressure", "ambient",
    }

    def _normalize_row_cells(row) -> list[str]:
        return [str(v).strip() for v in row if pd.notna(v) and str(v).strip()]

    def _detect_header_row(probe_df: pd.DataFrame) -> tuple[int | None, int]:
        best_header = None
        best_hits = -1
        for i, row in probe_df.iterrows():
            cells = _normalize_row_cells(row.tolist())
            if len(cells) < 3:
                continue
            row_text = " ".join(cells).lower()
            hits = sum(1 for kw in header_keywords if kw in row_text)
            if hits >= 3:
                if hits > best_hits:
                    best_hits = hits
                    best_header = i
                    if hits >= 6:
                        break
        return best_header, best_hits

    def _extract_metadata_lines(probe_df: pd.DataFrame, stop_row: int | None = None) -> list[str]:
        lines: list[str] = []
        max_rows = len(probe_df) if stop_row is None else min(stop_row, len(probe_df))
        for i in range(max_rows):
            cells = _normalize_row_cells(probe_df.iloc[i].tolist())
            if not cells:
                continue
            # Typical metadata rows are key/value (2 columns) or a compact
            # descriptive line in the first column.
            if len(cells) >= 2:
                key = cells[0]
                value = " ".join(cells[1:])
                lines.append(f"{key} : {value}")
            else:
                lines.append(cells[0])
        return lines

    xls = pd.ExcelFile(file)
    best_sheet = None
    best_probe = None
    best_header_row = None
    best_score = (-1, -1)

    for sheet_name in xls.sheet_names:
        probe = pd.read_excel(file, sheet_name=sheet_name, header=None, nrows=60)
        header_row, hits = _detect_header_row(probe)
        name_lower = str(sheet_name).lower()
        # Prefer likely data sheets on score ties.
        name_bonus = 1 if any(token in name_lower for token in ("test", "report", "data", "log")) else 0
        score = (hits, name_bonus)
        if header_row is not None and score > best_score:
            best_sheet = sheet_name
            best_probe = probe
            best_header_row = header_row
            best_score = score

    if best_sheet is None:
        # Fallback: first sheet with default header parsing.
        df = pd.read_excel(file, sheet_name=xls.sheet_names[0] if xls.sheet_names else 0)
    else:
        # 1) Metadata before the data header in the selected data sheet.
        info_parts: list[str] = []
        if best_probe is not None and best_header_row is not None:
            info_parts.extend(_extract_metadata_lines(best_probe, stop_row=best_header_row))

        # 2) Metadata from non-data sheets (e.g. Powertrain info).
        for sheet_name in xls.sheet_names:
            if sheet_name == best_sheet:
                continue
            try:
                probe_other = pd.read_excel(file, sheet_name=sheet_name, header=None, nrows=80)
                header_row_other, hits_other = _detect_header_row(probe_other)
                # If this sheet also looks like data, skip metadata extraction.
                if header_row_other is not None and hits_other >= 3:
                    continue
                info_parts.extend(_extract_metadata_lines(probe_other))
            except Exception:
                continue

        if info_parts:
            try:
                info_text = clean_file_info_text("\n".join(info_parts))
                st.session_state.report_file_info_text = info_text
            except Exception:
                pass

        df = pd.read_excel(file, sheet_name=best_sheet, header=best_header_row)

    if df is not None and not df.empty:
        df = df.dropna(how="all").dropna(axis=1, how="all")
        df = fix_duplicate_columns(df)
        df = filter_info_rows(df)
    return df


def load_data(file, filetype, key_suffix):
    """Load data from various file types."""
    try:
        if filetype == ".csv":
            df = load_csv(file)
            if df is None:
                st.error("Failed to load CSV file: load_csv returned None")
                return None, None
        elif filetype == ".xlsx":
            df = _load_excel_with_header_detection(file)
            if df is not None and not df.empty:
                df = _apply_size_guards(df)
        else:
            st.error(f"Unsupported file type: {filetype}")
            return None, None

        if df is not None and not df.empty:
            df = normalize_rotrix_columns(df)
            df = ensure_seconds_column(df)
            return df, None
        else:
            if df is None:
                st.error("No data found in file: DataFrame is None")
            else:
                st.error(f"No data found in file: DataFrame is empty (shape: {df.shape})")
            return None, None
    except Exception as e:
        import traceback
        st.error(f"Error loading file: {str(e)}")
        with st.expander("Show error details"):
            st.code(traceback.format_exc())
        return None, None


def normalize_rotrix_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common RotriX exports to canonical dashboard columns/units.

    Adds derived aliases when possible:
    - Throttle (%)            -> Throttle - %
    - Rotational Speed (RPM)  -> RPM
    - Thrust (lb) / kgf       -> Thrust - gf
    - Torque (lb-in)          -> Torque - N*m
    - Mechanical Power (HP)   -> MotorPower - W
    - Overall/Prop Efficiency (lb/HP) -> SysEffect - gf/W
    """
    if df is None or df.empty:
        return df

    cols = [str(c).strip() for c in df.columns]
    lower = [c.lower() for c in cols]
    rotrix_signals = (
        "testrecordid",
        "timestamp (hh:mm:ss)",
        "throttle (%)",
        "rotational speed (rpm)",
    )
    is_rotrix_like = sum(1 for sig in rotrix_signals if sig in lower) >= 2

    out = df.copy()
    out = fix_duplicate_columns(out)

    def _find_col(candidates):
        cand_l = [str(x).strip().lower() for x in candidates]
        for c in out.columns:
            cc = str(c).strip().lower()
            if cc in cand_l:
                return c
        for c in out.columns:
            cc = str(c).strip().lower()
            for t in cand_l:
                if t in cc:
                    return c
        return None

    def _to_num(col):
        return pd.to_numeric(out[col], errors="coerce") if col in out.columns else None

    def _set_alias_if_missing(dest_col, source_candidates):
        if dest_col in out.columns:
            return
        src = _find_col(source_candidates)
        if src is not None:
            out[dest_col] = _to_num(src)

    # Basic aliases for RotriX and other similar logger outputs.
    _set_alias_if_missing("Throttle - %", ["Throttle (%)", "Throttle", "throttle"])
    _set_alias_if_missing("RPM", ["Rotational Speed (RPM)", "RPM1 - RPM", "RPM"])
    _set_alias_if_missing("Vol - V", ["Voltage (V)", "Voltage", "Voltage_V"])
    _set_alias_if_missing("Cur - A", ["Current (A)", "Current", "Current_A"])
    _set_alias_if_missing("Electrical Power - W", ["Electrical Power (W)", "Electrical Power - W", "Power_W", "Power (W)"])
    _set_alias_if_missing("SysEffect - gf/W", ["Overall Efficiency (gf/W)", "SysEffect (gf/W)", "SysEffect - gf/W", "Efficiency_g_per_W"])
    _set_alias_if_missing("AccX (g)", ["AccX (g)", "AccX", "X_g"])
    _set_alias_if_missing("AccY (g)", ["AccY (g)", "AccY", "Y_g"])
    _set_alias_if_missing("Vibration (g)", ["Vibration (g)", "Vibration RMS (g)", "Vibration"])

    # Derive throttle from PWM when no explicit throttle exists.
    if "Throttle - %" not in out.columns:
        pwm_col = _find_col(["PWM", "PWM Value"])
        if pwm_col is not None:
            pwm = _to_num(pwm_col)
            if pwm is not None and pwm.notna().any():
                # Auto-detect PWM style: 1000-2000 pulse width or 0-100 percentage.
                if pwm.between(800, 2200).mean() >= 0.6:
                    out["Throttle - %"] = ((pwm - 1000.0) / 1000.0 * 100.0).clip(0.0, 100.0)
                else:
                    out["Throttle - %"] = pwm.clip(0.0, 100.0)

    # If file is not RotriX-like, still keep universal aliases and return.
    if not is_rotrix_like:
        if "Thrust - gf" not in out.columns:
            _set_alias_if_missing("Thrust - gf", ["Thrust (gf)", "Thrust - gf", "Weight_grams", "Weight (g)"])
        if "Torque - N*m" not in out.columns:
            _set_alias_if_missing("Torque - N*m", ["Torque (N·m)", "Torque (N*m)", "Torque", "torque"])
        if "MotorPower - W" not in out.columns:
            _set_alias_if_missing("MotorPower - W", ["MotorPower - W", "Mechanical Power (W)", "Electrical Power - W", "Power_W"])
        return out

    # Thrust normalization -> gf
    if "Thrust - gf" not in out.columns:
        thrust_lb = _find_col(["Thrust (lb)"])
        thrust_kgf = _find_col(["Thrust (kgf)"])
        thrust_gf = _find_col(["Thrust (gf)", "Thrust - gf"])
        thrust_any = _find_col(["Thrust"])
        if thrust_lb is not None:
            out["Thrust - gf"] = _to_num(thrust_lb) * 453.59237
        elif thrust_kgf is not None:
            out["Thrust - gf"] = _to_num(thrust_kgf) * 1000.0
        elif thrust_gf is not None:
            out["Thrust - gf"] = _to_num(thrust_gf)
        elif thrust_any is not None:
            out["Thrust - gf"] = _to_num(thrust_any)

    # Torque normalization -> N*m
    if "Torque - N*m" not in out.columns:
        torque_lbin = _find_col(["Torque (lb-in)"])
        torque_nm = _find_col(["Torque (N·m)", "Torque (N*m)", "Torque (Nm)", "Torque - N*m"])
        torque_any = _find_col(["Torque"])
        if torque_lbin is not None:
            out["Torque - N*m"] = _to_num(torque_lbin) * 0.112984829
        elif torque_nm is not None:
            out["Torque - N*m"] = _to_num(torque_nm)
        elif torque_any is not None:
            out["Torque - N*m"] = _to_num(torque_any)

    # Mechanical/Electrical power normalization -> W
    if "MotorPower - W" not in out.columns:
        mech_hp = _find_col(["Mechanical Power (HP)"])
        mech_w = _find_col(["Mechanical Power (W)", "Mechanical (W)"])
        elec_w = _find_col(["Electrical Power (W)", "Electrical Power - W", "InPower - W"])
        if mech_hp is not None:
            out["MotorPower - W"] = _to_num(mech_hp) * 745.699872
        elif mech_w is not None:
            out["MotorPower - W"] = _to_num(mech_w)
        elif elec_w is not None:
            out["MotorPower - W"] = _to_num(elec_w)

    # Efficiency normalization -> gf/W
    if "SysEffect - gf/W" not in out.columns:
        sys_lb_hp = _find_col(["Overall System Efficiency (lb/HP)"])
        prop_lb_hp = _find_col(["Propeller Efficiency (lb/HP)", "Propeller Mech. Efficiency (lb/HP)"])
        sys_gf_w = _find_col(["Overall Efficiency (gf/W)", "SysEffect (gf/W)", "SysEffect - gf/W"])
        conv_lb_hp_to_gf_w = 453.59237 / 745.699872
        if sys_lb_hp is not None:
            out["SysEffect - gf/W"] = _to_num(sys_lb_hp) * conv_lb_hp_to_gf_w
        elif prop_lb_hp is not None:
            out["SysEffect - gf/W"] = _to_num(prop_lb_hp) * conv_lb_hp_to_gf_w
        elif sys_gf_w is not None:
            out["SysEffect - gf/W"] = _to_num(sys_gf_w)

    return out


# ---------------------------------------------------------------------------
# Per-file insights
# ---------------------------------------------------------------------------

def find_column_by_pattern(df, patterns):
    """Find column name by matching patterns (case-insensitive)."""
    for pattern in patterns:
        for col in df.columns:
            if pattern.lower() == str(col).lower():
                return col
            if pattern.lower() in str(col).lower():
                return col
    return None


def compute_basic_file_insights(file_obj, filetype: str) -> dict:
    """
    Compute lightweight per-file insights for use in the Multi-Parameter
    file selector: total runtime, max thrust, and max RPM.
    """
    insights = {"runtime_s": None, "max_thrust": None, "max_rpm": None, "max_power": None}

    if filetype not in [".csv", ".xlsx"]:
        return insights

    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=filetype) as tmp:
            try:
                file_obj.seek(0)
            except Exception:
                pass
            data = file_obj.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            tmp.write(data)
            tmp.flush()
            tmp_name = tmp.name

        df, _ = load_data(tmp_name, filetype, key_suffix="_insights")
    except Exception:
        df = None
    finally:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except Exception:
                pass

    if df is None or df.empty:
        return insights

    df = ensure_seconds_column(df)

    if "timestamp_seconds" in df.columns:
        ts = pd.to_numeric(df["timestamp_seconds"], errors="coerce").dropna()
        if not ts.empty:
            insights["runtime_s"] = float(ts.max() - ts.min())

    thrust_col = find_column_by_pattern(
        df, ["Thrust - gf", "Thrust (gf)", "Thrust (kgf)", "Thrust [g]",
             "Thrust", "thrust"],
    )
    if thrust_col and thrust_col in df.columns:
        thrust_vals = pd.to_numeric(df[thrust_col], errors="coerce").dropna()
        if not thrust_vals.empty:
            insights["max_thrust"] = float(thrust_vals.max())

    rpm_col = find_column_by_pattern(
        df,
        ["RPM", "RPM1 - RPM", "RPM1", "rpm1",
         "Motor Electrical Speed (RPM)", "Motor Electrical Speed",
         "Electrical Speed (RPM)", "Electrical Speed",
         "Rotational Speed (RPM)", "Rotational Speed"],
    )
    if rpm_col and rpm_col in df.columns:
        rpm_vals = pd.to_numeric(df[rpm_col], errors="coerce").dropna()
        if not rpm_vals.empty:
            insights["max_rpm"] = float(rpm_vals.max())

    power_col = find_column_by_pattern(
        df,
        ["Electrical Power (W)", "Electrical Power - W", "Electrical (W)",
         "MotorPower - W", "MotorPower", "Mechanical (W)",
         "InPower - W", "InPower",
         "PowerInLine - W", "PowerInLine", "Power"],
    )
    if power_col and power_col in df.columns:
        power_vals = pd.to_numeric(df[power_col], errors="coerce").dropna()
        if not power_vals.empty:
            insights["max_power"] = float(power_vals.max())

    return insights


# ---------------------------------------------------------------------------
# Cached file loading (eliminates redundant parsing on Streamlit reruns)
# ---------------------------------------------------------------------------

def content_hash(file_bytes: bytes) -> str:
    """Compute a fast MD5 hash of file content for cache keying."""
    return hashlib.md5(file_bytes).hexdigest()


@st.cache_data(show_spinner="Parsing file...", max_entries=20)
def cached_load_csv(file_hash: str, _file_bytes: bytes) -> tuple:
    """Parse CSV bytes into a DataFrame, cached by content hash.

    Returns (df, info_text) — info_text is the pre-header file info string
    that callers should set into session state if needed.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(_file_bytes)
        tmp.flush()
        try:
            # Capture and restore session-scoped file info so cached results
            # depend only on file content, not on the caller's session state.
            prev_info = st.session_state.get("report_file_info_text", "")
            df = load_csv(tmp.name)
            info_text = st.session_state.get("report_file_info_text", prev_info)
            st.session_state["report_file_info_text"] = prev_info
            return df, info_text
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


@st.cache_data(show_spinner="Parsing ULog...", max_entries=10)
def cached_load_ulog(file_hash: str, _file_bytes: bytes) -> tuple:
    """Parse ULog bytes into DataFrames, cached by content hash.

    Returns (dfs_dict, topic_names).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ulg") as tmp:
        tmp.write(_file_bytes)
        tmp.flush()
        try:
            dfs_dict, topics = load_ulog(tmp.name)
            return dfs_dict, topics
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


@st.cache_data(show_spinner="Loading file...", max_entries=20)
def cached_load_file(file_hash: str, _file_bytes: bytes, filename: str, file_ext: str) -> tuple:
    """Unified cached loader for CSV, ULog, and Excel files.

    Returns (result, file_ext) where result is:
      - CSV/Excel: (df, info_text)
      - ULog: (dfs_dict, topic_names)
    """
    # Hard cap very large uploads even if Streamlit's maxUploadSize is raised.
    if len(_file_bytes) > MAX_FILE_BYTES:
        return (None, "File too large")

    if file_ext == ".ulg":
        return cached_load_ulog(file_hash, _file_bytes)

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(_file_bytes)
        tmp.flush()
        try:
            prev_info = st.session_state.get("report_file_info_text", "")
            df, info = load_data(tmp.name, file_ext, key_suffix=f"_cached_{filename}")
            info_text = st.session_state.get("report_file_info_text", prev_info)
            st.session_state["report_file_info_text"] = prev_info
            return (df, info_text) if df is not None else (None, "")
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
