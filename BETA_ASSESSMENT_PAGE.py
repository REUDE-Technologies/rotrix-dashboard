import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
import tempfile
import os
from datetime import datetime
from pyulog import ULog
import warnings
import requests
import re
from io import BytesIO
warnings.filterwarnings('ignore')

class UploadedGitHubFile:
    def __init__(self, content, name, filetype): 
        self.file = BytesIO(content)
        self.name = name
        self.type = filetype
        self.size = len(content)
    def read(self, *args, **kwargs):
        return self.file.read(*args, **kwargs)
    def seek(self, *args, **kwargs):
        return self.file.seek(*args, **kwargs)

def process_url(url):
    if "github.com" in url:
        try:
            # Handle folder URL (e.g., https://github.com/username/repo/tree/main/folder)
            if "tree" in url:
                parts = url.split("tree/")
                if len(parts) < 2:
                    st.error("Invalid GitHub folder URL format. Please include 'tree/' followed by the folder path.")
                    return None
                base_url = parts[0].rstrip('/')
                path = parts[1].lstrip('/')
                url_parts = base_url.split("/")
                if len(url_parts) < 5 or url_parts[2] != "github.com":
                    st.error("Unable to parse repository from URL.")
                    return None
                repo = f"{url_parts[3]}/{url_parts[4]}"
                folder_path = path.split('/', 1)[-1] if '/' in path else path
                api_url = f"https://api.github.com/repos/{repo}/contents/{folder_path}"
                response = requests.get(api_url, headers={"Accept": "application/vnd.github.v3+json"})
                if response.status_code == 200:
                    files = [item for item in response.json() if item['type'] == 'file' and item['name'].endswith(('.csv', '.ulg'))]
                    if not files:
                        st.warning("No .csv or .ulg files found in the folder.")
                        return None
                    file_data = {}
                    for file in files:
                        file_response = requests.get(file['download_url'])
                        if file_response.status_code == 200:
                            file_ext = os.path.splitext(file['name'])[-1].lower()
                            file_data[file['name']] = (file_response.content, file_ext)
                    return file_data if file_data else None
                else:
                    st.error(f"Failed to fetch folder contents. Status code: {response.status_code}, Message: {response.text}, API URL: {api_url}")
                    return None
            # Handle raw file URL (e.g., https://raw.githubusercontent.com/username/repo/main/file.csv)
            elif "raw.githubusercontent.com" in url:
                file_name = url.split("/")[-1]
                file_ext = os.path.splitext(file_name)[-1].lower()
                if file_ext in [".csv", ".ulg"]:
                    response = requests.get(url)
                    if response.status_code == 200:
                        return {file_name: (response.content, file_ext)}
                    else:
                        st.error(f"Failed to download file. Status code: {response.status_code}, URL: {url}")
                        return None
            # Handle blob URL (e.g., https://github.com/username/repo/blob/main/file.csv)
            elif "/blob/" in url:
                raw_url = url.replace("/blob/", "/raw/")
                file_name = raw_url.split("/")[-1]
                file_ext = os.path.splitext(file_name)[-1].lower()
                if file_ext in [".csv", ".ulg"]:
                    response = requests.get(raw_url)
                    if response.status_code == 200:
                        return {file_name: (response.content, file_ext)}
                    else:
                        st.error(f"Failed to download file. Status code: {response.status_code}, URL: {raw_url}")
                        return None
            # Handle direct file URL (e.g., https://github.com/username/repo/filename.csv)
            elif len(url.split("/")) > 4 and url.split("/")[4] not in ["tree", "blob", "raw"]:
                base_parts = url.split("/")
                repo = f"{base_parts[3]}/{base_parts[4]}"
                file_path = "/".join(base_parts[5:])
                raw_url = f"https://raw.githubusercontent.com/{repo}/main/{file_path}"
                file_name = file_path.split("/")[-1]
                file_ext = os.path.splitext(file_name)[-1].lower()
                if file_ext in [".csv", ".ulg"]:
                    response = requests.get(raw_url)
                    if response.status_code == 200:
                        return {file_name: (response.content, file_ext)}
                    else:
                        st.error(f"Failed to download file. Status code: {response.status_code}, URL: {raw_url}")
                        return None
            else:
                st.warning("Unsupported GitHub URL format. Please use a folder URL with 'tree/' or a raw/blob file URL.")
                return None
        except Exception as e:
            st.error(f"Error processing URL: {str(e)}")
            return None
    return None

# Page configuration
st.set_page_config(
    page_title="Motor Data Vantage",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants and configurations
TOPIC_ASSESSMENT_PAIRS = [
    ("vehicle_local_position", "Actualposition"),
    ("vehicle_local_position_setpoint", "Setpointposition"),
    ("vehicle_local_position_setpoint", "Thrust"),
    ("vehicle_torque_setpoint", "Torque"),
    ("px4io_status", "Control"),
    ("battery_status", "Battery"),
]

ASSESSMENT_Y_AXIS_MAP = {
    "Actualposition": ["x", "y", "z"],
    "Setpointposition": ["x", "y", "z"],
    "Thrust": ["thrust[0]", "thrust[1]", "thrust[2]", "thrust[3]", "thrust[4]", "thrust[5]"],
    "Torque": ["xyz[0]", "xyz[1]", "xyz[2]"],
    "Control": ["pwm[0]", "pwm[1]", "pwm[2]", "pwm[3]", "pwm[4]", "pwm[5]"],
    "Battery": ["voltage_v", "current_average_a", "discharged_mah"],
}

COLUMN_DISPLAY_NAMES = {
    "pwm[0]": "Motor 1 pwm",
    "pwm[1]": "Motor 2 pwm",
    "pwm[2]": "Motor 3 pwm",
    "pwm[3]": "Motor 4 pwm",
    "pwm[4]": "Motor 5 pwm",
    "pwm[5]": "Motor 6 pwm",
    "thrust[0]": "Thrust Motor 1",
    "thrust[1]": "Thrust Motor 2",
    "thrust[2]": "Thrust Motor 3",
    "thrust[3]": "Thrust Motor 4",
    "thrust[4]": "Thrust Motor 5",
    "thrust[5]": "Thrust Motor 6",
    "xyz[0]": "Torque x",
    "xyz[1]": "Torque y",
    "xyz[2]": "Torque z",
    "voltage_v": "Battery Voltage",
    "current_average_a": "Current",
    "discharged_mah": "Discharged Capacity",
}

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'files_submitted' not in st.session_state:
    st.session_state.files_submitted = False
if 'show_upload_area' not in st.session_state:
    st.session_state.show_upload_area = True
if 'upload_opened_by_plus' not in st.session_state:
    st.session_state.upload_opened_by_plus = False
# Add new session state variables for the new file upload interface
if 'upload_source' not in st.session_state:
    st.session_state.upload_source = "desktop"  # desktop
if 'show_file_preview' not in st.session_state:
    st.session_state.show_file_preview = False
if 'file_rename_mode' not in st.session_state:
    st.session_state.file_rename_mode = {}

# Initialize session state variables for Multi-Parameter Analysis
if 'multi_param_file_selection' not in st.session_state:
    st.session_state.multi_param_file_selection = "None"
if 'multi_param_ulog_topic' not in st.session_state:
    st.session_state.multi_param_ulog_topic = None
if 'multi_param_x_axis' not in st.session_state:
    st.session_state.multi_param_x_axis = ''
if 'multi_param_left_y_axes' not in st.session_state:
    st.session_state.multi_param_left_y_axes = []
if 'multi_param_right_y_axes' not in st.session_state:
    st.session_state.multi_param_right_y_axes = []
if 'multi_param_data_selected_cols' not in st.session_state:
    st.session_state.multi_param_data_selected_cols = []
if 'multi_param_smoothing' not in st.session_state:
    st.session_state.multi_param_smoothing = False
if 'multi_param_smoothing_window' not in st.session_state:
    st.session_state.multi_param_smoothing_window = 5
if 'multi_param_saved_graphs' not in st.session_state:
    st.session_state.multi_param_saved_graphs = []

# Initialize session state variables for Multi-File Multi-Parameter Analysis
if 'multi_file_multi_param_selections' not in st.session_state:
    st.session_state.multi_file_multi_param_selections = []
if 'multi_file_multi_param_x_axis' not in st.session_state:
    st.session_state.multi_file_multi_param_x_axis = ''
if 'multi_file_multi_param_left_y_axes' not in st.session_state:
    st.session_state.multi_file_multi_param_left_y_axes = []
if 'multi_file_multi_param_right_y_axes' not in st.session_state:
    st.session_state.multi_file_multi_param_right_y_axes = []
if 'multi_file_multi_param_data_selected_cols' not in st.session_state:
    st.session_state.multi_file_multi_param_data_selected_cols = []

# Initialize global variables (if needed)

# Function to change page
def change_page(page):
    if page == 'home':
        # Store the current analysis type and data source before going back
        st.session_state.previous_analysis_type = st.session_state.analysis_type
        st.session_state.previous_data_source = st.session_state.data_source
    st.session_state.current_page = page

# Utility functions
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
    
    # Try to convert to numeric, coercing errors to NaN
    converted = pd.to_numeric(series, errors='coerce')
    return converted

def filter_info_rows(df):
    """
    Filter out rows that contain mostly "None" values or string information rows.
    These are typically metadata rows that shouldn't be included in the data analysis.
    Also converts columns that should be numeric to proper numeric types.
    """
    if df is None or df.empty:
        return df
    
    # Convert "None" strings to NaN for easier processing
    df_filtered = df.copy()
    
    # Replace string "None" (case-insensitive) with NaN
    df_filtered = df_filtered.replace(['None', 'none', 'NONE', 'null', 'Null', 'NULL'], np.nan)
    
    # Identify columns that should be numeric based on column names
    # Common numeric column patterns
    numeric_patterns = [
        'throttle', 'voltage', 'current', 'rpm', 'thrust', 'torque', 
        'power', 'effect', 'speed', 'time', 'timestamp', 'index',
        'vol v', 'cur a', 'thrust gf', 'torque nm', 'rpm1 rpm',
        'sys', 'motor', 'electrical'
    ]
    
    # Find columns that match numeric patterns but aren't already numeric
    potential_numeric_cols = []
    for col in df_filtered.columns:
        col_lower = str(col).lower()
        # Skip Index column and timestamp_seconds as they're handled separately
        if col in ['Index', 'index'] or 'timestamp' in col_lower:
            continue
        # Check if column name suggests it should be numeric
        if any(pattern in col_lower for pattern in numeric_patterns):
            if df_filtered[col].dtype == 'object':  # String/object type
                potential_numeric_cols.append(col)
    
    # Convert potential numeric columns to numeric
    for col in potential_numeric_cols:
        df_filtered[col] = convert_to_numeric_safe(df_filtered[col])
    
    # Get numeric columns (now includes converted ones)
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter rows where most numeric columns are NaN/None or contain string values
    if numeric_cols:
        # Count how many numeric columns have valid (non-NaN) values in each row
        valid_numeric_count = df_filtered[numeric_cols].notna().sum(axis=1)
        total_numeric_cols = len(numeric_cols)
        
        # Keep rows where at least 50% of numeric columns have valid values
        # This filters out info rows that have mostly "None" values
        threshold = max(1, total_numeric_cols * 0.5)  # At least 50% or minimum 1 column
        df_filtered = df_filtered[valid_numeric_count >= threshold].copy()
    
    # Additional check: filter rows where most columns (numeric + non-numeric) are NaN/None
    # This catches info rows that might have text in one column but None in others
    if len(df_filtered.columns) > 0:
        total_cols = len(df_filtered.columns)
        valid_count = df_filtered.notna().sum(axis=1)
        # Keep rows where at least 30% of all columns have valid values
        threshold = max(1, total_cols * 0.3)
        df_filtered = df_filtered[valid_count >= threshold].copy()
    
    # Final pass: convert all remaining object columns that look numeric
    for col in df_filtered.columns:
        if col in ['Index', 'index'] or 'timestamp' in str(col).lower():
            continue
        if df_filtered[col].dtype == 'object':
            # Try converting to numeric
            converted = pd.to_numeric(df_filtered[col], errors='coerce')
            # If most values converted successfully, use the converted version
            if converted.notna().sum() > len(df_filtered) * 0.5:
                df_filtered[col] = converted
    
    # Reset index after filtering
    df_filtered = df_filtered.reset_index(drop=True)
    
    return df_filtered

def load_csv(file):
    if isinstance(file, str):
        with open(file, 'r', encoding='utf-8') as f:
            lines = [next(f) for _ in range(20)]
        file_obj = file
    else:
        file.seek(0)
        lines = [file.readline().decode('utf-8') for _ in range(20)]
        file.seek(0)
        file_obj = file
    
    # Find the header row index - look for common header patterns
    header_row = None
    for i, line in enumerate(lines):
        line_lower = line.lower()
        # Check for common header indicators
        if any(keyword in line_lower for keyword in [
            "testrecordid", "recordid", "timestamp", "time", "throttle", 
            "rpm", "thrust", "torque", "voltage", "current", "speedmms",
            "vol v", "cur a", "thrust gf", "torque nm", "rpm1 rpm"
        ]):
            # Verify it looks like a header (has multiple comma-separated values)
            if line.count(',') >= 2:
                header_row = i
                break
    
    # If no header found, try common positions
    if header_row is None:
        # Try different skip values
        for skip in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]:
            try:
                if not isinstance(file, str):
                    file_obj.seek(0)
                test_df = pd.read_csv(file_obj, encoding='utf-8', skiprows=skip, header=0, nrows=1)
                if len(test_df.columns) >= 3:  # Reasonable number of columns
                    header_row = skip
                    break
            except:
                continue
    
    if header_row is None:
        header_row = 0  # Fallback to first row
    
    # Try to load with error handling
    try:
        if not isinstance(file, str):
            file_obj.seek(0)
        df = pd.read_csv(file_obj, encoding='utf-8', skiprows=header_row, header=0, on_bad_lines='skip')
        # Clean up any completely empty rows/columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        # Handle duplicate column names
        df = fix_duplicate_columns(df)
        # Filter out rows with mostly "None" values or string info rows
        df = filter_info_rows(df)
        return df
    except Exception as e1:
        try:
            if not isinstance(file, str):
                file_obj.seek(0)
            df = pd.read_csv(file_obj, encoding='utf-8-sig', skiprows=header_row, header=0, on_bad_lines='skip')
            df = df.dropna(how='all').dropna(axis=1, how='all')
            # Handle duplicate column names
            df = fix_duplicate_columns(df)
            # Filter out rows with mostly "None" values or string info rows
            df = filter_info_rows(df)
            return df
        except Exception as e2:
            # Last resort: try without skiprows
            try:
                if not isinstance(file, str):
                    file_obj.seek(0)
                df = pd.read_csv(file_obj, encoding='utf-8', header=0, on_bad_lines='skip')
                df = df.dropna(how='all').dropna(axis=1, how='all')
                # Handle duplicate column names
                df = fix_duplicate_columns(df)
                # Filter out rows with mostly "None" values or string info rows
                df = filter_info_rows(df)
                return df
            except Exception as e3:
                raise Exception(f"Failed to parse CSV: {str(e1)}, {str(e2)}, {str(e3)}")

def load_ulog(file, key_suffix=""):
    ALLOWED_TOPICS = set(t for t, _ in TOPIC_ASSESSMENT_PAIRS)
    
    # Create a temporary file to store the content
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ulg') as tmp_file:
        try:
            # If file is a string (path), read directly
            if isinstance(file, str):
                with open(file, 'rb') as f:
                    tmp_file.write(f.read())
            else:
                # If file is a file object, write its content
                file.seek(0)
                tmp_file.write(file.read())
            tmp_file.flush()
            
            # Process the ULog file
            ulog = ULog(tmp_file.name)
            if not ulog.data_list:
                st.warning("⚠️ No data found in the ULog file")
                return {}, []
                
            extracted_dfs = {}
            for msg in ulog.data_list:
                if msg.data:  # Only process messages with data
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
            # Clean up the temporary file
            try:
                os.unlink(tmp_file.name)
            except:
                pass

def get_axis_title(axis_name):
    if axis_name == 'timestamp_seconds':
        return 'TIME(secs)'
    return axis_name

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
    except Exception as e:
        st.error(f"Error generating timestamp ticks: {str(e)}")
        return [], []

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

def format_seconds_to_mmss(seconds):
    """Format seconds to MM:SS format."""
    try:
        minutes = int(float(seconds) // 60)
        remaining_seconds = int(float(seconds) % 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"
    except Exception as e:
        st.error(f"Error formatting seconds: {str(e)}")
        return "00:00"

def mmss_to_seconds(mmss_str):
    """Convert MM:SS format to seconds."""
    try:
        if ':' in mmss_str:
            minutes, seconds = mmss_str.split(':')
            return int(minutes) * 60 + int(seconds)
        else:
            return float(mmss_str)
    except:
        return 0.0

def seconds_to_mmss(seconds):
    """Convert seconds to MM:SS format."""
    try:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"
    except:
        return "00:00"

def ensure_seconds_column(df):
    """Ensure timestamp_seconds column exists in the dataframe."""
    if 'timestamp_seconds' not in df.columns:
        if 'timestamp' in df.columns:
            df['timestamp_seconds'] = df['timestamp'] / 1e6  # Convert microseconds to seconds
        elif 'Timestamp (hh:mm:ss)' in df.columns:
            df['timestamp_seconds'] = df['Timestamp (hh:mm:ss)'].apply(lambda x: mmss_to_seconds(str(x)))
        else:
            df['timestamp_seconds'] = range(len(df))
    
    # Check if timestamp_seconds column has valid data (not all NaN)
    if 'timestamp_seconds' in df.columns:
        if df['timestamp_seconds'].isna().all() or df['timestamp_seconds'].isnull().all():
            df = df.drop('timestamp_seconds', axis=1)
    
    return df

def load_data(file, filetype, key_suffix):
    """Load data from various file types."""
    try:
        if filetype == ".csv":
            df = load_csv(file)
        elif filetype == ".xlsx":
            df = pd.read_excel(file)
            # Clean up any completely empty rows/columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            # Handle duplicate column names
            df = fix_duplicate_columns(df)
            # Filter out rows with mostly "None" values or string info rows
            df = filter_info_rows(df)
        else:
            st.error(f"Unsupported file type: {filetype}")
            return None, None
        
        if df is not None and not df.empty:
            df = ensure_seconds_column(df)
            return df, None
        else:
            st.error("No data found in file")
            return None, None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None

def get_display_name(col):
    """Get a display-friendly name for a column."""
    if col == 'timestamp_seconds':
        return 'Time (secs)'
    elif col == 'Index':
        return 'Index'
    else:
        return col

def get_numeric_columns(df):
    """Get numeric columns from dataframe"""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def is_column_empty(df, col):
    """Check if a column is empty (all null/NaN values)"""
    if col not in df.columns:
        return True
    return df[col].isna().all() or (df[col] == 0).all() or len(df[col].dropna()) == 0

def get_non_empty_columns(df1, df2, columns):
    """Get columns that are not empty in both dataframes"""
    non_empty_cols = []
    for col in columns:
        if col in df1.columns and col in df2.columns:
            if not is_column_empty(df1, col) and not is_column_empty(df2, col):
                non_empty_cols.append(col)
    return non_empty_cols

def smooth_data_for_plotting(df, x_col, y_cols, method='linear', num_points=None, smoothing_window=5):
    """
    Smooth/interpolate data to create smooth lines while preserving all original data points.
    
    Parameters:
    - df: DataFrame with data
    - x_col: Column name for X-axis
    - y_cols: List of column names for Y-axes
    - method: Smoothing method ('linear', 'cubic', 'moving_average', 'savgol')
    - num_points: Number of points for interpolation (None = auto based on data density)
    - smoothing_window: Window size for moving average or Savitzky-Golay filter
    
    Returns:
    - DataFrame with smoothed/interpolated data including all original points
    """
    if df is None or df.empty or x_col not in df.columns:
        return df
    
    # Sort by X-axis
    df_sorted = df.sort_values(by=x_col).copy()
    
    # Remove duplicates on X-axis (keep first occurrence)
    df_sorted = df_sorted.drop_duplicates(subset=[x_col], keep='first')
    
    if len(df_sorted) < 2:
        return df_sorted
    
    # Determine number of interpolation points
    x_min = df_sorted[x_col].min()
    x_max = df_sorted[x_col].max()
    x_range = x_max - x_min
    
    if num_points is None:
        # Auto-determine: use more points for larger ranges, but cap at reasonable number
        original_points = len(df_sorted)
        if x_range > 0:
            # Aim for ~5-10x the original points for very smooth curves
            num_points = max(200, min(original_points * 10, 2000))
        else:
            num_points = original_points
    
    # Create evenly spaced X values for interpolation
    x_interp = np.linspace(x_min, x_max, num_points)
    
    # Create result dataframe starting with interpolated X
    result_df = pd.DataFrame({x_col: x_interp})
    
    # Process each Y column
    for y_col in y_cols:
        if y_col not in df_sorted.columns:
            continue
        
        # Get valid data points (non-NaN)
        valid_mask = df_sorted[y_col].notna()
        if valid_mask.sum() < 2:
            result_df[y_col] = np.nan
            continue
        
        x_valid = df_sorted.loc[valid_mask, x_col].values
        y_valid = df_sorted.loc[valid_mask, y_col].values
        
        # Remove duplicates in X for interpolation
        unique_mask = np.concatenate(([True], np.diff(x_valid) != 0))
        x_unique = x_valid[unique_mask]
        y_unique = y_valid[unique_mask]
        
        if len(x_unique) < 2:
            result_df[y_col] = np.nan
            continue
        
        try:
            # Apply smoothing first if method requires it
            if method == 'moving_average' and len(y_unique) > smoothing_window:
                # Apply moving average smoothing
                window = min(smoothing_window, len(y_unique) // 3)
                if window >= 3:
                    # Use pandas rolling for moving average
                    temp_series = pd.Series(y_unique, index=x_unique)
                    y_smoothed = temp_series.rolling(window=window, center=True, min_periods=1).mean().values
                    y_unique = y_smoothed
            elif method == 'savgol' and len(y_unique) > smoothing_window:
                # Use Savitzky-Golay filter for better smoothing
                try:
                    from scipy.signal import savgol_filter
                    window_length = min(smoothing_window, len(y_unique))
                    if window_length % 2 == 0:
                        window_length += 1  # Must be odd
                    if window_length >= 3 and window_length <= len(y_unique):
                        polyorder = min(3, window_length - 1)
                        y_unique = savgol_filter(y_unique, window_length, polyorder)
                except ImportError:
                    # Fall back to moving average if scipy not available
                    window = min(smoothing_window, len(y_unique) // 3)
                    if window >= 3:
                        temp_series = pd.Series(y_unique, index=x_unique)
                        y_unique = temp_series.rolling(window=window, center=True, min_periods=1).mean().values
            
            # Now interpolate to create smooth curve
            if method == 'cubic' and len(x_unique) >= 4:
                try:
                    from scipy.interpolate import interp1d
                    # Use cubic spline for very smooth curves
                    f = interp1d(x_unique, y_unique, kind='cubic', bounds_error=False, fill_value='extrapolate')
                    y_interp = f(x_interp)
                except (ImportError, ValueError):
                    # Fall back to linear if cubic fails
                    y_interp = np.interp(x_interp, x_unique, y_unique)
            else:
                # Use linear interpolation (smooth enough with pre-smoothed data)
                y_interp = np.interp(x_interp, x_unique, y_unique)
            
            result_df[y_col] = y_interp
        except Exception as e:
            # If smoothing/interpolation fails, use simple linear interpolation
            try:
                y_interp = np.interp(x_interp, x_unique, y_valid[unique_mask])
                result_df[y_col] = y_interp
            except:
                result_df[y_col] = np.nan
    
    # Merge with original data points to ensure we keep all original values
    # But use interpolated values for plotting smooth lines
    original_df = df_sorted[[x_col] + [c for c in y_cols if c in df_sorted.columns]].copy()
    
    # For plotting, we want the interpolated smooth values, not the original noisy ones
    # But we can optionally merge to show original points as markers
    # For now, just use the interpolated values for smooth lines
    result_df = result_df.sort_values(by=x_col).reset_index(drop=True)
    
    return result_df

def detect_abnormalities(series, threshold=3.0):
    """Detect abnormal points in a series using z-score threshold."""
    if len(series) < 2:  # Need at least 2 points to calculate z-score
        return pd.Series(False, index=series.index), pd.Series(0, index=series.index)
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold, z_scores

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

def add_hhmmss_seconds_column(df, timestamp_col='Timestamp (hh:mm:ss)'):
    def hhmmss_to_seconds(ts):
        try:
            if pd.isnull(ts):
                return None
            ts = str(ts).strip()
            # Try parsing with AM/PM
            try:
                dt = datetime.strptime(ts, "%I:%M:%S %p")
            except ValueError:
                try:
                    dt = datetime.strptime(ts, "%H:%M:%S")
                except ValueError:
                    print(f"DEBUG: Unrecognized timestamp format: '{ts}'")
                    return None
            return dt.hour * 3600 + dt.minute * 60 + dt.second
        except Exception as e:
            print(f"DEBUG: Error parsing timestamp '{ts}': {e}")
            return None
    abs_seconds = df[timestamp_col].apply(hhmmss_to_seconds)
    if abs_seconds.isnull().all():
        print("DEBUG: All values in timestamp_seconds are None. Check the format of your timestamp column!")
    if not abs_seconds.isnull().all():
        elapsed_seconds = abs_seconds - abs_seconds.iloc[0]
        df['timestamp_seconds'] = elapsed_seconds.astype('Int64')
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

# Main application
def main():
    # Fixed header with improved structure
    st.markdown("""
    <style>
    .fixed-header {
        top: 18px;
        left: 18px;
        z-index: 1001;
        background: #fff;
        border-radius: 14px;
        box-shadow: 0 4px 16px rgba(44, 62, 80, 0.10);
        padding: 16px 28px 14px 22px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        min-width: 260px;
        max-width: 380px;
        border: 1px solid #e0e0e0;
    }
    .fixed-header h1 {
        color: #2E86C1;
        margin: 0 0 2px 0;
        font-size: 1.7rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        line-height: 1.1;
        font-weight: 700;
    }
    .fixed-header .rocket-icon {
        font-size: 1.7rem;
        line-height: 1;
    }
    .fixed-header p {
        color: #666;
        margin: 0;
        font-size: 0.98rem;
        line-height: 1.2;
        font-weight: 400;
    }
    /* Add padding to main content to prevent overlap with fixed header */
    .main .block-container {
        padding-top: 40px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="fixed-header">
        <h1><span class="rocket-icon">🚀</span> Motor Data Vantage </h1>
    </div>
    """, unsafe_allow_html=True)

    # File Upload Section
    if st.session_state.show_upload_area:
        st.markdown("""
        <style>
        .upload-section {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            border: 2px solid #e9ecef;
            transition: all 0.3s ease;
        }
        .upload-section.active {
            border-color: #007bff;
            background: #f0f8ff;
            box-shadow: 0 4px 12px rgba(0, 123, 255, 0.15);
        }
        .upload-section:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        }
        .file-preview-card {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
        }
        .file-preview-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-color: #007bff;
            transform: translateY(-1px);
        }
        .file-actions {
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }
        .file-action-btn {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .file-action-btn:hover {
            background: #e9ecef;
            border-color: #adb5bd;
        }
        .file-action-btn.primary {
            background: #007bff;
            color: white;
            border-color: #007bff;
        }
        .file-action-btn.primary:hover {
            background: #0056b3;
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
            border: 2px dashed #dee2e6;
            border-radius: 8px;
            padding: 30px;
            text-align: center;
            background: white;
            transition: all 0.2s;
            cursor: pointer;
        }
        .upload-zone:hover {
            border-color: #007bff;
            background: #f8f9ff;
            transform: scale(1.02);
        }
        .upload-zone.dragover {
            border-color: #007bff;
            background: #e3f2fd;
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
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .bulk-actions h6 {
            color: #495057;
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
        .video-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .video-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 12px;
            color: white;
        }
        .video-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: white;
        }
        .video-description {
            font-size: 0.9rem;
            opacity: 0.9;
            color: white;
            margin-bottom: 12px;
        }
        .video-placeholder {
            padding: 20px;
            text-align: center;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            border: 2px dashed rgba(255, 255, 255, 0.3);
            color: white;
        }
        .upload-section-enhanced {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            border: 1px solid #dee2e6;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        .github-section-enhanced {
            background: linear-gradient(135deg, #f0f8ff 0%, #e3f2fd 100%);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            border: 1px solid #b3d8fd;
            box-shadow: 0 2px 8px rgba(0, 123, 255, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<h3 style='text-align: center; color: #2E86C1; margin-bottom: 30px;'>📁 File Management</h3>", unsafe_allow_html=True)
        github_col, upload_col, video_col = st.columns([0.35, 0.35, 0.3])
        with github_col:
            st.markdown("""
            <div class="github-section-enhanced">
                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 8px;'>
                    <img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' width='22' style='margin-right: 4px;'/>
                    <span style='font-size: 1.08rem; font-weight: 700; color: #24292f;'>GitHub</span>
                    <span style='font-size: 0.98rem; color: #2980b9; margin-left: 6px;'>(<a style='color:#2980b9; text-decoration:underline; cursor:pointer;' href='#'>.csv, .ulg</a>)</span>
                </div>
                <div style='font-size: 0.98rem; color: #444; margin-bottom: 12px;'>
                    Paste a <b>GitHub <span style='font-weight:700;'>raw/blob/folder URL</span></b> to fetch files.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.session_state.get("clear_github_url_input", False):
                st.session_state.github_url_input = ""
                st.session_state.clear_github_url_input = False
            github_col, fetch_col  = st.columns([5, 1])
            with fetch_col:
                fetch_github = st.button("Fetch", key="fetch_github_btn", use_container_width=True)
            with github_col:
                github_url = st.text_input("GitHub URL (raw, blob, or folder)", key="github_url_input", label_visibility="collapsed", placeholder="e.g. https://github.com/user/repo/blob/main/data.csv")
            if fetch_github and github_url:
                result = process_url(github_url)
                if result:
                    existing_names = [f.name for f in st.session_state.uploaded_files]
                    for file_name, (file_content, file_ext) in result.items():
                        if file_name not in existing_names:
                            filetype = "text/csv" if file_ext == ".csv" else "application/octet-stream"
                            file_like = UploadedGitHubFile(file_content, file_name, filetype)
                            st.session_state.uploaded_files.append(file_like)
                    st.session_state.clear_github_url_input = True  # Set flag to clear input on next rerun
                    st.rerun()
                else:
                    st.warning("No valid .csv or .ulg files found at the provided URL.")
        with upload_col:
            # File uploader with increased height
            st.markdown("""
            <style>
            section[data-testid="stFileUploader"] > div {
                min-height: 120px !important;
                height: 120px !important;
                display: flex;
                align-items: center;
            }
            </style>
            """, unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "Choose files to upload", 
                type=["csv", "ulg"], 
                key="desktop_uploader", 
                label_visibility="collapsed", 
                accept_multiple_files=True,
                help="Drag and drop files here or click to browse")
        with video_col:            
            # Video player
            try:
                with open("streamlit-assessment_dashboard.mp4", "rb") as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes, start_time=0)
            except FileNotFoundError:
                st.markdown("""
                <div class="video-placeholder">
                    <div style='font-size: 2rem; margin-bottom: 10px;'>🎥</div>
                    <div style='font-size: 0.9rem; margin-bottom: 5px;'>Tutorial video not found</div>
                    <div style='font-size: 0.8rem; opacity: 0.8;'>Add 'streamlit-assessment_dashboard.webm' to your project</div>
                </div>
                """, unsafe_allow_html=True)
        # Process uploaded files
        if uploaded_files:
            new_files_added = False
            existing_names = [f.name for f in st.session_state.uploaded_files]
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in existing_names:
                    st.session_state.uploaded_files.append(uploaded_file)
                    new_files_added = True
            if new_files_added:
                st.rerun()
        # Show file preview section if files are uploaded
        if st.session_state.uploaded_files:
            st.markdown("<h4 style='margin-top: 0px; color: #495057;'>📋 File Preview & Management</h4>", unsafe_allow_html=True)
            st.markdown("<h5 style='color: #6c757d; margin-bottom: 15px;'>📎 Uploaded Files</h5>", unsafe_allow_html=True)
            for i, file in enumerate(st.session_state.uploaded_files):
                file_ext = file.name.split('.')[-1].lower() if '.' in file.name else 'unknown'
                file_type_badge = f"<span class='file-type-badge {file_ext}'>{file_ext}</span>"
                
                # Use columns to align file name/details and action buttons in a single row
                file_cols = st.columns([12, 1, 1, 1])  # Preview, Rename, Remove
                with file_cols[0]:
                    st.markdown(f"""
                    <div style="font-weight: 600; color: #495057;">
                        📄 {file.name}
                        <span style="font-size: 12px; color: #6c757d; margin-left: 10px;">
                            Size: {file.size / (1024*1024):.1f} MB | Type: {file.type or 'Unknown'} {file_type_badge}
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
                        st.rerun()
                # Quick Preview UI
                if st.session_state.get(f"preview_mode_{i}", False):
                    with st.expander("Quick Preview", expanded=True):
                        file.seek(0)
                        file_ext = file.name.split('.')[-1].lower() if '.' in file.name else 'unknown'
                        if file_ext == "csv":
                            try:
                                df = pd.read_csv(file, nrows=5)
                                st.dataframe(df, use_container_width=True)
                            except Exception as e:
                                st.error(f"Could not preview CSV: {e}")
                        elif file_ext == "ulg":
                            size_mb = file.size / (1024 * 1024)
                            st.info("ULG preview: Only file name, size, and type shown.")
                            st.write({"Name": file.name, "Size (MB)": f"{size_mb:.2f} MB", "Type": file.type})
                        else:
                            st.warning("Preview not supported for this file type.")
                        file.seek(0)
                if st.session_state.file_rename_mode.get(i, False):
                    with st.container():
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
            # Empty state
            st.markdown("""
            <div class="upload-zone">
                <div style="font-size: 48px; margin-bottom: 20px;">📁</div>
                <h4 style="color: #6c757d; margin-bottom: 10px;">No files uploaded yet</h4>
                <p style="color: #adb5bd; margin-bottom: 20px;">Upload your CSV or ULG files to begin analysis</p>
                <p style="font-size: 12px; color: #ced4da;">Supported formats: .csv, .ulg</p>
            </div>
            """, unsafe_allow_html=True)

        # Centered Submit Files button
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        col_center = st.columns([5, 1.7, 5])
        with col_center[1]:  # Use the center column
            if st.button("✅ Submit Files", type="primary", use_container_width=True):
                st.session_state.files_submitted = True
                st.session_state.show_upload_area = False
                st.session_state.upload_opened_by_plus = False
                st.rerun()

    if st.session_state.files_submitted:
        col1, col2 = st.columns([8, 0.75])
        with col1:
            # Add the radio button for analysis type selection
            analysis_type = st.radio(
                "Choose the type of analysis you want to perform",
                ["Multi-Parameter Analysis", "Multi-File Multi-Parameter"],
                index=0,
                horizontal=True
            )
        with col2:
            if st.session_state.files_submitted and not st.session_state.show_upload_area:
                if st.button("➕ UPLOAD", type="primary", use_container_width=True, help="Upload or manage files"):
                    st.session_state.show_upload_area = True
                    st.session_state.upload_opened_by_plus = True
                    st.rerun()
        st.session_state.analysis_type = analysis_type

        if analysis_type == "Multi-Parameter Analysis":
            st.markdown("### 📈 Multi-Parameter Analysis (Single File)")

            # File selection (single file) - stateless defaults for immediate updates
            file_options = ["None"] + [f.name for f in st.session_state.uploaded_files]
            selected_file = st.selectbox(
                "Select File",
                file_options,
                index=0,
                key="multi_param_file_selector",
            )

            if selected_file == "None":
                st.info("📋 Please select a file to begin Multi-Parameter Analysis.")
                st.stop()
            
            # Update the old session state variable for compatibility
            st.session_state.multi_param_file_selection = selected_file

            file_ext = os.path.splitext(selected_file)[-1].lower()
            try:
                file = [f for f in st.session_state.uploaded_files if f.name == selected_file][0]
            except IndexError:
                st.error("Selected file not found. Please re-select the file.")
                st.stop()

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
                            st.error("No usable topics found in ULG file.")
                            st.stop()

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
                            st.stop()
                finally:
                    try:
                        os.unlink(tmp_file.name)
                    except Exception:
                        pass

            # Ensure numeric data and columns
            if df is None or df.empty:
                st.error("No data available after loading the file.")
                st.stop()

            df = ensure_seconds_column(df)
            numeric_cols = get_numeric_columns(df)

            # Remove columns that are completely empty or constant
            numeric_cols = [
                col for col in numeric_cols
                if not is_column_empty(df, col) and df[col].nunique(dropna=True) > 1
            ]

            if not numeric_cols:
                st.error("No numeric columns available for multi-parameter analysis.")
                st.stop()

            # Helper function to find column by partial name match (accessible throughout this section)
            def find_column_by_name(cols, target_name):
                """Find a column that contains the target name (case-insensitive, handles partial matches)"""
                target_lower = target_name.lower()
                # First try exact match
                for col in cols:
                    if col.lower() == target_lower:
                        return col
                # Then try partial match (column contains target)
                for col in cols:
                    if target_lower in col.lower():
                        return col
                return None
            
            # Track current file to detect file changes
            # Use the actual selected file from the widget
            selected_file_name = st.session_state.get("multi_param_file_selector", "None")
            current_file_key = f"multi_param_current_file_{selected_file_name}"
            
            # Check if file has changed or if this is first load
            file_changed = st.session_state.get("multi_param_last_file") != current_file_key
            
            # Initialize defaults on first load or when file changes
            if file_changed or "multi_param_last_file" not in st.session_state:
                # Clear existing selections to ensure fresh defaults
                if "multi_param_x_axis_selector" in st.session_state:
                    del st.session_state["multi_param_x_axis_selector"]
                if "multi_param_left_y_axis_selector" in st.session_state:
                    del st.session_state["multi_param_left_y_axis_selector"]
                if "multi_param_right_y_axis_selector" in st.session_state:
                    del st.session_state["multi_param_right_y_axis_selector"]
                # Reset smoothing to False (unselected) for Graph 1
                st.session_state["multi_param_smoothing_check"] = False
                st.session_state["multi_param_smoothing"] = False
                
                # Clear all saved graph widget states (Graph 2, 3, 4+, etc.)
                # This ensures fresh state when file changes
                keys_to_clear = [key for key in st.session_state.keys() if key.startswith("multi_param_saved_") and ("_smooth_check" in key or "_smooth_method" in key or "_smooth_window" in key or "_x_axis" in key or "_left_y" in key or "_right_y" in key)]
                for key in keys_to_clear:
                    del st.session_state[key]
                
                # Also clear graph parameter states
                graph_keys_to_clear = [key for key in st.session_state.keys() if key.startswith("multi_param_graph_")]
                for key in graph_keys_to_clear:
                    del st.session_state[key]
                
                # Reset Graph 1 defaults
                # Graph 1 defaults: X-axis = "RPM", Left Y-axis = "Thrust", Right Y-axis = "SysEffect"
                rpm_col = find_column_by_name(numeric_cols, "RPM")
                if rpm_col:
                    st.session_state["multi_param_x_axis_selector"] = rpm_col
                else:
                    st.session_state["multi_param_x_axis_selector"] = numeric_cols[0] if numeric_cols else None
                
                # Set Graph 1 Y-axis defaults
                x_axis_init = st.session_state.get("multi_param_x_axis_selector")
                y_candidates_init = [c for c in numeric_cols if c != x_axis_init]
                
                thrust_col = find_column_by_name(y_candidates_init, "Thrust")
                if thrust_col:
                    st.session_state["multi_param_left_y_axis_selector"] = [thrust_col]
                else:
                    st.session_state["multi_param_left_y_axis_selector"] = [y_candidates_init[0]] if y_candidates_init else []
                
                remaining_for_right_init = [c for c in y_candidates_init if c not in st.session_state.get("multi_param_left_y_axis_selector", [])]
                syseffect_col = find_column_by_name(remaining_for_right_init, "SysEffect")
                if syseffect_col:
                    st.session_state["multi_param_right_y_axis_selector"] = [syseffect_col]
                else:
                    st.session_state["multi_param_right_y_axis_selector"] = [remaining_for_right_init[0]] if remaining_for_right_init else []
                
                # Reset Graph 2 - clear saved graphs and initialize with defaults
                st.session_state.multi_param_saved_graphs = []
                # Graph 2 defaults: X-axis = "Torque", Left Y-axis = "Thrust", Right Y-axis = "SysEffect"
                torque_col = find_column_by_name(numeric_cols, "Torque")
                graph2_x = torque_col if torque_col else (numeric_cols[0] if numeric_cols else None)
                
                # For Graph 2, use same Thrust and SysEffect columns as Graph 1
                graph2_left = st.session_state.get("multi_param_left_y_axis_selector", []).copy()
                graph2_right = st.session_state.get("multi_param_right_y_axis_selector", []).copy()
                
                if graph2_x:
                    # Create a placeholder figure for Graph 2 (will be regenerated when displayed)
                    st.session_state.multi_param_saved_graphs.append({
                        "x_axis": graph2_x,
                        "left_y_axes": graph2_left,
                        "right_y_axes": graph2_right,
                        "smoothing_enabled": False,
                        "smoothing_method": "savgol",
                        "smoothing_window": 5,
                        "fig": None  # Will be created when displayed
                    })
                
                # Update last file tracker
                st.session_state["multi_param_last_file"] = current_file_key
                
                # Trigger rerun to apply defaults
                st.rerun()
                
            # Initialize Graph 2 with defaults if not already present (for cases where file hasn't changed)
            elif len(st.session_state.multi_param_saved_graphs) == 0:
                # Graph 2 defaults: X-axis = "Torque", Left Y-axis = "Thrust", Right Y-axis = "SysEffect"
                torque_col = find_column_by_name(numeric_cols, "Torque")
                graph2_x = torque_col if torque_col else (numeric_cols[0] if numeric_cols else None)
                
                thrust_col = find_column_by_name(numeric_cols, "Thrust")
                graph2_left = [thrust_col] if thrust_col else []
                
                syseffect_col = find_column_by_name(numeric_cols, "SysEffect")
                graph2_right = [syseffect_col] if syseffect_col else []
                
                if graph2_x:
                    # Create a placeholder figure for Graph 2 (will be regenerated when displayed)
                    st.session_state.multi_param_saved_graphs.append({
                        "x_axis": graph2_x,
                        "left_y_axes": graph2_left,
                        "right_y_axes": graph2_right,
                        "smoothing_enabled": False,
                        "smoothing_method": "savgol",
                        "smoothing_window": 5,
                        "fig": None  # Will be created when displayed
                    })

            # Tabs for Plot and Data
            tab_plot, tab_data = st.tabs(["📊 Plot", "📋 Data"])
            
            with tab_data:
                # Display data
                st.markdown("### 📋 Data")
                df_display = df.copy()
                
                # Handle duplicate column names in the original DataFrame
                df_display = fix_duplicate_columns(df_display)
                
                # Add Index column if it doesn't exist
                if 'Index' not in df_display.columns:
                    df_display.insert(0, 'Index', range(1, len(df_display) + 1))
                
                # Column selection with checkbox control
                all_cols = list(df_display.columns)
                # Remove duplicates from all_cols (in case there are any)
                all_cols = list(dict.fromkeys(all_cols))  # Preserves order while removing duplicates
                
                # Checkbox to enable column selection
                enable_column_selection = st.checkbox(
                    "Select columns to display",
                    value=False,
                    key="multi_param_enable_column_selection",
                    help="Check this to select specific columns. By default, all columns are displayed."
                )
                
                if enable_column_selection:
                    # Show multiselect when checkbox is checked
                    default_selected = all_cols[:20] if len(all_cols) > 20 else all_cols
                    selected_cols = st.multiselect(
                        "Columns",
                        all_cols,
                        default=default_selected,
                        key="multi_param_data_column_selector",
                        help="Select columns to display in the data table"
                    )
                    
                    # If no columns selected, show all
                    if not selected_cols:
                        selected_cols = all_cols.copy()
                    
                    # Remove duplicates from selected_cols and ensure Index is always first
                    selected_cols = list(dict.fromkeys(selected_cols))  # Remove duplicates
                    if 'Index' in selected_cols:
                        selected_cols.remove('Index')
                    selected_cols = ['Index'] + selected_cols
                    
                    # Ensure all selected columns exist in the dataframe
                    selected_cols = [col for col in selected_cols if col in df_display.columns]
                else:
                    # By default, show all columns
                    selected_cols = all_cols.copy()
                    # Ensure Index is always first
                    if 'Index' in selected_cols:
                        selected_cols.remove('Index')
                    selected_cols = ['Index'] + selected_cols
                
                st.dataframe(df_display[selected_cols].rename(columns=COLUMN_DISPLAY_NAMES), use_container_width=True, height=400)
                
                # Show summary statistics
                st.markdown("### 📊 Summary Statistics")
                stats_cols = [col for col in selected_cols if col != 'Index' and pd.api.types.is_numeric_dtype(df_display[col]) and not is_column_empty(df_display, col)]
                if stats_cols:
                    summary_stats = df_display[stats_cols].describe()
                    st.dataframe(summary_stats, use_container_width=True)
            
            with tab_plot:
                # Layout: parameters on left, plot in middle, table on right
                param_col, plot_col, table_col = st.columns([0.25, 0.5, 0.25])

                with param_col:
                    st.markdown("""
                    <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                        <span style='font-size: 1.2rem;'>🧮</span>
                        <span style='font-size: 1.1rem; font-weight: 600;'>Parameters - Graph 1</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # X-axis selection with default "RPM" for Graph 1
                    # Initialize default if not set (fallback for edge cases)
                    if "multi_param_x_axis_selector" not in st.session_state:
                        def find_col(cols, target):
                            target_lower = target.lower()
                            for col in cols:
                                if col.lower() == target_lower or target_lower in col.lower():
                                    return col
                            return None
                        default_x = find_col(numeric_cols, "RPM") or (numeric_cols[0] if numeric_cols else None)
                        if default_x:
                            st.session_state["multi_param_x_axis_selector"] = default_x
                    
                    # Get current selection or default
                    current_x = st.session_state.get("multi_param_x_axis_selector")
                    if not current_x or current_x not in numeric_cols:
                        def find_col(cols, target):
                            target_lower = target.lower()
                            for col in cols:
                                if col.lower() == target_lower or target_lower in col.lower():
                                    return col
                            return None
                        current_x = find_col(numeric_cols, "RPM") or (numeric_cols[0] if numeric_cols else None)
                        if current_x:
                            st.session_state["multi_param_x_axis_selector"] = current_x
                    x_axis_index = numeric_cols.index(current_x) if current_x in numeric_cols else 0
                    
                    x_axis = st.selectbox(
                        "X-Axis",
                        numeric_cols,
                        index=x_axis_index,
                        key="multi_param_x_axis_selector",
                    )

                    # Dual Y-axis selection
                    y_candidates = [c for c in numeric_cols if c != x_axis]
                    if not y_candidates:
                        st.error("Not enough numeric columns to create Y-axis parameters.")
                        st.stop()

                    # Left Y-axis selection - preserve selection when X-axis changes
                    previous_left_y_axes = st.session_state.get("multi_param_left_y_axis_selector", [])
                    # Filter out any items that are now the X-axis or not in y_candidates
                    preserved_left_y_axes = [col for col in previous_left_y_axes if col in y_candidates]
                    
                    # Allow both Y-axes to be empty - removed auto-selection logic
                    # Defaults are set on file change, not when user clears selections
                    left_y_axes = st.multiselect(
                        "Left Y-Axis Parameters",
                        y_candidates,
                        default=preserved_left_y_axes if preserved_left_y_axes else [],  # Preserve selection, excluding conflicts
                        key="multi_param_left_y_axis_selector",
                        max_selections=4,
                        help="Parameters plotted on the left Y-axis"
                    )
                    # Right Y-axis selection (exclude left Y-axis selections)
                    right_candidates = [c for c in y_candidates if c not in left_y_axes]
                    # Preserve existing right Y-axis selection, but remove any conflicts with left Y-axis
                    previous_right_y_axes = st.session_state.get("multi_param_right_y_axis_selector", [])
                    # Filter out any items that are now in left_y_axes or not in right_candidates
                    preserved_right_y_axes = [col for col in previous_right_y_axes if col in right_candidates]
                    
                    # Allow right Y-axis to be empty - removed auto-selection logic
                    right_y_axes = st.multiselect(
                        "Right Y-Axis Parameters",
                        right_candidates,
                        default=preserved_right_y_axes if preserved_right_y_axes else [],  # Preserve selection, excluding conflicts
                        key="multi_param_right_y_axis_selector",
                        max_selections=4,
                        help="Parameters plotted on the right Y-axis"
                    )

                    # Check if Y-axis parameters are selected (auto-selection happens before widgets, so this should always be True)
                    has_y_axes = bool(left_y_axes or right_y_axes)

                    # Smoothing controls in parameter column
                    # Use False as default (unselected)
                    # The widget with key automatically manages its state in st.session_state[key]
                    smoothing_enabled = st.checkbox(
                        "Enable Data Smoothing",
                        value=False,  # Default to unselected
                        key="multi_param_smoothing_check",
                        help="Smooth data to create continuous lines instead of zig-zag patterns."
                    )
                    # Store the value for reference (sync with widget state)
                    st.session_state["multi_param_smoothing"] = smoothing_enabled
                    
                    smoothing_method = "savgol"
                    smoothing_window = 5
                    if smoothing_enabled:
                        smoothing_method = st.selectbox(
                            "Smoothing Method",
                            ["linear", "cubic", "moving_average", "savgol"],
                            index=3,  # Default to savgol for best smoothing
                            key="multi_param_smoothing_method",
                            help="Method for smoothing: linear (fast), cubic (smooth), moving_average (noise reduction), savgol (best for noisy data)",
                        )
                        if smoothing_method in ["moving_average", "savgol"]:
                            smoothing_window = st.slider(
                                "Smoothing Window",
                                min_value=3,
                                max_value=21,
                                value=st.session_state.get("multi_param_smoothing_window", 5),
                                step=2,
                                key="multi_param_smoothing_window",
                                help="Larger window = more smoothing (must be odd number)",
                            )
                            # Ensure odd (but don't modify session state - widget manages it)
                            if smoothing_window % 2 == 0:
                                smoothing_window += 1  # Use odd value for smoothing

                    # X-axis range - use full data range automatically
                    x_min = float(df[x_axis].min())
                    x_max = float(df[x_axis].max())

                    # Left Y-axis range - calculate ONLY from currently selected parameters, starting from 0
                    if left_y_axes and len(left_y_axes) > 0:
                        # Only use columns that are actually selected and exist in dataframe
                        valid_left_cols = [col for col in left_y_axes if col in df.columns]
                        if valid_left_cols:
                            left_y_values = []
                            for col in valid_left_cols:
                                left_y_values.extend(df[col].dropna().tolist())
                            if left_y_values:
                                left_y_min = float(min(left_y_values))
                                left_y_max = float(max(left_y_values))
                                # Ensure minimum is 0 or less
                                if left_y_min > 0:
                                    left_y_min = 0.0
                                # Add small buffer above max for better visualization
                                if left_y_max > 0:
                                    left_y_buffer = max(left_y_max * 0.05, left_y_max * 0.02)
                                    left_y_max = left_y_max + left_y_buffer
                                else:
                                    left_y_max = 1.0
                            else:
                                left_y_min, left_y_max = 0.0, 1.0
                        else:
                            left_y_min, left_y_max = 0.0, 1.0
                    else:
                        left_y_min, left_y_max = None, None  # No range when nothing selected

                    # Right Y-axis range - calculate ONLY from currently selected parameters, starting from 0
                    if right_y_axes and len(right_y_axes) > 0:
                        # Only use columns that are actually selected and exist in dataframe
                        valid_right_cols = [col for col in right_y_axes if col in df.columns]
                        if valid_right_cols:
                            right_y_values = []
                            for col in valid_right_cols:
                                right_y_values.extend(df[col].dropna().tolist())
                            if right_y_values:
                                right_y_min = float(min(right_y_values))
                                right_y_max = float(max(right_y_values))
                                # Ensure minimum is 0 or less
                                if right_y_min > 0:
                                    right_y_min = 0.0
                                # Add small buffer above max for better visualization
                                if right_y_max > 0:
                                    right_y_buffer = max(right_y_max * 0.05, right_y_max * 0.02)
                                    right_y_max = right_y_max + right_y_buffer
                                else:
                                    right_y_max = 1.0
                            else:
                                right_y_min, right_y_max = 0.0, 1.0
                        else:
                            right_y_min, right_y_max = 0.0, 1.0
                    else:
                        right_y_min, right_y_max = None, None  # No range when nothing selected

                with plot_col:
                    # Initialize fig variable
                    fig = None
                    
                    # Only create and display plot if Y-axis parameters are selected
                    if has_y_axes:
                        # Use full dataframe (no filtering needed since we're using full range)
                        df_filtered = df.copy()

                        if df_filtered.empty:
                            st.warning("No data available to display.")
                        else:
                            # Apply smoothing if enabled
                            if smoothing_enabled:
                                all_y_cols = (left_y_axes or []) + (right_y_axes or [])
                                df_filtered = smooth_data_for_plotting(
                                    df_filtered,
                                    x_axis,
                                    all_y_cols,
                                    method=smoothing_method,
                                    smoothing_window=smoothing_window,
                                )

                            # Create figure with dual Y-axes
                            fig = make_subplots(specs=[[{"secondary_y": True}]])
                            
                            # Color palette for different series
                            colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
                            color_idx = 0

                            # Add traces for left Y-axis
                            if left_y_axes:
                                for y_col in left_y_axes:
                                    # Remove NaN values for clean plotting
                                    plot_data = df_filtered[[x_axis, y_col]].dropna()
                                    if not plot_data.empty:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=plot_data[x_axis],
                                                y=plot_data[y_col],
                                                mode="lines+markers",
                                                name=get_display_name(y_col),
                                                line=dict(color=colors[color_idx % len(colors)], width=2),
                                                marker=dict(size=4, color=colors[color_idx % len(colors)]),
                                            ),
                                            secondary_y=False,
                                        )
                                        color_idx += 1

                            # Add traces for right Y-axis
                            if right_y_axes:
                                for y_col in right_y_axes:
                                    # Remove NaN values for clean plotting
                                    plot_data = df_filtered[[x_axis, y_col]].dropna()
                                    if not plot_data.empty:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=plot_data[x_axis],
                                                y=plot_data[y_col],
                                                mode="lines+markers",
                                                name=get_display_name(y_col),
                                                line=dict(color=colors[color_idx % len(colors)], width=2),
                                                marker=dict(size=4, color=colors[color_idx % len(colors)]),
                                            ),
                                            secondary_y=True,
                                        )
                                        color_idx += 1

                            # Set X-axis title
                            fig.update_xaxes(title_text=get_axis_title(x_axis))

                            # Set Y-axis titles and ranges with grid on left Y-axis
                            if left_y_axes and left_y_min is not None and left_y_max is not None:
                                left_display_names = [get_display_name(col) for col in left_y_axes]
                                left_title = " & ".join(left_display_names) if len(left_display_names) <= 2 else f"{left_display_names[0]} & {len(left_display_names)-1} more"
                                fig.update_yaxes(
                                    title_text=left_title,
                                    range=[left_y_min, left_y_max],
                                    secondary_y=False,
                                    showgrid=True,  # Show grid on left Y-axis
                                    gridcolor='lightgray',
                                    gridwidth=1
                                )
                            
                            if right_y_axes and right_y_min is not None and right_y_max is not None:
                                right_display_names = [get_display_name(col) for col in right_y_axes]
                                right_title = " & ".join(right_display_names) if len(right_display_names) <= 2 else f"{right_display_names[0]} & {len(right_display_names)-1} more"
                                fig.update_yaxes(
                                    title_text=right_title,
                                    range=[right_y_min, right_y_max],
                                    secondary_y=True,
                                    showgrid=False  # No grid on right Y-axis (like matplotlib twinx)
                                )
                            
                            # Also add grid to X-axis
                            fig.update_xaxes(
                                showgrid=True,
                                gridcolor='lightgray',
                                gridwidth=1
                            )

                            # Generate title
                            title_parts = []
                            if left_y_axes:
                                left_title_parts = [get_display_name(col) for col in left_y_axes]
                                title_parts.append(" & ".join(left_title_parts))
                            if right_y_axes:
                                right_title_parts = [get_display_name(col) for col in right_y_axes]
                                title_parts.append(" & ".join(right_title_parts))
                            title_text = " & ".join(title_parts) + f" Vs {get_axis_title(x_axis)}" if title_parts else f"Multi-Parameter Analysis: {get_axis_title(x_axis)}"

                            # Update layout
                            fig.update_layout(
                                title=dict(
                                    text=title_text,
                                    x=0.5,
                                    xanchor='center',
                                    font=dict(size=16)
                                ),
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1,
                                    font=dict(size=10)
                                ),
                                template="plotly_white",
                                margin=dict(l=60, r=60, t=60, b=50),
                                hovermode='x unified',
                            )

                            # Show main (current) graph as Graph 1
                            st.markdown("##### Graph 1")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Show warning if no Y-axis parameters selected
                        st.warning("⚠️ Please select at least one parameter for Left or Right Y-axis to display the plot.")

                # Table column showing parameter values
                with table_col:
                    st.markdown("""
                    <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                        <span style='font-size: 1.2rem;'>📊</span>
                        <span style='font-size: 1.1rem; font-weight: 600;'>Graph 1 - Parameter Values</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create a table with actual parameter values
                    if x_axis:
                        # Get the data columns to display (always show X-axis, add Y-axes if selected)
                        cols_to_show = [x_axis]
                        if left_y_axes:
                            cols_to_show.extend(left_y_axes)
                        if right_y_axes:
                            cols_to_show.extend(right_y_axes)
                        
                        # Filter to only show columns that exist in dataframe
                        cols_to_show = [col for col in cols_to_show if col in df.columns]
                        
                        if cols_to_show:
                            # Get the data (use filtered data if smoothing is enabled and Y-axes are selected)
                            if smoothing_enabled and has_y_axes and 'df_filtered' in locals():
                                display_df = df_filtered.copy()
                            else:
                                display_df = df.copy()
                            
                            # Select only the relevant columns
                            table_df = display_df[cols_to_show].copy()
                            
                            # Rename columns with display names
                            table_df.columns = [get_display_name(col) for col in table_df.columns]
                            
                            # Display the table
                            st.dataframe(
                                table_df,
                                use_container_width=True,
                                hide_index=True,
                                height=400
                            )
                        else:
                            st.info("No valid parameters selected.")
                    else:
                        st.info("Please select X-axis to view data.")

                # Show saved graphs below with their parameter summary (outside plot_col for proper layout)
                # Graph 2 is always shown (from saved_graphs[0] if it exists, or with defaults)
                # Graph 3 is always shown as an empty template
                # Graph 4+ are shown from saved_graphs[1:] if they exist
                
                # Helper function to render a graph with its parameters
                def render_saved_graph(idx, saved_data=None, is_empty=False):
                    graph_key = f"multi_param_graph_{idx}"
                    
                    # Initialize graph parameters
                    if graph_key not in st.session_state:
                        if is_empty:
                            # Empty graph (Graph 3) - no defaults
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
                    
                    st.markdown(f"##### Graph {idx}")
                    param_col_saved, plot_col_saved, table_col_saved = st.columns([0.25, 0.5, 0.25])

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
                        graph_params["x_axis"] = saved_x

                        # Y-axis candidates based on saved X
                        y_candidates_saved = [c for c in numeric_cols if c != saved_x]
                        
                        # Update graph_params if X-axis changed
                        if graph_params.get("x_axis") != saved_x:
                            # Filter out invalid Y-axis selections
                            graph_params["left_y_axes"] = [c for c in graph_params.get("left_y_axes", []) if c in y_candidates_saved]
                            graph_params["right_y_axes"] = [c for c in graph_params.get("right_y_axes", []) if c in y_candidates_saved]

                        # Left Y-axis (editable multiselect)
                        # Get current selection from session state or graph_params
                        if f"multi_param_saved_{idx}_left_y" in st.session_state:
                            previous_left = st.session_state[f"multi_param_saved_{idx}_left_y"]
                        else:
                            previous_left = graph_params.get("left_y_axes", [])
                        
                        preserved_left = [col for col in previous_left if col in y_candidates_saved]
                        # Update session state BEFORE widget reads it (Streamlit uses session state value, not default)
                        st.session_state[f"multi_param_saved_{idx}_left_y"] = preserved_left
                        
                        saved_left = st.multiselect(
                            "Left Y-Axis Parameters",
                            y_candidates_saved,
                            default=preserved_left if preserved_left else [],
                            key=f"multi_param_saved_{idx}_left_y",
                            max_selections=4,
                        )
                        graph_params["left_y_axes"] = saved_left

                        # Right Y-axis (editable multiselect)
                        right_candidates_saved = [c for c in y_candidates_saved if c not in saved_left]
                        
                        # Get current selection from session state or graph_params
                        if f"multi_param_saved_{idx}_right_y" in st.session_state:
                            previous_right = st.session_state[f"multi_param_saved_{idx}_right_y"]
                        else:
                            previous_right = graph_params.get("right_y_axes", [])
                        
                        preserved_right = [col for col in previous_right if col in right_candidates_saved]
                        # Update session state BEFORE widget reads it (Streamlit uses session state value, not default)
                        st.session_state[f"multi_param_saved_{idx}_right_y"] = preserved_right
                        
                        saved_right = st.multiselect(
                            "Right Y-Axis Parameters",
                            right_candidates_saved,
                            default=preserved_right if preserved_right else [],
                            key=f"multi_param_saved_{idx}_right_y",
                            max_selections=4,
                        )
                        graph_params["right_y_axes"] = saved_right

                        # Note: We'll check for Y-axis parameters in the plot section

                        # Smoothing controls (editable)
                        # Initialize widget state from graph_params if not already set
                        smooth_check_key = f"multi_param_saved_{idx}_smooth_check"
                        if smooth_check_key not in st.session_state:
                            st.session_state[smooth_check_key] = graph_params.get("smoothing_enabled", False)
                        
                        # The widget with key automatically manages its state in st.session_state[key]
                        saved_smooth_enabled = st.checkbox(
                            "Enable Data Smoothing",
                            value=False,  # Stateless default (widget uses session state if key exists)
                            key=smooth_check_key,
                        )
                        # Sync the widget state to graph_params
                        graph_params["smoothing_enabled"] = saved_smooth_enabled
                        
                        saved_smooth_method = "savgol"
                        saved_smooth_window = 5
                        if saved_smooth_enabled:
                            method_options = ["linear", "cubic", "moving_average", "savgol"]
                            current_method = graph_params.get("smoothing_method", "savgol")
                            method_index = method_options.index(current_method) if current_method in method_options else 3
                            saved_smooth_method = st.selectbox(
                                "Smoothing Method",
                                method_options,
                                index=method_index,
                                key=f"multi_param_saved_{idx}_smooth_method",
                            )
                            graph_params["smoothing_method"] = saved_smooth_method
                            
                            if saved_smooth_method in ["moving_average", "savgol"]:
                                saved_smooth_window = st.slider(
                                    "Smoothing Window",
                                    min_value=3,
                                    max_value=21,
                                    value=graph_params.get("smoothing_window", 5),
                                    step=2,
                                    key=f"multi_param_saved_{idx}_smooth_window",
                                )
                                if saved_smooth_window % 2 == 0:
                                    saved_smooth_window += 1
                                graph_params["smoothing_window"] = saved_smooth_window

                    with plot_col_saved:
                        # Recreate plot with current parameters
                        df_filtered_saved = df.copy()
                        
                        # Apply smoothing if enabled
                        if saved_smooth_enabled:
                            all_y_cols_saved = (saved_left or []) + (saved_right or [])
                            df_filtered_saved = smooth_data_for_plotting(
                                df_filtered_saved,
                                saved_x,
                                all_y_cols_saved,
                                method=saved_smooth_method,
                                smoothing_window=saved_smooth_window,
                            )
                        
                        # Create figure with dual Y-axes
                        saved_fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        # Check if Y-axis parameters are selected
                        if not saved_left and not saved_right:
                            # Show empty plot with warning
                            saved_fig.update_layout(
                                title=dict(text=f"Graph {idx}: Please select Y-axis parameters", x=0.5, xanchor='center', font=dict(size=16)),
                                template="plotly_white",
                                xaxis=dict(title=get_axis_title(saved_x)),
                            )
                            st.plotly_chart(saved_fig, use_container_width=True)
                            # Don't return - continue to show table with X-axis data
                        
                        colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
                        color_idx = 0
                        
                        # Add traces for left Y-axis
                        if saved_left:
                            for y_col in saved_left:
                                plot_data = df_filtered_saved[[saved_x, y_col]].dropna()
                                if not plot_data.empty:
                                    saved_fig.add_trace(
                                        go.Scatter(
                                            x=plot_data[saved_x],
                                            y=plot_data[y_col],
                                            mode="lines+markers",
                                            name=get_display_name(y_col),
                                            line=dict(color=colors[color_idx % len(colors)], width=2),
                                            marker=dict(size=4, color=colors[color_idx % len(colors)]),
                                        ),
                                        secondary_y=False,
                                    )
                                    color_idx += 1
                        
                        # Add traces for right Y-axis
                        if saved_right:
                            for y_col in saved_right:
                                plot_data = df_filtered_saved[[saved_x, y_col]].dropna()
                                if not plot_data.empty:
                                    saved_fig.add_trace(
                                        go.Scatter(
                                            x=plot_data[saved_x],
                                            y=plot_data[y_col],
                                            mode="lines+markers",
                                            name=get_display_name(y_col),
                                            line=dict(color=colors[color_idx % len(colors)], width=2),
                                            marker=dict(size=4, color=colors[color_idx % len(colors)]),
                                        ),
                                        secondary_y=True,
                                    )
                                    color_idx += 1
                        
                        # Set axes
                        saved_fig.update_xaxes(title_text=get_axis_title(saved_x))
                        
                        # Calculate ranges
                        if saved_left:
                            left_y_values = []
                            for col in saved_left:
                                if col in df.columns:
                                    left_y_values.extend(df[col].dropna().tolist())
                            if left_y_values:
                                left_y_min = float(min(left_y_values))
                                left_y_max = float(max(left_y_values))
                                if left_y_min > 0:
                                    left_y_min = 0.0
                                if left_y_max > 0:
                                    left_y_max = left_y_max + max(left_y_max * 0.05, left_y_max * 0.02)
                                else:
                                    left_y_max = 1.0
                                left_display_names = [get_display_name(col) for col in saved_left]
                                left_title = " & ".join(left_display_names) if len(left_display_names) <= 2 else f"{left_display_names[0]} & {len(left_display_names)-1} more"
                                saved_fig.update_yaxes(
                                    title_text=left_title,
                                    range=[left_y_min, left_y_max],
                                    secondary_y=False,
                                    showgrid=True,
                                    gridcolor='lightgray',
                                    gridwidth=1
                                )
                        
                        if saved_right:
                            right_y_values = []
                            for col in saved_right:
                                if col in df.columns:
                                    right_y_values.extend(df[col].dropna().tolist())
                            if right_y_values:
                                right_y_min = float(min(right_y_values))
                                right_y_max = float(max(right_y_values))
                                if right_y_min > 0:
                                    right_y_min = 0.0
                                if right_y_max > 0:
                                    right_y_max = right_y_max + max(right_y_max * 0.05, right_y_max * 0.02)
                                else:
                                    right_y_max = 1.0
                                right_display_names = [get_display_name(col) for col in saved_right]
                                right_title = " & ".join(right_display_names) if len(right_display_names) <= 2 else f"{right_display_names[0]} & {len(right_display_names)-1} more"
                                saved_fig.update_yaxes(
                                    title_text=right_title,
                                    range=[right_y_min, right_y_max],
                                    secondary_y=True,
                                    showgrid=False
                                )
                        
                        saved_fig.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=1)
                        
                        # Generate title
                        title_parts = []
                        if saved_left:
                            left_title_parts = [get_display_name(col) for col in saved_left]
                            title_parts.append(" & ".join(left_title_parts))
                        if saved_right:
                            right_title_parts = [get_display_name(col) for col in saved_right]
                            title_parts.append(" & ".join(right_title_parts))
                        title_text = " & ".join(title_parts) + f" Vs {get_axis_title(saved_x)}" if title_parts else f"Graph {idx}: {get_axis_title(saved_x)}"
                        
                        saved_fig.update_layout(
                            title=dict(text=title_text, x=0.5, xanchor='center', font=dict(size=16)),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                            template="plotly_white",
                            margin=dict(l=60, r=60, t=60, b=50),
                            hovermode='x unified',
                        )
                        
                        st.plotly_chart(saved_fig, use_container_width=True)
                    
                    # Table column showing parameter values for saved graphs
                    with table_col_saved:
                        st.markdown(f"""
                        <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                            <span style='font-size: 1.2rem;'>📊</span>
                            <span style='font-size: 1.1rem; font-weight: 600;'>Graph {idx} - Parameter Values</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create a table with actual parameter values
                        # Get the data columns to display (always show X-axis, add Y-axes if selected)
                        cols_to_show = [saved_x]
                        if saved_left:
                            cols_to_show.extend(saved_left)
                        if saved_right:
                            cols_to_show.extend(saved_right)
                        
                        # Filter to only show columns that exist in dataframe
                        cols_to_show = [col for col in cols_to_show if col in df.columns]
                        
                        if cols_to_show:
                            # Get the data (use filtered data if smoothing is enabled)
                            if saved_smooth_enabled and saved_left or saved_right:
                                display_df = df_filtered_saved.copy()
                            else:
                                display_df = df.copy()
                            
                            # Select only the relevant columns
                            table_df = display_df[cols_to_show].copy()
                            
                            # Rename columns with display names
                            table_df.columns = [get_display_name(col) for col in table_df.columns]
                            
                            # Display the table
                            st.dataframe(
                                table_df,
                                use_container_width=True,
                                hide_index=True,
                                height=400
                            )
                        else:
                            st.info("No valid parameters selected.")
                
                # Always show Graph 2 (from saved_graphs[0] if it exists, or with defaults)
                graph2_saved = st.session_state.multi_param_saved_graphs[0] if len(st.session_state.multi_param_saved_graphs) > 0 else None
                if graph2_saved:
                    render_saved_graph(2, graph2_saved)
                else:
                    # Graph 2 defaults: X-axis = "Torque", Left Y-axis = "Thrust", Right Y-axis = "SysEffect"
                    # Use helper function to find columns (handles partial matches)
                    torque_col = find_column_by_name(numeric_cols, "Torque")
                    thrust_col = find_column_by_name(numeric_cols, "Thrust")
                    syseffect_col = find_column_by_name(numeric_cols, "SysEffect")
                    
                    graph2_defaults = {
                        "x_axis": torque_col if torque_col else (numeric_cols[0] if numeric_cols else None),
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
                btn_col_add, btn_col_remove = st.columns([0.15, 0.15])
                with btn_col_add:
                    # Always allow adding graph (Graph 4+) - creates empty graph
                    if st.button("➕ Add Graph", key="multi_param_add_graph"):
                        # Create empty graph (no Y-axis parameters selected)
                        # Use first available column as X-axis default
                        save_x_axis = numeric_cols[0] if numeric_cols else None
                        save_left = []  # Empty left Y-axis
                        save_right = []  # Empty right Y-axis
                        
                        # Store empty graph parameters (no figure created yet)
                        if save_x_axis:
                            st.session_state.multi_param_saved_graphs.append(
                                {
                                    "x_axis": save_x_axis,
                                    "left_y_axes": save_left,
                                    "right_y_axes": save_right,
                                    "smoothing_enabled": False,
                                    "smoothing_method": "savgol",
                                    "smoothing_window": 5,
                                    "fig": None  # No figure yet - will be created when user selects parameters
                                }
                            )
                            st.rerun()
                with btn_col_remove:
                    if st.button("➖ Remove Last Graph", key="multi_param_remove_graph"):
                        # Only remove Graph 4+ (from saved_graphs[1:]), not Graph 2 or 3
                        if len(st.session_state.multi_param_saved_graphs) > 1:
                            # Remove the last graph from saved_graphs[1:] (Graph 4+)
                            removed_idx = len(st.session_state.multi_param_saved_graphs)
                            graph_key = f"multi_param_graph_{removed_idx}"
                            if graph_key in st.session_state:
                                del st.session_state[graph_key]
                            st.session_state.multi_param_saved_graphs.pop()
                            st.rerun()
                        else:
                            st.info("Graph 2 and Graph 3 cannot be removed. Only Graph 4 and above can be removed.")

        elif analysis_type == "Multi-File Multi-Parameter":
            st.markdown("### 📈 Multi-File Analysis")
            st.markdown(f"<p style='color: #666; font-size: 0.9rem;'>Compare multiple files with dual Y-axis parameters. Select multiple files and choose parameters for left and right Y-axes.</p>", unsafe_allow_html=True)

            # File selection (multiple files) - stateless defaults
            available_files = [f.name for f in st.session_state.uploaded_files]
            selected_files = st.multiselect(
                "Select Files (up to 8)",
                options=available_files,
                default=[],
                key="multi_file_multi_param_file_selector",
                help="Select multiple files to compare. All files should have similar structure."
            )

            # Limit to 8 files
            if len(selected_files) > 8:
                st.warning(f"⚠️ Maximum 8 files allowed. Only the first 8 files will be used.")
                selected_files = selected_files[:8]

            if not selected_files or len(selected_files) < 1:
                st.info("📋 Please select at least one file to begin Multi-File Multi-Parameter Analysis.")
                st.stop()

            # Load all selected files
            file_data = {}
            file_extensions = {}
            all_numeric_cols = set()
            
            for filename in selected_files:
                try:
                    file = [f for f in st.session_state.uploaded_files if f.name == filename][0]
                    file_ext = os.path.splitext(filename)[-1].lower()
                    file_extensions[filename] = file_ext
                    
                    file.seek(0)
                    content = file.read()
                    file.seek(0)
                    
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
                                    st.warning(f"⚠️ No usable topics found in {filename}")
                                    continue
                                
                                # For ULG, we'll use the first available topic or let user select
                                topic_keys = list(dfs_dict.keys())
                                selected_topic = st.session_state.get(f"multi_file_multi_param_topic_{filename}", topic_keys[0] if topic_keys else None)
                                
                                if selected_topic and selected_topic in dfs_dict:
                                    df = dfs_dict[selected_topic].copy()
                                else:
                                    df = dfs_dict[topic_keys[0]].copy() if topic_keys else None
                                
                                if df is not None:
                                    df = ensure_seconds_column(df)
                                    file_data[filename] = df
                                    numeric_cols = get_numeric_columns(df)
                                    all_numeric_cols.update(numeric_cols)
                            else:
                                df, _ = load_data(tmp_file.name, file_ext, key_suffix=f"_multi_file_{filename}")
                                if df is not None and not df.empty:
                                    df = ensure_seconds_column(df)
                                    file_data[filename] = df
                                    numeric_cols = get_numeric_columns(df)
                                    all_numeric_cols.update(numeric_cols)
                                else:
                                    st.warning(f"⚠️ No data found in {filename}")
                        finally:
                            try:
                                os.unlink(tmp_file.name)
                            except:
                                pass
                except Exception as e:
                    st.error(f"Error loading file {filename}: {str(e)}")
                    continue

            if not file_data:
                st.error("❌ No files could be loaded. Please check your file selections.")
                st.stop()

            # Find common numeric columns across all files
            common_numeric_cols = list(all_numeric_cols)
            for filename, df in file_data.items():
                df_numeric = set(get_numeric_columns(df))
                common_numeric_cols = [col for col in common_numeric_cols if col in df_numeric and not is_column_empty(df, col) and df[col].nunique(dropna=True) > 1]

            if not common_numeric_cols:
                st.error("❌ No common numeric columns found across selected files.")
                st.stop()

            # Tabs for Plot and Data
            tab_plot, tab_data = st.tabs(["📊 Plot", "📋 Data"])
            
            with tab_data:
                # Display data for all selected files
                st.markdown("### 📋 Data")
                num_files = len(selected_files)
                if num_files == 0:
                    st.warning("No files selected for comparison.")
                else:
                    # Column selection with checkbox control
                    all_available_cols = list(set([col for df in file_data.values() for col in df.columns if col in common_numeric_cols or col == 'Index' or 'timestamp' in col.lower()]))
                    if 'Index' not in all_available_cols:
                        all_available_cols = ['Index'] + all_available_cols
                    
                    # Checkbox to enable column selection
                    enable_column_selection = st.checkbox(
                        "Select columns to display",
                        value=False,
                        key="multi_file_multi_param_enable_column_selection",
                        help="Check this to select specific columns. By default, all columns are displayed."
                    )
                    
                    if enable_column_selection:
                        # Show multiselect when checkbox is checked
                        default_selected = ['Index'] + (common_numeric_cols[:15] if len(common_numeric_cols) > 15 else common_numeric_cols)
                        selected_cols = st.multiselect(
                            "Columns",
                            all_available_cols,
                            default=default_selected,
                            key="multi_file_multi_param_data_column_selector",
                            help="Select columns to display in the data tables"
                        )
                        
                        # If no columns selected, show common columns
                        if not selected_cols:
                            selected_cols = ['Index'] + common_numeric_cols[:15] if len(common_numeric_cols) > 15 else ['Index'] + common_numeric_cols
                        
                        # Remove duplicates from selected_cols and ensure Index is always first
                        selected_cols = list(dict.fromkeys(selected_cols))  # Remove duplicates
                        if 'Index' in selected_cols:
                            selected_cols.remove('Index')
                        selected_cols = ['Index'] + selected_cols
                    else:
                        # By default, show all available columns
                        selected_cols = all_available_cols.copy()
                        # Ensure Index is always first
                        if 'Index' in selected_cols:
                            selected_cols.remove('Index')
                        selected_cols = ['Index'] + selected_cols
                    
                    # Create columns dynamically based on number of files (max 4 per row)
                    cols_per_row = min(num_files, 4)
                    num_rows = (num_files + cols_per_row - 1) // cols_per_row
                    
                    for row in range(num_rows):
                        cols = st.columns(cols_per_row)
                        for col_idx in range(cols_per_row):
                            file_idx = row * cols_per_row + col_idx
                            if file_idx < num_files:
                                filename = selected_files[file_idx]
                                df = file_data.get(filename)
                                
                                with cols[col_idx]:
                                    st.markdown(f"<h4 style='font-size: 16px;'>{filename}</h4>", unsafe_allow_html=True)
                                    if df is not None and hasattr(df, 'empty') and not df.empty:
                                        df_display = df.copy()
                                        
                                        # Handle duplicate column names in the original DataFrame
                                        df_display = fix_duplicate_columns(df_display)
                                        
                                        # Add Index column if it doesn't exist
                                        if 'Index' not in df_display.columns:
                                            df_display.insert(0, 'Index', range(1, len(df_display) + 1))
                                        
                                        # Show only selected columns that exist in this dataframe
                                        display_cols = [col for col in selected_cols if col in df_display.columns]
                                        # Remove duplicates from display_cols
                                        display_cols = list(dict.fromkeys(display_cols))
                                        
                                        st.dataframe(
                                            df_display[display_cols].rename(columns=COLUMN_DISPLAY_NAMES),
                                            use_container_width=True,
                                            height=400
                                        )
                                        
                                        # Show summary statistics
                                        st.markdown(f"**Summary Statistics for {filename}**")
                                        stats_cols = [col for col in display_cols if col != 'Index' and pd.api.types.is_numeric_dtype(df_display[col]) and not is_column_empty(df_display, col)]
                                        if stats_cols:
                                            summary_stats = df_display[stats_cols].describe()
                                            st.dataframe(summary_stats, use_container_width=True)
                                    else:
                                        st.warning(f"No data loaded for {filename}.")
            
            with tab_plot:
                # Layout: parameters on left, plot on right
                param_col, plot_col = st.columns([0.25, 0.75])

                with param_col:
                    st.markdown("""
                    <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                        <span style='font-size: 1.2rem;'>🧮</span>
                        <span style='font-size: 1.1rem; font-weight: 600;'>Parameters</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # X-axis selection (stateless default = first common numeric column)
                    x_axis = st.selectbox(
                        "X-Axis",
                        common_numeric_cols,
                        index=0,
                        key="multi_file_multi_param_x_axis_selector",
                    )

                    # Dual Y-axis selection
                    y_candidates = [c for c in common_numeric_cols if c != x_axis]
                    if not y_candidates:
                        st.error("Not enough common numeric columns to create Y-axis parameters.")
                        st.stop()

                    # Left Y-axis selection - preserve selection when X-axis changes
                    previous_left_y_axes = st.session_state.get("multi_file_multi_param_left_y_axis_selector", [])
                    # Filter out any items that are now the X-axis or not in y_candidates
                    preserved_left_y_axes = [col for col in previous_left_y_axes if col in y_candidates]
                    # Update session state BEFORE widget reads it (Streamlit uses session state value, not default)
                    if "multi_file_multi_param_left_y_axis_selector" in st.session_state:
                        st.session_state["multi_file_multi_param_left_y_axis_selector"] = preserved_left_y_axes
                    left_y_axes = st.multiselect(
                        "Left Y-Axis Parameters",
                        y_candidates,
                        default=preserved_left_y_axes if preserved_left_y_axes else [],  # Preserve selection, excluding conflicts
                        key="multi_file_multi_param_left_y_axis_selector",
                        max_selections=4,
                        help="Parameters plotted on the left Y-axis for all files"
                    )
                    # Right Y-axis selection
                    right_candidates = [c for c in y_candidates if c not in left_y_axes]
                    # Preserve existing right Y-axis selection, but remove any conflicts with left Y-axis
                    previous_right_y_axes = st.session_state.get("multi_file_multi_param_right_y_axis_selector", [])
                    # Filter out any items that are now in left_y_axes or not in right_candidates
                    preserved_right_y_axes = [col for col in previous_right_y_axes if col in right_candidates]
                    # Update session state BEFORE widget reads it (Streamlit uses session state value, not default)
                    if "multi_file_multi_param_right_y_axis_selector" in st.session_state:
                        st.session_state["multi_file_multi_param_right_y_axis_selector"] = preserved_right_y_axes
                    right_y_axes = st.multiselect(
                        "Right Y-Axis Parameters",
                        right_candidates,
                        default=preserved_right_y_axes if preserved_right_y_axes else [],  # Preserve selection, excluding conflicts
                        key="multi_file_multi_param_right_y_axis_selector",
                        max_selections=4,
                        help="Parameters plotted on the right Y-axis for all files"
                    )

                    if not left_y_axes and not right_y_axes:
                        st.warning("Please select at least one parameter for Left or Right Y-axis.")
                        st.stop()

                    # Calculate ranges ONLY from currently selected parameters across all files
                    all_x_values = []
                    all_left_y_values = []
                    all_right_y_values = []
                    
                    for df in file_data.values():
                        if x_axis in df.columns:
                            all_x_values.extend(df[x_axis].dropna().tolist())
                        # Only collect values from actually selected left Y-axis parameters
                        if left_y_axes and len(left_y_axes) > 0:
                            for col in left_y_axes:
                                if col in df.columns:
                                    all_left_y_values.extend(df[col].dropna().tolist())
                        # Only collect values from actually selected right Y-axis parameters
                        if right_y_axes and len(right_y_axes) > 0:
                            for col in right_y_axes:
                                if col in df.columns:
                                    all_right_y_values.extend(df[col].dropna().tolist())

                    # X-axis range - use full data range automatically
                    if all_x_values:
                        x_min = float(min(all_x_values))
                        x_max = float(max(all_x_values))
                    else:
                        x_min, x_max = 0.0, 1.0

                    # Left Y-axis range - calculate ONLY from currently selected parameters, starting from 0
                    if all_left_y_values:
                        left_y_min = float(min(all_left_y_values))
                        left_y_max = float(max(all_left_y_values))
                        # Ensure minimum is 0 or less
                        if left_y_min > 0:
                            left_y_min = 0.0
                        # Add small buffer above max for better visualization
                        if left_y_max > 0:
                            left_y_buffer = max(left_y_max * 0.05, left_y_max * 0.02)
                            left_y_max = left_y_max + left_y_buffer
                        else:
                            left_y_max = 1.0
                    else:
                        left_y_min, left_y_max = None, None  # No range when nothing selected

                    # Right Y-axis range - calculate ONLY from currently selected parameters, starting from 0
                    if all_right_y_values:
                        right_y_min = float(min(all_right_y_values))
                        right_y_max = float(max(all_right_y_values))
                        # Ensure minimum is 0 or less
                        if right_y_min > 0:
                            right_y_min = 0.0
                        # Add small buffer above max for better visualization
                        if right_y_max > 0:
                            right_y_buffer = max(right_y_max * 0.05, right_y_max * 0.02)
                            right_y_max = right_y_max + right_y_buffer
                        else:
                            right_y_max = 1.0
                    else:
                        right_y_min, right_y_max = None, None  # No range when nothing selected

                with plot_col:
                    # Add smoothing options
                    smoothing_col1, smoothing_col2 = st.columns([1, 1])
                    with smoothing_col1:
                        smoothing_enabled = st.checkbox(
                            "Enable Data Smoothing",
                            value=st.session_state.get("multi_file_multi_param_smoothing", True),
                            key="multi_file_multi_param_smoothing_check",
                            help="Smooth data to create continuous lines instead of zig-zag patterns.",
                        )
                        st.session_state["multi_file_multi_param_smoothing"] = smoothing_enabled
                    
                    smoothing_method = "savgol"
                    smoothing_window = 5
                    if smoothing_enabled:
                        with smoothing_col2:
                            smoothing_method = st.selectbox(
                                "Smoothing Method",
                                ["linear", "cubic", "moving_average", "savgol"],
                                index=3,  # Default to savgol for best smoothing
                                key="multi_file_multi_param_smoothing_method",
                                help="Method for smoothing: linear (fast), cubic (smooth), moving_average (noise reduction), savgol (best for noisy data)",
                            )
                            if smoothing_method in ["moving_average", "savgol"]:
                                smoothing_window = st.slider(
                                    "Smoothing Window",
                                    min_value=3,
                                    max_value=21,
                                    value=st.session_state.get("multi_file_multi_param_smoothing_window", 5),
                                    step=2,
                                    key="multi_file_multi_param_smoothing_window",
                                    help="Larger window = more smoothing (must be odd number)",
                                )
                                # Ensure odd (but don't modify session state - widget manages it)
                                if smoothing_window % 2 == 0:
                                    smoothing_window += 1  # Use odd value for smoothing
                    
                    # Use all dataframes (no filtering needed since we're using full range)
                    filtered_data = {}
                    for filename, df in file_data.items():
                        if x_axis in df.columns:
                            filtered = df.copy()
                            if not filtered.empty:
                                # Apply smoothing if enabled
                                if smoothing_enabled:
                                    all_y_cols = (left_y_axes or []) + (right_y_axes or [])
                                    filtered = smooth_data_for_plotting(
                                        filtered,
                                        x_axis,
                                        all_y_cols,
                                        method=smoothing_method,
                                        smoothing_window=smoothing_window,
                                    )
                                filtered_data[filename] = filtered

                    if not filtered_data:
                        st.warning("No data available to display.")
                        st.stop()

                    # Create figure with dual Y-axes
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Color palette for different files (orange and blue first to match reference style)
                    # Orange (#ff7f0e) and Blue (#1f77b4) are first for voltage comparisons
                    colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
                             '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94']
                    
                    file_idx = 0
                    line_styles = ['solid', 'dash', 'dot', 'dashdot']

                    # Add traces for each file
                    for filename, df_filtered in filtered_data.items():
                        file_color = colors[file_idx % len(colors)]
                        line_style = line_styles[(file_idx // len(colors)) % len(line_styles)]
                        
                        # Extract condition name from filename (e.g., "AXI 5345/16D V3 195KV @ 50V")
                        # Try to extract voltage or condition info
                        display_name = filename.replace('.csv', '').replace('.ulg', '')
                        # Look for voltage pattern like "@ 50V" or "@42V"
                        voltage_match = re.search(r'@\s*(\d+V)', display_name, re.IGNORECASE)
                        if voltage_match:
                            # Extract motor model and voltage
                            motor_match = re.search(r'(AXI[^@]+)', display_name, re.IGNORECASE)
                            if motor_match:
                                motor_part = motor_match.group(1).strip()
                                voltage_part = voltage_match.group(1)
                                display_name = f"{motor_part} @ {voltage_part}"
                            else:
                                display_name = display_name
                        else:
                            # Try to shorten if too long
                            if len(display_name) > 50:
                                display_name = display_name[:47] + "..."
                        
                        # Track if we've shown legend for this file
                        legend_shown = False
                        
                        # Add left Y-axis traces
                        if left_y_axes:
                            for y_col in left_y_axes:
                                if y_col in df_filtered.columns:
                                    # Remove NaN values for clean plotting
                                    plot_data = df_filtered[[x_axis, y_col]].dropna()
                                    if not plot_data.empty:
                                        # Show legend only for the first trace of this file
                                        show_legend = not legend_shown
                                        if show_legend:
                                            legend_shown = True
                                        fig.add_trace(
                                            go.Scatter(
                                                x=plot_data[x_axis],
                                                y=plot_data[y_col],
                                                mode="lines+markers",
                                                name=display_name,  # Use condition name, not parameter name
                                                line=dict(color=file_color, width=2, dash=line_style),
                                                marker=dict(size=4, color=file_color),
                                                legendgroup=filename,
                                                showlegend=show_legend,  # Only show once per file
                                            ),
                                            secondary_y=False,
                                        )

                        # Add right Y-axis traces
                        if right_y_axes:
                            for y_col in right_y_axes:
                                if y_col in df_filtered.columns:
                                    # Remove NaN values for clean plotting
                                    plot_data = df_filtered[[x_axis, y_col]].dropna()
                                    if not plot_data.empty:
                                        # Show legend only if we haven't shown it yet (for first trace of this file)
                                        show_legend = not legend_shown
                                        if show_legend:
                                            legend_shown = True
                                        fig.add_trace(
                                            go.Scatter(
                                                x=plot_data[x_axis],
                                                y=plot_data[y_col],
                                                mode="lines+markers",
                                                name=display_name,  # Use condition name, not parameter name
                                                line=dict(color=file_color, width=2, dash=line_style),
                                                marker=dict(size=4, color=file_color),
                                                legendgroup=filename,
                                                showlegend=show_legend,  # Only show once per file
                                            ),
                                            secondary_y=True,
                                        )
                        
                        file_idx += 1

                    # Set X-axis title
                    fig.update_xaxes(title_text=get_axis_title(x_axis))

                    # Set Y-axis titles and ranges with grid on left Y-axis
                    if left_y_axes and left_y_min is not None and left_y_max is not None:
                        left_display_names = [get_display_name(col) for col in left_y_axes]
                        left_title = " & ".join(left_display_names) if len(left_display_names) <= 2 else f"{left_display_names[0]} & {len(left_display_names)-1} more"
                        fig.update_yaxes(
                            title_text=left_title,
                            range=[left_y_min, left_y_max],
                            secondary_y=False,
                            showgrid=True,  # Show grid on left Y-axis
                            gridcolor='lightgray',
                            gridwidth=1
                        )
                    
                    if right_y_axes and right_y_min is not None and right_y_max is not None:
                        right_display_names = [get_display_name(col) for col in right_y_axes]
                        right_title = " & ".join(right_display_names) if len(right_display_names) <= 2 else f"{right_display_names[0]} & {len(right_display_names)-1} more"
                        fig.update_yaxes(
                            title_text=right_title,
                            range=[right_y_min, right_y_max],
                            secondary_y=True,
                            showgrid=False  # No grid on right Y-axis (like matplotlib twinx)
                        )
                    
                    # Also add grid to X-axis
                    fig.update_xaxes(
                        showgrid=True,
                        gridcolor='lightgray',
                        gridwidth=1
                    )

                    # Generate title
                    title_parts = []
                    if left_y_axes:
                        left_title_parts = [get_display_name(col) for col in left_y_axes]
                        title_parts.append(" & ".join(left_title_parts))
                    if right_y_axes:
                        right_title_parts = [get_display_name(col) for col in right_y_axes]
                        title_parts.append(" & ".join(right_title_parts))
                    title_text = " & ".join(title_parts) + f" Vs {get_axis_title(x_axis)}" if title_parts else f"Multi-File Analysis: {get_axis_title(x_axis)}"

                    # Update layout
                    fig.update_layout(
                        title=dict(
                            text=title_text,
                            x=0.5,
                            xanchor='center',
                            font=dict(size=16)
                        ),
                        legend=dict(
                            orientation="h",  # Horizontal legend at bottom
                            yanchor="bottom",
                            y=-0.15,  # Position below the plot
                            xanchor="center",
                            x=0.5,  # Center the legend
                            font=dict(size=11),
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="rgba(0,0,0,0.2)",
                            borderwidth=1
                        ),
                        template="plotly_white",
                        margin=dict(l=60, r=60, t=60, b=80),  # More bottom margin for horizontal legend
                        hovermode='x unified',
                    )

                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
