# Modified script to support both single file analysis and comparative assessment modes


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
from io import StringIO
from PIL import Image
import base64
# import open3d as o3d
import tempfile
import os
from pyulog import ULog


st.set_page_config(page_title="ROTRIX Dashboard", layout="wide")

# 🔹 Logo
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image(os.path.join(os.path.dirname(__file__), "Rotrix-Logo.png"))
# st.logo(logo_base64, *, size="medium", link=None, icon_image=None)
st.markdown(f"""
    <div style="display: flex; position: fixed; top:50px; left: 50px; z-index:50; justify-content: left; align-items: center; padding: 1px; background-color:white; border-radius:25px;">
        <a href="http://rotrixdemo.reude.tech/" target="_blank">
            <img src="data:image/png;base64,{logo_base64}" width="180" alt="Rotrix Logo">
        </a>
    </div>
""", unsafe_allow_html=True)

st.markdown("### 🚀 Comparative Assessment")

# Loaders
def load_csv(file):
    file.seek(0)
    return pd.read_csv(StringIO(file.read().decode("utf-8")))

# def load_pcd(file):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pcd") as tmp:
#         tmp.write(file.read())
#         pcd = o3d.io.read_point_cloud(tmp.name, format='xyz')
#         points = np.asarray(pcd.points)
#         df = pd.DataFrame(points, columns=["X", "Y", "Z"])
#         if len(np.asarray(pcd.colors)) > 0:
#             df["Temperature"] = np.mean(np.asarray(pcd.colors), axis=1)
#     return df

# List of (topic, assessment name) pairs
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
    "Thrust": ["thrust[0]", "thrust[1]", "thrust[2]"],
    "Torque": ["xyz[0]", "xyz[1]", "xyz[2]"],
    "Control": ["pwm[0]", "pwm[1]", "pwm[2]", "pwm[3]"],
    "Battery": ["voltage_v", "current_average_a", "discharged_mah"],
}

def load_ulog(file, key_suffix=""):
    ALLOWED_TOPICS = set(t for t, _ in TOPIC_ASSESSMENT_PAIRS)
    
    ulog = ULog(file)
    extracted_dfs = {msg.name: pd.DataFrame(msg.data) for msg in ulog.data_list}
    
    # Filter topics to only include allowed ones
    filtered_dfs = {topic: df for topic, df in extracted_dfs.items() if topic in ALLOWED_TOPICS}
    topic_names = ["None"] + list(filtered_dfs.keys())
    
    if not topic_names:
        st.warning("No extractable topics found in ULOG file.")
        return {}, []
    
    return filtered_dfs, topic_names

# def detect_trend(series):
#     if series.iloc[-1] > series.iloc[0]:
#         return "increasing"
#     elif series.iloc[-1] < series.iloc[0]:
#         return "decreasing"
#     return "flat"

def detect_abnormalities(series, threshold=3.0):
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold, z_scores

def convert_timestamps_to_seconds(df):
    """Convert timestamp columns to seconds from start"""
    if df is None or df.empty:
        return df
    # Find timestamp columns (case insensitive)
    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
    for col in timestamp_cols:
        # If numeric, check for ms/us and convert
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].max() > 1e12:
                df[col] = df[col] / 1e6  # microseconds to seconds
            elif df[col].max() > 1e9:
                df[col] = df[col] / 1e3  # milliseconds to seconds
        # If datetime, convert to seconds from start
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = (df[col] - df[col].min()).dt.total_seconds()
    return df

def ensure_seconds_column(df):
    """Ensure a 'timestamp_seconds' column exists and is always in seconds from start, using robust heuristics."""
    if df is None or df.empty:
        return df
    timestamp_col = next((col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower()), None)
    if timestamp_col is None:
        return df
    series = df[timestamp_col]
    # Heuristic: check the difference between first two values (if possible)
    if len(series) > 1:
        delta = series.iloc[1] - series.iloc[0]
        if delta > 1e6:  # microseconds
            df['timestamp_seconds'] = (series - series.min()) / 1e6
        elif delta > 1e3:  # milliseconds
            df['timestamp_seconds'] = (series - series.min()) / 1e3
        else:  # already in seconds or close
            df['timestamp_seconds'] = series - series.min()
    else:
        df['timestamp_seconds'] = series - series.min()
    return df

def resample_to_common_time(df1, df2, freq=1.0):
    """
    Resample two DataFrames to a common time base using interpolation.
    freq: float, in seconds (e.g., 1.0 for 1 second)
    Returns: (df1_resampled, df2_resampled, common_time_index)
    """
    # Ensure timestamp_seconds exists and is sorted
    for i, df in enumerate([df1, df2]):
        if 'timestamp_seconds' not in df.columns:
            raise ValueError("timestamp_seconds column missing")
        df = df.copy()
        df.sort_values('timestamp_seconds', inplace=True)
        if i == 0:
            df1 = df
        else:
            df2 = df
    # Create a common time index (from max of min to min of max)
    start = max(df1['timestamp_seconds'].min(), df2['timestamp_seconds'].min())
    end = min(df1['timestamp_seconds'].max(), df2['timestamp_seconds'].max())
    if start >= end:
        raise ValueError("No overlapping time range for resampling.")
    common_time = np.arange(start, end, freq)
    # Set index and interpolate
    df1_interp = df1.set_index('timestamp_seconds').interpolate(method='linear').reindex(common_time, method='nearest').reset_index().rename(columns={'index': 'timestamp_seconds'})
    df2_interp = df2.set_index('timestamp_seconds').interpolate(method='linear').reindex(common_time, method='nearest').reset_index().rename(columns={'index': 'timestamp_seconds'})
    return df1_interp, df2_interp, common_time

def format_seconds_to_mmss(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"

def get_tick_spacing(data_range):
    """Calculate appropriate tick spacing based on data range"""
    if data_range <= 10:  # Less than 10 seconds
        return 1  # Show every second
    elif data_range <= 60:  # Less than 1 minute
        return 10  # Show every 10 seconds
    elif data_range <= 300:  # Less than 5 minutes
        return 30  # Show every 30 seconds
    elif data_range <= 600:  # Less than 10 minutes
        return 60  # Show every minute
    else:
        return 120  # Show every 2 minutes

def get_axis_title(axis_name):
    """Get formatted axis title"""
    if axis_name == 'timestamp_seconds':
        return 'TIME(secs)'
    return axis_name

def get_timestamp_ticks(data):
    """Generate evenly spaced timestamp ticks"""
    if data is None or len(data) == 0:
        return [], []
    data_range = data.max() - data.min()
    spacing = get_tick_spacing(data_range)
    ticks = np.arange(data.min(), data.max() + spacing, spacing)
    return ticks, [format_seconds_to_mmss(t) for t in ticks]

# Load logic
def load_data(file, filetype, key_suffix):
    if filetype == ".csv":
        df_csv = load_csv(file)
        df_csv = convert_timestamps_to_seconds(df_csv)
        return df_csv, None
    elif filetype == ".ulg":
        df_ulog, topic_names = load_ulog(file, key_suffix)
        df_ulog = convert_timestamps_to_seconds(df_ulog)
        return df_ulog, topic_names
    return None, None

def add_remove_common_column(b_df, v_df):
    if b_df is None or v_df is None or b_df.empty or v_df.empty:
        st.warning("⚠ Both Benchmark and Target data must be loaded.")
        return b_df, v_df

    if "pending_column" in st.session_state:
        new_col = st.session_state["pending_column"]
        try:
            for df_key in ["b_df", "v_df"]:
                df = st.session_state[df_key]
                if new_col["name"] not in df.columns:
                    df.insert(1, new_col["name"], df.eval(new_col["formula"]))
                else:
                    df[new_col["name"]] = df.eval(new_col["formula"])
                st.session_state[df_key] = df
            st.success(f"✅ Added {new_col['name']} using {new_col['formula']} to both Benchmark and Target.")
        except Exception as e:
            st.error(f"❌ Failed to add column: {e}")
        del st.session_state["pending_column"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("###### 🧮 New Column")
        new_col_name = st.text_input("New Column Name", key="common_add")
        custom_formula = st.text_input("Formula (e.g., Voltage * Current)", key="common_formula")

        if st.button("Add Column"):
            if new_col_name and custom_formula:
                st.session_state["pending_column"] = {
                    "name": new_col_name,
                    "formula": custom_formula
                }
                st.rerun()

    with col2:
        st.markdown("###### 🗑 Remove Column")
        common_cols = list(set(b_df.columns) & set(v_df.columns))
        cols_to_drop = st.multiselect("Select column(s) to drop", common_cols, key="common_drop")

        if st.button("Remove Columns"):
            if cols_to_drop:
                st.session_state.b_df.drop(columns=cols_to_drop, inplace=True)
                st.session_state.v_df.drop(columns=cols_to_drop, inplace=True)
                st.success(f"🗑 Removed columns: {', '.join(cols_to_drop)} from both Benchmark and Target.")
                st.rerun()

    return st.session_state.b_df, st.session_state.v_df


def add_remove_column(target_df, df_name="DataFrame"):
     # CREATE COLUMN
    if target_df is None or target_df.empty:
        st.warning(f"⚠ {df_name} is empty or not loaded.")
        return target_df
    
#     col1, col2 = st.columns(2)
#     with col1:
    st.markdown("##### 🧮 New Column")
    new_col_name = st.text_input("New Column Name", key=f"{df_name}_add")
    custom_formula = st.text_input("Formula (e.g., Voltage * Current)", key=f"{df_name}_formula")

    if st.button(f"Add Column to {df_name}"):
        try:
            if new_col_name and custom_formula:
                target_df[new_col_name] = target_df.eval(custom_formula)
                st.success(f"✅ Added column {new_col_name} to {df_name} using: {custom_formula}")
        except Exception as e:
            st.error(f"❌ Error creating column: {e}")
#     with col2:
    # REMOVE COLUMN
    st.markdown("##### 🗑 Remove Column")
    columns_to_drop = st.multiselect("Select columns to drop", target_df.columns, key=f"{df_name}_drop")

    if st.button(f"Remove Column to {df_name}"):
        if columns_to_drop:
            target_df.drop(columns=columns_to_drop, inplace=True)
            st.success(f"🗑 Removed columns: {', '.join(columns_to_drop)} from {df_name}")
            
    st.markdown("##### ✏ Rename Column")
    rename_col = st.selectbox("Select column to rename", target_df.columns, key=f"{df_name}_rename_col")
    new_name = st.text_input("New column name", key=f"{df_name}_rename_input")

    if st.button(f"Rename Column in {df_name}", key=f"{df_name}_rename_button"):
        if rename_col and new_name:
            target_df.rename(columns={rename_col: new_name}, inplace=True)
            st.success(f"✏ Renamed column {rename_col} to {new_name} in {df_name}")

    return target_df

# st.markdown("#### 🔼 Upload Benchmark & Validation Files")
st.markdown("<h4 style='font-size:20px; color:#FFFF00;'>🔼 Upload Benchmark & Target Files</h4>", unsafe_allow_html=True)

# Simulate a topbar with two upload sections
top_col1, top_col2, top_col3, top_col4 = st.columns(4)

with top_col1:
    benchmark_files = st.file_uploader("📂 Upload Benchmark File", type=["csv", "pcd", "ulg"], accept_multiple_files=True)
    benchmark_names = [f.name for f in benchmark_files] if benchmark_files else []
    
with top_col3:
    b_df = None
    if benchmark_files:
        selected_bench = st.selectbox("Select Benchmark File", ["None"] + benchmark_names)
        if selected_bench != "None":
            b_file = benchmark_files[benchmark_names.index(selected_bench)]
            b_file_ext = os.path.splitext(b_file.name)[-1].lower()
            if b_file_ext == ".ulg":
                b_dfs, b_topics = load_ulog(b_file)
                if "common_topic" not in st.session_state:
                    st.session_state.common_topic = b_topics[0] if b_topics else "None"
                # Map topic names to assessment names for display
                assessment_names = ["None"] + [a for _, a in TOPIC_ASSESSMENT_PAIRS]
                # Map assessment name to topic
                assessment_to_topic = {a: t for t, a in TOPIC_ASSESSMENT_PAIRS}
                selected_assessment = st.selectbox("Select Topic", options=assessment_names, key="common_topic")
                selected_topic = assessment_to_topic[selected_assessment] if selected_assessment in assessment_to_topic else "None"
                if selected_topic != "None" and selected_topic in b_dfs:
                    st.session_state.b_df = b_dfs[selected_topic]
            else:
                df, _ = load_data(b_file, b_file_ext, key_suffix="bench")
                if df is not None:
                    st.session_state.b_df = df
        
with top_col2:
    validation_files = st.file_uploader("📂 Upload Target File", type=["csv", "pcd", "ulg"], accept_multiple_files=True)
    validation_names = [f.name for f in validation_files] if validation_files else []
    
with top_col4:
    v_df = None
    if validation_files:
        selected_val = st.selectbox("Select Target File", ["None"] + validation_names)
        if selected_val != "None":
            v_file = validation_files[validation_names.index(selected_val)]
            v_file_ext = os.path.splitext(v_file.name)[-1].lower()
            if v_file_ext == ".ulg":
                v_dfs, v_topics = load_ulog(v_file)
                
                # Only show target topic selection if no benchmark file is selected
                if not benchmark_files or selected_bench == "None":
                    # Show topic selection for target file independently
                    if "target_topic" not in st.session_state:
                        st.session_state.target_topic = v_topics[0] if v_topics else "None"
                    # Map topic names to assessment names for display
                    assessment_names = ["None"] + [a for _, a in TOPIC_ASSESSMENT_PAIRS]
                    # Map assessment name to topic
                    assessment_to_topic = {a: t for t, a in TOPIC_ASSESSMENT_PAIRS}
                    selected_assessment = st.selectbox("Select Target Topic", options=assessment_names, key="target_topic")
                    selected_topic = assessment_to_topic[selected_assessment] if selected_assessment in assessment_to_topic else "None"
                else:
                    # Use the common topic from benchmark selection
                    selected_assessment = st.session_state.get("common_topic", "None")
                    assessment_to_topic = {a: t for t, a in TOPIC_ASSESSMENT_PAIRS}
                    selected_topic = assessment_to_topic[selected_assessment] if selected_assessment in assessment_to_topic else "None"
                
                if selected_topic != "None" and selected_topic in v_dfs:
                    st.session_state.v_df = v_dfs[selected_topic]
                else:
                    st.session_state.v_df = None
            else:
                df, _ = load_data(v_file, v_file_ext, key_suffix="val")
                if df is not None:
                    st.session_state.v_df = df
        
if "b_df" not in st.session_state:
    st.session_state.b_df = None
if "v_df" not in st.session_state:
    st.session_state.v_df = None
    
b_df = st.session_state.get("b_df")
v_df = st.session_state.get("v_df")

# After loading data for b_df and v_df, call ensure_seconds_column
b_df = ensure_seconds_column(b_df)
v_df = ensure_seconds_column(v_df)

# Main layout
# col_main1, col_main2 = st.columns([0.25, 0.75])

# with col_main1:
#     st.markdown("<h4 style='font-size:18px; color:#0099ff;'>🔧 Data Analysis Settings</h4>", unsafe_allow_html=True)
#     selected_df = st.multiselect("Select DataFrame to Modify", ["Benchmark", "Target", "Both"], key='data_analysis')

#     if selected_df:  # Only process if something is selected
#         for param in selected_df:
#             if param == "Both":
#                 st.session_state.b_df, st.session_state.v_df = add_remove_common_column(st.session_state.b_df, st.session_state.v_df)
#             elif param == "Benchmark":
#                 st.session_state.b_df = add_remove_column(st.session_state.b_df, df_name="Benchmark")
#             elif param == "Target":
#                 st.session_state.v_df = add_remove_column(st.session_state.v_df, df_name="Target")

# with col_main2:
tab1, tab2 = st.tabs(["📊 Plot", "📋 Data"])
with tab2:
    st.subheader("📁 Imported Data Preview")
    # Data Analysis Settings moved here
    st.markdown("<h4 style='font-size:18px; color:#0099ff;'>🔧 Data Analysis Settings</h4>", unsafe_allow_html=True)
    selected_df = st.multiselect("Select DataFrame to Modify", ["Benchmark", "Target", "Both"], key='data_analysis')
    if selected_df:  # Only process if something is selected
        for param in selected_df:
            if param == "Both":
                st.session_state.b_df, st.session_state.v_df = add_remove_common_column(st.session_state.b_df, st.session_state.v_df)
            elif param == "Benchmark":
                st.session_state.b_df = add_remove_column(st.session_state.b_df, df_name="Benchmark")
            elif param == "Target":
                st.session_state.v_df = add_remove_column(st.session_state.v_df, df_name="Target")
            if "Both" in selected_df and st.session_state.b_df is not None and st.session_state.v_df is not None:
                st.markdown("##### ✏ Rename Column in Both")
                common_cols = list(set(st.session_state.b_df.columns) & set(st.session_state.v_df.columns))
                if common_cols:
                    rename_col = st.selectbox("Select column to rename", common_cols, key="both_rename_col")
                    new_name = st.text_input("New column name", key="both_rename_input")
                    if st.button("Rename Column in Both", key="both_rename_button"):
                        if rename_col and new_name:
                            st.session_state.b_df.rename(columns={rename_col: new_name}, inplace=True)
                            st.session_state.v_df.rename(columns={rename_col: new_name}, inplace=True)
                            st.success(f"✏ Renamed column {rename_col} to {new_name} in both Benchmark and Target")
    # Data preview as before
    b_df = st.session_state.get("b_df")
    v_df = st.session_state.get("v_df")
    if b_df is not None and v_df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🧪 Benchmark Data")
            st.dataframe(b_df)
        with col2:
            st.markdown("### 🔬 Target Data")
            st.dataframe(v_df)
    elif b_df is not None:
        st.markdown("### 🧪 Benchmark Data")
        st.dataframe(b_df)
    elif v_df is not None:
        st.markdown("### 🔬 Target Data")
        st.dataframe(v_df)
    else:
        st.info("No data uploaded yet.")
    
with tab1:
    st.subheader(" 🔍 Data Analysis")
    
    # Get the active dataframe(s)
    b_df = st.session_state.get("b_df")
    v_df = st.session_state.get("v_df")
    
    # Check if at least one file is loaded
    if b_df is not None or v_df is not None:
        # If both files are loaded, do comparative analysis
        if b_df is not None and v_df is not None:
            st.subheader("Comparative Analysis")
            b_df.insert(0, "Index", range(1, len(b_df) + 1))
            v_df.insert(0, "Index", range(1, len(v_df) + 1))
            
            common_cols = list(set(b_df.columns) & set(v_df.columns))
            if common_cols:
                col1, col2 = st.columns([0.20, 0.80])
                with col1:
                    st.markdown("#### 📈 Parameters")
                    # Define allowed columns for axes
                    if 'selected_assessment' in locals() and isinstance(selected_assessment, str) and selected_assessment != "None":
                        allowed_y_axis = ASSESSMENT_Y_AXIS_MAP.get(selected_assessment, [])
                        # Filter columns that actually exist in the dataframe
                        allowed_y_axis = [col for col in allowed_y_axis if col in b_df.columns]
                        if not allowed_y_axis:
                            allowed_y_axis = list(b_df.columns)
                        # Filter out non-numeric columns for better visualization
                        allowed_y_axis = [col for col in allowed_y_axis if pd.api.types.is_numeric_dtype(b_df[col])]
                        ALLOWED_X_AXIS = ["Index", "timestamp_seconds"] + [col for col in allowed_y_axis if col not in ["Index", "timestamp_seconds"]]
                    else:
                        # For non-topic files, only show numeric columns
                        allowed_y_axis = [col for col in b_df.columns if pd.api.types.is_numeric_dtype(b_df[col])]
                        ALLOWED_X_AXIS = ["Index", "timestamp_seconds"] + [col for col in allowed_y_axis if col not in ["Index", "timestamp_seconds"]]
                    x_axis_options = ALLOWED_X_AXIS
                    y_axis_options = allowed_y_axis
                    # Set default x_axis to 'Index', y_axis to first numeric column
                    default_x = 'Index' if 'Index' in x_axis_options else x_axis_options[0]
                    # If CSV and cD2detailpeak exists, use as default y if x is Index
                    if b_df is not None and v_df is not None and hasattr(b_df, 'columns') and hasattr(v_df, 'columns'):
                        if 'cD2detailpeak' in b_df.columns and 'cD2detailpeak' in v_df.columns:
                            default_y = 'cD2detailpeak' if default_x == 'Index' else (y_axis_options[0] if y_axis_options else None)
                        else:
                            default_y = y_axis_options[0] if y_axis_options else None
                    else:
                        default_y = y_axis_options[0] if y_axis_options else None
                    x_axis = st.selectbox("X-Axis", x_axis_options, key="x_axis_select", index=x_axis_options.index(default_x))
                    # If user selects Index for both, force y to cD2detailpeak if available
                    if x_axis == 'Index' and 'cD2detailpeak' in y_axis_options:
                        y_axis = st.selectbox("Y-Axis", y_axis_options, key="y_axis_select", index=y_axis_options.index('cD2detailpeak'))
                    else:
                        y_axis = st.selectbox("Y-Axis", y_axis_options, key="y_axis_select", index=y_axis_options.index(default_y) if default_y in y_axis_options else 0)
                    if not x_axis or not y_axis:
                        st.info("📌 Please select valid X and Y axes to compare.")
                    else:
                        z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, key="z-slider-comparative")
                        x_min = st.number_input("X min", value=float(b_df[x_axis].min()), key="x_min_param")
                        x_max = st.number_input("X max", value=float(b_df[x_axis].max()), key="x_max_param")
                        y_min = st.number_input("Y min", value=float(b_df[y_axis].min()), key="y_min_param")
                        y_max = st.number_input("Y max", value=float(b_df[y_axis].max()), key="y_max_param")
                        plot_mode = st.radio("Plot Mode", ["Superimposed", "Separate"], horizontal=True, key="comparative_plot_mode")
    
                        # Filter data
                        b_filtered = b_df[(b_df[x_axis] >= x_min) & (b_df[x_axis] <= x_max) &
                                          (b_df[y_axis] >= y_min) & (b_df[y_axis] <= y_max)]
                        v_filtered = v_df[(v_df[x_axis] >= x_min) & (v_df[x_axis] <= x_max) &
                                          (v_df[y_axis] >= y_min) & (v_df[y_axis] <= y_max)]
    
                        if x_axis == 'timestamp_seconds':
                            try:
                                b_filtered, v_filtered, common_time = resample_to_common_time(b_filtered, v_filtered, freq=1.0)
                                st.info(f"Resampled to 1 second intervals. Overlapping time: {common_time[0]:.2f} to {common_time[-1]:.2f} s. Points: {len(common_time)}")
                                if y_axis not in b_filtered.columns or y_axis not in v_filtered.columns:
                                    st.warning("Selected Y-axis not found in both files after resampling.")
                                elif b_filtered[y_axis].isnull().all() or v_filtered[y_axis].isnull().all():
                                    st.warning("No valid data in one or both files for the selected Y-axis after resampling.")
                            except Exception as e:
                                st.warning(f"Resampling failed: {e}")
    
                        if y_axis == "discharged_mah":
                            if "discharged_mah" in b_filtered.columns:
                                b_filtered["discharged_mah"] = b_filtered["discharged_mah"] - b_filtered["discharged_mah"].min()
                            if "discharged_mah" in v_filtered.columns:
                                v_filtered["discharged_mah"] = v_filtered["discharged_mah"] - v_filtered["discharged_mah"].min()
    
                        merged = pd.merge(b_filtered, v_filtered, on=x_axis, suffixes=('_benchmark', '_validation'))
                        
                        if merged.empty:
                            st.warning("No overlapping data found for the selected X-axis. Try a different X-axis or check your files.")
                        else:
                            val_col = f"{y_axis}_validation"
                            bench_col = f"{y_axis}_benchmark"
                            # Calculate difference and z-score for abnormality detection
                            merged["Difference"] = merged[val_col] - merged[bench_col]
                            merged["Z_Score"] = (merged["Difference"] - merged["Difference"].mean()) / merged["Difference"].std()
                            abnormal_mask = merged["Z_Score"].abs() > z_threshold
                            merged["Abnormal"] = abnormal_mask
                            abnormal_points = merged[merged["Abnormal"]]
                            # Calculate statistics for the selected y_axis in the validation data
                            stats = merged[val_col].describe()
                with col2:
                    if x_axis != "None" and y_axis != "None":
                        st.markdown("<div style='min-height: 10px'>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("### 🎯 Metrics")
                        # Comparative analysis metrics: RMSE, Similarity Index, Abnormal Points
                        rmse = np.sqrt(np.mean((merged[val_col] - merged[bench_col]) ** 2))
                        bench_range = merged[bench_col].max() - merged[bench_col].min()
                        if bench_range == 0:
                            similarity = 1.0 if rmse == 0 else 0.0
                        else:
                            similarity = 1 - (rmse / bench_range)
                        similarity_index = similarity * 100
                        abnormal_count = int(abnormal_mask.sum())
                        # Use fixed-width columns for metrics
                        metric_col1, metric_col2, metric_col3 = st.columns([1, 1, 1])
                        with metric_col1:
                            fig1 = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=rmse,
                                title={'text': "RMSE"},
                                number={'valueformat': ',.2f'},
                                domain={'x': [0.15, 0.85], 'y': [0, 1]},
                                gauge={
                                    'axis': {'range': [0, max(rmse * 2, 1)], 'tickformat': ',.2f'},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, rmse], 'color': "lightgray"},
                                        {'range': [rmse, max(rmse * 2, 1)], 'color': "gray"}
                                    ]
                                }
                            ))
                            fig1.update_layout(width=200, height=140, margin=dict(t=60, b=10))
                            st.plotly_chart(fig1)
                        with metric_col2:
                            fig2 = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=similarity_index,
                                title={'text': "Similarity Index (%)"},
                                number={'valueformat': '.2f', 'suffix': '%'},
                                domain={'x': [0.15, 0.85], 'y': [0, 1]},
                                gauge={
                                    'axis': {'range': [0, 100], 'tickformat': '.0f'},
                                    'bar': {'color': "orange"},
                                    'steps': [
                                        {'range': [0, 33], 'color': "#d4f0ff"},
                                        {'range': [33, 66], 'color': "#ffeaa7"},
                                        {'range': [66, 100], 'color': "#c8e6c9"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 50
                                    }
                                }
                            ))
                            fig2.update_layout(width=200, height=140, margin=dict(t=60, b=10))
                            st.plotly_chart(fig2)
                        with metric_col3:
                            fig3 = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=abnormal_count,
                                title={'text': "Abnormal Points"},
                                number={'valueformat': 'd'},
                                domain={'x': [0.15, 0.85], 'y': [0, 1]},
                                gauge={
                                    'axis': {'range': [0, max(10, abnormal_count * 2)]},
                                    'bar': {'color': "crimson"},
                                    'steps': [
                                        {'range': [0, 10], 'color': "#c8e6c9"},
                                        {'range': [10, 25], 'color': "#ffcc80"},
                                        {'range': [25, 100], 'color': "#ef5350"}
                                    ]
                                }
                            ))
                            fig3.update_layout(width=200, height=140, margin=dict(t=60, b=10))
                            st.plotly_chart(fig3)
                        st.markdown("### 🧮 Plot Visualization")
                        if plot_mode == "Superimposed":
                            fig = go.Figure()
                            # Add main line plot
                            fig.add_trace(go.Scatter(
                                x=merged[x_axis],
                                y=merged[bench_col],
                                mode='lines',
                                name='Benchmark'
                            ))
                            # Add validation data (Target) in green
                            fig.add_trace(go.Scatter(
                                x=merged[x_axis],
                                y=merged[val_col],
                                mode='lines',
                                name='Target',
                                line=dict(color='green')
                            ))
                            # Add abnormal points to target plot
                            if not abnormal_points.empty:
                                fig.add_trace(
                                    go.Scatter(
                                        x=abnormal_points[x_axis], 
                                        y=abnormal_points[val_col], 
                                        mode='markers', 
                                        marker=dict(color='red', size=8), 
                                        name='Abnormal Points'
                                    ), 
                                )
                            
                            # Get timestamp ticks if needed
                            if x_axis == 'timestamp_seconds':
                                tick_vals, tick_texts = get_timestamp_ticks(merged[x_axis])
                            else:
                                tick_vals, tick_texts = None, None
                                
                            fig.update_layout(
                                height=900,
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.05,
                                    xanchor="center",
                                    x=0.5
                                ),
                                margin=dict(t=100),
                                xaxis=dict(
                                    showticklabels=True,
                                    title=get_axis_title(x_axis),
                                    tickvals=tick_vals,
                                    ticktext=tick_texts,
                                    type='linear'
                                ),
                                yaxis=dict(
                                    showticklabels=True,
                                    title=y_axis
                                )
                            )
                            # Robustly set x-axis ticks/labels for timestamp_seconds
                            if x_axis == 'timestamp_seconds' and tick_vals is not None and tick_texts is not None:
                                fig.update_xaxes(
                                    tickvals=tick_vals,
                                    ticktext=tick_texts,
                                    title_text=get_axis_title(x_axis),
                                    type='linear'
                                )
                            st.plotly_chart(fig, use_container_width=True)
                        else:  # Separate
                            from plotly.subplots import make_subplots
                            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Benchmark", "Target"), vertical_spacing=0.2)
                            # Benchmark plot
                            fig.add_trace(go.Scatter(
                                x=merged[x_axis],
                                y=merged[bench_col],
                                mode='lines',
                                name='Benchmark',
                                line=dict(color='blue')
                            ), row=1, col=1)
                            # Target plot
                            fig.add_trace(go.Scatter(
                                x=merged[x_axis],
                                y=merged[val_col],
                                mode='lines',
                                name='Target',
                                line=dict(color='green')
                            ), row=2, col=1)
                            # Abnormal points on target
                            if not abnormal_points.empty:
                                fig.add_trace(
                                    go.Scatter(
                                        x=abnormal_points[x_axis],
                                        y=abnormal_points[val_col],
                                        mode='markers',
                                        marker=dict(color='red', size=8),
                                        name='Abnormal Points'
                                    ), row=2, col=1
                                )
                                
                            # Get timestamp ticks if needed
                            if x_axis == 'timestamp_seconds':
                                tick_vals, tick_texts = get_timestamp_ticks(merged[x_axis])
                            else:
                                tick_vals, tick_texts = None, None
                                
                            fig.update_layout(
                                height=900,
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.05,
                                    xanchor="center",
                                    x=0.5
                                ),
                                margin=dict(t=100),
                                xaxis=dict(
                                    showticklabels=True,
                                    title=get_axis_title(x_axis),
                                    tickvals=tick_vals,
                                    ticktext=tick_texts,
                                    type='linear'
                                ),
                                yaxis=dict(
                                    showticklabels=True,
                                    title=y_axis
                                ),
                                xaxis2=dict(
                                    showticklabels=True,
                                    title=get_axis_title(x_axis),
                                    tickvals=tick_vals,
                                    ticktext=tick_texts,
                                    type='linear'
                                ),
                                yaxis2=dict(
                                    showticklabels=True,
                                    title=y_axis
                                )
                            )
                            # Robustly set x-axis ticks/labels for timestamp_seconds in both subplots
                            if x_axis == 'timestamp_seconds' and tick_vals is not None and tick_texts is not None:
                                fig.update_xaxes(
                                    tickvals=tick_vals,
                                    ticktext=tick_texts,
                                    title_text=get_axis_title(x_axis),
                                    type='linear',
                                    row=1, col=1
                                )
                                fig.update_xaxes(
                                    tickvals=tick_vals,
                                    ticktext=tick_texts,
                                    title_text=get_axis_title(x_axis),
                                    type='linear',
                                    row=2, col=1
                                )
                            st.plotly_chart(fig, use_container_width=True)

        # Single file analysis
        else:
            df = b_df if b_df is not None else v_df
            if df is not None:  # Add explicit None check
                st.subheader("Single File Analysis")
                df.insert(0, "Index", range(1, len(df) + 1))
                
                col1, col2 = st.columns([0.20, 0.80])
                with col1:
                    st.markdown("#### 📈 Parameters")
                    # Define allowed columns for axes
                    if 'selected_assessment' in locals() and isinstance(selected_assessment, str) and selected_assessment != "None":
                        allowed_y_axis = ASSESSMENT_Y_AXIS_MAP.get(selected_assessment, [])
                        # Filter columns that actually exist in the dataframe
                        allowed_y_axis = [col for col in allowed_y_axis if col in df.columns]
                        if not allowed_y_axis:
                            allowed_y_axis = list(df.columns)
                        # Filter out non-numeric columns for better visualization
                        allowed_y_axis = [col for col in allowed_y_axis if pd.api.types.is_numeric_dtype(df[col])]
                        ALLOWED_X_AXIS = ["Index", "timestamp_seconds"] + [col for col in allowed_y_axis if col not in ["Index", "timestamp_seconds"]]
                    else:
                        # For non-topic files, only show numeric columns
                        allowed_y_axis = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                        ALLOWED_X_AXIS = ["Index", "timestamp_seconds"] + [col for col in allowed_y_axis if col not in ["Index", "timestamp_seconds"]]
                    x_axis_options = ALLOWED_X_AXIS
                    y_axis_options = allowed_y_axis
                    # Set default x_axis to 'Index', y_axis to first numeric column
                    default_x = 'Index' if 'Index' in x_axis_options else x_axis_options[0]
                    # If CSV and cD2detailpeak exists, use as default y if x is Index
                    if df is not None and hasattr(df, 'columns'):
                        if 'cD2detailpeak' in df.columns:
                            default_y = 'cD2detailpeak' if default_x == 'Index' else (y_axis_options[0] if y_axis_options else None)
                        else:
                            default_y = y_axis_options[0] if y_axis_options else None
                    else:
                        default_y = y_axis_options[0] if y_axis_options else None
                    x_axis = st.selectbox("X-Axis", x_axis_options, key="single_x_axis", index=x_axis_options.index(default_x))
                    # If user selects Index for both, force y to cD2detailpeak if available
                    if x_axis == 'Index' and 'cD2detailpeak' in y_axis_options:
                        y_axis = st.selectbox("Y-Axis", y_axis_options, key="single_y_axis", index=y_axis_options.index('cD2detailpeak'))
                    else:
                        y_axis = st.selectbox("Y-Axis", y_axis_options, key="single_y_axis", index=y_axis_options.index(default_y) if default_y in y_axis_options else 0)
                    if not x_axis or not y_axis:
                        st.info("📌 Please select valid X and Y axes to compare.")
                    else:
                        z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, key="z-slider-single")
                        x_min = st.number_input("X min", value=float(df[x_axis].min()))
                        x_max = st.number_input("X max", value=float(df[x_axis].max()))
                        y_min = st.number_input("Y min", value=float(df[y_axis].min()))
                        y_max = st.number_input("Y max", value=float(df[y_axis].max()))
                        
                        # Filter data
                        filtered_df = df[(df[x_axis] >= x_min) & (df[x_axis] <= x_max) &
                                       (df[y_axis] >= y_min) & (df[y_axis] <= y_max)]
                        
                        # Calculate statistics and detect abnormalities
                        stats = filtered_df[y_axis].describe()
                        abnormal_mask, z_scores = detect_abnormalities(filtered_df[y_axis], z_threshold)
                        filtered_df["Z_Score"] = z_scores
                        filtered_df["Abnormal"] = abnormal_mask
                        abnormal_points = filtered_df[filtered_df["Abnormal"]]
                        
                with col2:
                    if x_axis != "None" and y_axis != "None":
                        st.markdown("<div style='min-height: 10px'>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("### 🎯 Metrics")
                        # Use fixed-width columns for metrics
                        metric_col1, metric_col2, metric_col3 = st.columns([1, 1, 1])
                        with metric_col1:
                            fig1 = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=float(stats['mean']),
                                title={'text': "Mean Value"},
                                domain={'x': [0.15, 0.85], 'y': [0, 1]},
                                gauge={
                                    'axis': {'range': [float(stats['min']), float(stats['max'])],
                                            'tickformat': '.2f'},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [float(stats['min']), float(stats['25%'])], 'color': "lightgray"},
                                        {'range': [float(stats['25%']), float(stats['75%'])], 'color': "gray"},
                                        {'range': [float(stats['75%']), float(stats['max'])], 'color': "darkgray"}
                                    ]
                                }
                            ))
                            fig1.update_layout(width=200, height=120, margin=dict(t=50, b=10))
                            st.plotly_chart(fig1)
                        with metric_col2:
                            fig2 = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=float(stats['std']),
                                title={'text': "Standard Deviation"},
                                domain={'x': [0.15, 0.85], 'y': [0, 1]},
                                gauge={
                                    'axis': {'range': [0, float(stats['std'] * 2)],
                                            'tickformat': '.2f'},
                                    'bar': {'color': "orange"},
                                    'steps': [
                                        {'range': [0, float(stats['std'])], 'color': "#d4f0ff"},
                                        {'range': [float(stats['std']), float(stats['std'] * 2)], 'color': "#ffeaa7"}
                                    ]
                                }
                            ))
                            fig2.update_layout(width=200, height=120, margin=dict(t=50, b=10))
                            st.plotly_chart(fig2)
                        with metric_col3:
                            abnormal_count = int(abnormal_mask.sum())
                            fig3 = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=abnormal_count,
                                title={'text': "Abnormal Points"},
                                domain={'x': [0.15, 0.85], 'y': [0, 1]},
                                gauge={
                                    'axis': {'range': [0, max(10, abnormal_count * 2)]},
                                    'bar': {'color': "crimson"},
                                    'steps': [
                                        {'range': [0, 10], 'color': "#c8e6c9"},
                                        {'range': [10, 25], 'color': "#ffcc80"},
                                        {'range': [25, 100], 'color': "#ef5350"}
                                    ]
                                }
                            ))
                            fig3.update_layout(width=200, height=120, margin=dict(t=50, b=10))
                            st.plotly_chart(fig3)
                        st.markdown("### 🧮 Plot Visualization")
                        fig = go.Figure()
                        
                        # Add main line plot
                        fig.add_trace(go.Scatter(
                            x=filtered_df[x_axis],
                            y=filtered_df[y_axis],
                            mode='lines',
                            name='Data'
                        ))
                        
                        # Add abnormal points
                        fig.add_trace(go.Scatter(
                            x=abnormal_points[x_axis],
                            y=abnormal_points[y_axis],
                            mode='markers',
                            marker=dict(color='red', size=6),
                            name='Abnormal Points'
                        ))
                        
                        # Add mean line
                        mean_value = filtered_df[y_axis].mean()
                        fig.add_hline(y=mean_value, line_dash="dash", line_color="green",
                                    annotation_text=f"Mean: {mean_value:.2f}")
                        
                        # Get timestamp ticks if needed
                        if x_axis == 'timestamp_seconds':
                            tick_vals, tick_texts = get_timestamp_ticks(filtered_df[x_axis])
                        else:
                            tick_vals, tick_texts = None, None
                            
                        fig.update_layout(
                            height=900,
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.05,
                                xanchor="center",
                                x=0.5
                            ),
                            margin=dict(t=100),  # Add top margin for subplot titles
                            xaxis=dict(
                                showticklabels=True,
                                title=get_axis_title(x_axis),
                                tickvals=tick_vals,
                                ticktext=tick_texts,
                                type='linear'
                            ),
                            yaxis=dict(
                                showticklabels=True,
                                title=y_axis
                            )
                        )
                        # Robustly set x-axis ticks/labels for timestamp_seconds
                        if x_axis == 'timestamp_seconds' and tick_vals is not None and tick_texts is not None:
                            fig.update_xaxes(
                                tickvals=tick_vals,
                                ticktext=tick_texts,
                                title_text=get_axis_title(x_axis),
                                type='linear'
                            )
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload at least one file to begin analysis.")
