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

logo_base64 = get_base64_image("Rotrix-Logo.png")
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

def detect_trend(series):
    if series.iloc[-1] > series.iloc[0]:
        return "increasing"
    elif series.iloc[-1] < series.iloc[0]:
        return "decreasing"
    return "flat"

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
                st.success(f"✅ Added column {new_col_name} to {selected_df} using: {custom_formula}")
        except Exception as e:
            st.error(f"❌ Error creating column: {e}")
#     with col2:
    # REMOVE COLUMN
    st.markdown("##### 🗑 Remove Column")
    columns_to_drop = st.multiselect("Select columns to drop", target_df.columns, key=f"{df_name}_drop")

    if st.button(f"Remove Column to {df_name}"):
        if columns_to_drop:
            target_df.drop(columns=columns_to_drop, inplace=True)
            st.success(f"🗑 Removed columns: {', '.join(columns_to_drop)} from {selected_df}")
            
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
                # No topic dropdown for target; use the selected topic from the single dropdown
                topic = None
                if "common_topic" in st.session_state:
                    # Get the selected assessment name from the single dropdown
                    selected_assessment = st.session_state.common_topic
                    # Map assessment name back to topic name
                    assessment_to_topic = {a: t for t, a in TOPIC_ASSESSMENT_PAIRS}
                    topic = assessment_to_topic[selected_assessment] if selected_assessment in assessment_to_topic else "None"
                else:
                    topic = v_topics[0] if v_topics else "None"
                if topic != "None" and topic in v_dfs:
                    st.session_state.v_df = v_dfs[topic]
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
col_main1, col_main2 = st.columns([0.25, 0.75])

with col_main1:
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

with col_main2:
    tab1, tab2 = st.tabs(["📊 Plot", "📋 Data"])
    with tab2:
        st.subheader("📁 Imported Data Preview")
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
                    col1, col2, col3 = st.columns([0.20, 0.60, 0.20])
                    with col1:
                        st.markdown("#### 📈 Parameters")
                        # Define allowed columns for axes
                        if 'selected_assessment' in locals() and isinstance(selected_assessment, str) and selected_assessment != "None":
                            allowed_y_axis = ASSESSMENT_Y_AXIS_MAP.get(selected_assessment, [])
                            allowed_y_axis = [col for col in allowed_y_axis if col in b_df.columns]
                            if not allowed_y_axis:
                                allowed_y_axis = list(b_df.columns)
                            ALLOWED_X_AXIS = ["Index", "timestamp", "timestamp_seconds"] + [col for col in allowed_y_axis if col not in ["Index", "timestamp", "timestamp_seconds"]]
                        else:
                            allowed_y_axis = list(common_cols)  # For CSV or no assessment, use all columns
                            ALLOWED_X_AXIS = ["Index", "timestamp", "timestamp_seconds"] + [col for col in allowed_y_axis if col not in ["Index", "timestamp", "timestamp_seconds"]]
                        y_axis_options = allowed_y_axis

                        x_axis = st.selectbox("X-Axis", ["None"] + ALLOWED_X_AXIS, key="x_axis_select")
                        y_axis = st.selectbox("Y-Axis", ["None"] + y_axis_options, key="y_axis_select")
                        if x_axis == "None" and y_axis == "None":
                            st.info("📌 Please select valid X and Y axes to compare.")
                        elif x_axis == "None":
                            st.info("📌 Please select a valid X axis.")
                        elif y_axis == "None":
                            st.info("📌 Please select a valid Y axis.")
                        else:
                            z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, key="z-slider-comparative")
                            x_min = st.number_input("X min", value=float(b_df[x_axis].min()), key="x_min_param")
                            x_max = st.number_input("X max", value=float(b_df[x_axis].max()), key="x_max_param")
                            y_min = st.number_input("Y min", value=float(b_df[y_axis].min()), key="y_min_param")
                            y_max = st.number_input("Y max", value=float(b_df[y_axis].max()), key="y_max_param")
        
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
                                abnormal_mask, z_scores = detect_abnormalities(merged[val_col], z_threshold)
                                merged["Z_Score"] = z_scores
                                merged["Abnormal"] = abnormal_mask
                                abnormal_points = merged[merged["Abnormal"]]
                                
                    with col2:
                        if x_axis != "None" and y_axis != "None":
                            st.markdown("<div style='min-height: 10px'>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                            st.markdown("### 🧮 Plot Visualization")
                            
                            from plotly.subplots import make_subplots
                            fig = make_subplots(
                                rows=2, 
                                cols=1, 
                                shared_xaxes=True, 
                                subplot_titles=("Benchmark", "Target"),
                                vertical_spacing=0.15  # Increased vertical gap between subplots
                            )
                            
                            # Add benchmark data
                            fig.add_trace(
                                go.Scatter(
                                    x=merged[x_axis], 
                                    y=merged[bench_col], 
                                    mode='lines', 
                                    name='Benchmark', 
                                    line=dict(color='blue')
                                ), 
                                row=1, 
                                col=1
                            )
                            
                            # Add validation data
                            fig.add_trace(
                                go.Scatter(
                                    x=merged[x_axis], 
                                    y=merged[val_col], 
                                    mode='lines', 
                                    name='Target', 
                                    line=dict(color='green')
                                ), 
                                row=2, 
                                col=1
                            )
                            
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
                                    row=2, 
                                    col=1
                                )
                            
                            # Update layout with better spacing and formatting
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
                                xaxis2_title=x_axis,
                                yaxis1_title=y_axis,
                                yaxis2_title=y_axis,
                                margin=dict(t=100),  # Add top margin for subplot titles
                                xaxis=dict(
                                    showticklabels=True,
                                    title=x_axis
                                ),
                                xaxis2=dict(
                                    showticklabels=True,
                                    title=x_axis
                                ),
                                yaxis=dict(
                                    showticklabels=True,
                                    title=y_axis
                                ),
                                yaxis2=dict(
                                    showticklabels=True,
                                    title=y_axis
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)

                    with col3:
                        if x_axis != "None" and y_axis != "None":
                            st.markdown("<div style='min-height: 10px'>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                            st.markdown("### 🎯 Metrics")
                            rmse = np.sqrt(mean_squared_error(merged[bench_col], merged[val_col]))
                            bench_range = merged[bench_col].max() - merged[bench_col].min()
                            if bench_range == 0:
                                similarity = 1.0 if rmse == 0 else 0.0
                            else:
                                similarity = 1 - (rmse / bench_range)
                            similarity_index = similarity*100

                            fig = make_subplots(
                                rows=3, cols=1,
                                specs=[[{"type": "indicator"}], [{"type": "indicator"}], [{"type": "indicator"}]],
                                vertical_spacing=0.05
                            )

                            fig.add_trace(go.Indicator(
                                mode="gauge+number+delta",
                                value=similarity_index,
                                title={'text': "Similarity Index (%)"},
                                delta={'reference': 100, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "red"},
                                        {'range': [50, 75], 'color': "orange"},
                                        {'range': [75, 100], 'color': "green"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "black", 'width': 4},
                                        'thickness': 0.75,
                                        'value': similarity_index
                                    }
                                }
                            ), row=1, col=1)

                            rmse_value = float(rmse)
                            fig.add_trace(go.Indicator(
                                mode="gauge+number",
                                value=rmse_value,
                                title={'text': "RMSE Error"},
                                gauge={
                                    'axis': {'range': [0, max(100, rmse_value * 2)]},
                                    'bar': {'color': "orange"},
                                    'steps': [
                                        {'range': [0, 10], 'color': "#d4f0ff"},
                                        {'range': [10, 30], 'color': "#ffeaa7"},
                                        {'range': [30, 100], 'color': "#ff7675"}
                                    ]
                                }
                            ), row=2, col=1)

                            abnormal_count = int(abnormal_mask.sum())
                            fig.add_trace(go.Indicator(
                                mode="gauge+number",
                                value=abnormal_count,
                                title={'text': "Abnormal Points"},
                                gauge={
                                    'axis': {'range': [0, max(10, abnormal_count * 2)]},
                                    'bar': {'color': "crimson"},
                                    'steps': [
                                        {'range': [0, 10], 'color': "#c8e6c9"},
                                        {'range': [10, 25], 'color': "#ffcc80"},
                                        {'range': [25, 100], 'color': "#ef5350"}
                                    ]
                                }
                            ), row=3, col=1)

                            fig.update_layout(height=700, margin=dict(t=10, b=10))
                            st.plotly_chart(fig, use_container_width=True)
            
                else:
                    st.warning("No common columns to compare between Benchmark and Validation.")
            
            # Single file analysis
            else:
                df = b_df if b_df is not None else v_df
                if df is not None:  # Add explicit None check
                    st.subheader("Single File Analysis")
                    df.insert(0, "Index", range(1, len(df) + 1))
                    
                    col1, col2, col3 = st.columns([0.20, 0.60, 0.20])
                    with col1:
                        st.markdown("#### 📈 Parameters")
                        # Define allowed columns for axes
                        if 'selected_assessment' in locals() and isinstance(selected_assessment, str) and selected_assessment != "None":
                            allowed_y_axis = ASSESSMENT_Y_AXIS_MAP.get(selected_assessment, [])
                            allowed_y_axis = [col for col in allowed_y_axis if col in df.columns]
                            if not allowed_y_axis:
                                allowed_y_axis = list(df.columns)
                            ALLOWED_X_AXIS = ["Index", "timestamp", "timestamp_seconds"] + [col for col in allowed_y_axis if col not in ["Index", "timestamp", "timestamp_seconds"]]
                        else:
                            allowed_y_axis = list(df.columns)
                            ALLOWED_X_AXIS = ["Index", "timestamp", "timestamp_seconds"] + [col for col in allowed_y_axis if col not in ["Index", "timestamp", "timestamp_seconds"]]
                        y_axis_options = allowed_y_axis

                        x_axis = st.selectbox("X-Axis", ["None"] + ALLOWED_X_AXIS, key="single_x_axis")
                        y_axis = st.selectbox("Y-Axis", ["None"] + y_axis_options, key="single_y_axis")
                        if x_axis == "None" and y_axis == "None":
                            st.info("📌 Please select valid X and Y axes to compare.")
                        elif x_axis == "None":
                            st.info("📌 Please select a valid X axis.")
                        elif y_axis == "None":
                            st.info("📌 Please select a valid Y axis.")
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
                            
                            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),     
                                title=f"{y_axis} vs {x_axis}",
                                xaxis_title=x_axis,
                                yaxis_title=y_axis,
                                height=700
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                            
                    with col3:
                        if x_axis != "None" and y_axis != "None":
                            st.markdown("<div style='min-height: 10px'>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                            st.markdown("### 🎯 Metrics")
                            
                            fig = make_subplots(
                                rows=3, cols=1,
                                specs=[[{"type": "indicator"}], [{"type": "indicator"}], [{"type": "indicator"}]],
                                vertical_spacing=0.05
                            )
                            
                            # Mean value gauge
                            fig.add_trace(go.Indicator(
                                mode="gauge+number",
                                value=float(stats['mean']),
                                title={'text': "Mean Value"},
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
                            ), row=1, col=1)
                            
                            # Standard Deviation gauge
                            fig.add_trace(go.Indicator(
                                mode="gauge+number",
                                value=float(stats['std']),
                                title={'text': "Standard Deviation"},
                                gauge={
                                    'axis': {'range': [0, float(stats['std'] * 2)],
                                            'tickformat': '.2f'},
                                    'bar': {'color': "orange"},
                                    'steps': [
                                        {'range': [0, float(stats['std'])], 'color': "#d4f0ff"},
                                        {'range': [float(stats['std']), float(stats['std'] * 2)], 'color': "#ffeaa7"}
                                    ]
                                }
                            ), row=2, col=1)
                            
                            # Abnormal Points gauge
                            abnormal_count = int(abnormal_mask.sum())
                            fig.add_trace(go.Indicator(
                                mode="gauge+number",
                                value=abnormal_count,
                                title={'text': "Abnormal Points"},
                                gauge={
                                    'axis': {'range': [0, max(10, abnormal_count * 2)]},
                                    'bar': {'color': "crimson"},
                                    'steps': [
                                        {'range': [0, 10], 'color': "#c8e6c9"},
                                        {'range': [10, 25], 'color': "#ffcc80"},
                                        {'range': [25, 100], 'color': "#ef5350"}
                                    ]
                                }
                            ), row=3, col=1)
                            
                            fig.update_layout(height=700, margin=dict(t=10, b=10))
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please upload at least one file to begin analysis.")
