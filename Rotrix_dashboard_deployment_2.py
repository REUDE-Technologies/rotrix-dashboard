#Modified script to support both single file analysis and comparative assessment modes


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

# Helper functions for data handling
def safe_get_range(df, column):
    """Safely get min and max values for a DataFrame column."""
    if not isinstance(df, pd.DataFrame) or column not in df.columns:
        return 0, 0
    try:
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            return 0, 0
        return float(series.min()), float(series.max())
    except:
        return 0, 0

def clean_dataframe(df):
    """Remove columns with all null values or no variation in data."""
    if df is None or not isinstance(df, pd.DataFrame):
        return df
        
    # Create a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # Remove columns with all null values
    null_columns = df_cleaned.columns[df_cleaned.isna().all()].tolist()
    if null_columns:
        df_cleaned = df_cleaned.drop(columns=null_columns)
    
    # Remove columns with constant values (no variation)
    constant_columns = df_cleaned.columns[df_cleaned.nunique() == 1].tolist()
    if constant_columns:
        df_cleaned = df_cleaned.drop(columns=constant_columns)
    
    return df_cleaned

def get_plot_ranges(b_df, v_df, x_axis, y_axis):
    """Get plot ranges for both DataFrames."""
    b_x_min, b_x_max = safe_get_range(b_df, x_axis)
    v_x_min, v_x_max = safe_get_range(v_df, x_axis)
    b_y_min, b_y_max = safe_get_range(b_df, y_axis)
    v_y_min, v_y_max = safe_get_range(v_df, y_axis)
    
    return (
        min(b_x_min, v_x_min),
        max(b_x_max, v_x_max),
        min(b_y_min, v_y_min),
        max(b_y_max, v_y_max)
    )

def get_numeric_columns(df):
    """Safely get numeric columns from a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        return []
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

def detect_abnormalities(series, threshold=3.0):
    """Detect abnormal points in a series using z-score threshold."""
    if len(series) < 2:  # Need at least 2 points to calculate z-score
        return pd.Series(False, index=series.index), pd.Series(0, index=series.index)
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold, z_scores

# Initialize session state variables if they don't exist
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None
if 'b_df' not in st.session_state:
    st.session_state.b_df = None
if 'v_df' not in st.session_state:
    st.session_state.v_df = None

# Function to change page
def change_page(page):
    st.session_state.current_page = page

# Utility functions
def load_csv(file):
    # Create a temporary file to store the content
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        # If file is a string (path), read directly
        if isinstance(file, str):
            with open(file, 'rb') as f:
                tmp_file.write(f.read())
        else:
            # If file is a file object, write its content
            file.seek(0)
            tmp_file.write(file.read())
        
        # Read the CSV from the temporary file
        return pd.read_csv(tmp_file.name)

def load_ulog(file, key_suffix=""):
    ALLOWED_TOPICS = set(t for t, _ in TOPIC_ASSESSMENT_PAIRS)
    
    # Create a temporary file to store the content
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ulg') as tmp_file:
        # If file is a string (path), read directly
        if isinstance(file, str):
            with open(file, 'rb') as f:
                tmp_file.write(f.read())
        else:
            # If file is a file object, write its content
            file.seek(0)
            tmp_file.write(file.read())
    
    # Process the ULog file
    try:
        ulog = ULog(tmp_file.name)
        extracted_dfs = {msg.name: pd.DataFrame(msg.data) for msg in ulog.data_list}
        filtered_dfs = {topic: df for topic, df in extracted_dfs.items() if topic in ALLOWED_TOPICS}
        topic_names = ["None"] + list(filtered_dfs.keys())
        if not topic_names:
            st.warning("No extractable topics found in ULOG file.")
            return {}, []
        return filtered_dfs, topic_names
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

def resample_to_common_time(df1, df2, freq=1.0):
    """Resample two dataframes to a common time base."""
    if 'timestamp_seconds' not in df1.columns or 'timestamp_seconds' not in df2.columns:
        st.error("timestamp_seconds column missing in one or both dataframes")
        return df1, df2, []
        
    try:
        df1 = df1.copy().sort_values('timestamp_seconds')
        df2 = df2.copy().sort_values('timestamp_seconds')
        
        start = max(df1['timestamp_seconds'].min(), df2['timestamp_seconds'].min())
        end = min(df1['timestamp_seconds'].max(), df2['timestamp_seconds'].max())
        
        if start >= end:
            st.error("No overlapping time range for resampling")
            return df1, df2, []
            
        common_time = np.arange(start, end, freq)
        if len(common_time) == 0:
            st.error("No common time points found after resampling")
            return df1, df2, []
        
        df1_interp = df1.set_index('timestamp_seconds').interpolate(method='linear').reindex(common_time, method='nearest')
        df2_interp = df2.set_index('timestamp_seconds').interpolate(method='linear').reindex(common_time, method='nearest')
        
        df1_interp = df1_interp.reset_index().rename(columns={'index': 'timestamp_seconds'})
        df2_interp = df2_interp.reset_index().rename(columns={'index': 'timestamp_seconds'})
        
        return df1_interp, df2_interp, common_time
    except Exception as e:
        st.error(f"Error during resampling: {str(e)}")
        return df1, df2, []

def load_data(file, filetype, key_suffix):
    """Load data from file and ensure proper timestamp handling."""
    if filetype == ".csv":
        df_csv = load_csv(file)
        df_csv = convert_timestamps_to_seconds(df_csv)
        df_csv = ensure_seconds_column(df_csv)
        return df_csv, None
    elif filetype == ".ulg":
        df_ulog, topic_names = load_ulog(file, key_suffix)
        if isinstance(df_ulog, dict):
            # Handle each dataframe in the dictionary
            for topic in df_ulog:
                df_ulog[topic] = convert_timestamps_to_seconds(df_ulog[topic])
                df_ulog[topic] = ensure_seconds_column(df_ulog[topic])
        return df_ulog, topic_names
    return None, None

def convert_timestamps_to_seconds(df):
    """Convert timestamp columns to seconds."""
    if df is None:
        return df
    if isinstance(df, pd.DataFrame) and len(df.index) > 0:
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
        for col in timestamp_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col] / 1000000
    return df

def ensure_seconds_column(df):
    """Ensure a 'timestamp_seconds' column exists and is always in seconds from start"""
    if df is None or df.empty:
        return df
    timestamp_col = next((col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower()), None)
    if timestamp_col is None:
        return df
    series = df[timestamp_col]
    
    # Get the current topic from the dataframe name or context
    current_topic = None
    for topic, _ in TOPIC_ASSESSMENT_PAIRS:
        if topic in str(df.name) if hasattr(df, 'name') else False:
            current_topic = topic
            break
    
    # Skip special conversion for Control topic
    if current_topic == "px4io_status":  # Control topic
        if series.max() > 1e12:
            df['timestamp_seconds'] = (series - series.min()) / 1e6
        elif series.max() > 1e9:
            df['timestamp_seconds'] = (series - series.min()) / 1e3
        else:
            df['timestamp_seconds'] = series - series.min()
    else:
        # For all other topics, move decimal point 3 places to the left
        df['timestamp_seconds'] = (series - series.min()) / 1000000
    
    # Drop timestamp_sample if it exists
    if 'timestamp_sample' in df.columns:
        df.drop('timestamp_sample', axis=1, inplace=True)
    
    return df

def add_remove_column(target_df, df_name="DataFrame"):
    if target_df is None or target_df.empty:
        st.warning(f"⚠ {df_name} is empty or not loaded.")
        return target_df
    
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

    st.markdown("##### 🗑 Remove Column")
    columns_to_drop = st.multiselect("Select columns to drop", target_df.columns, key=f"{df_name}_drop")

    if st.button(f"Remove Column from {df_name}"):
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
    "Thrust": ["thrust[0]", "thrust[1]", "thrust[2]"],
    "Torque": ["xyz[0]", "xyz[1]", "xyz[2]"],
    "Control": ["pwm[0]", "pwm[1]", "pwm[2]", "pwm[3]"],
    "Battery": ["voltage_v", "current_average_a", "discharged_mah"],
}

# 🔹 Logo
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# logo_base64 = get_base64_image(os.path.join(os.path.dirname(__file__), "Rotrix-Logo.png"))
# # st.logo(logo_base64, *, size="medium", link=None, icon_image=None)
# st.markdown(f"""
#     <div style="display: flex; position: fixed; top:50px; left: 50px; z-index:50; justify-content: left; align-items: center; padding: 1px; background-color:white; border-radius:25px;">
#         <a href="http://rotrixdemo.reude.tech/" target="_blank">
#             <img src="data:image/png;base64,{logo_base64}" width="180" alt="Rotrix Logo">
#         </a>
#     </div>
# """, unsafe_allow_html=True)

# Home Page
if st.session_state.current_page == 'home':
    # Logo is already handled separately

    # st.markdown("""
    # <div style='text-align: center; padding: 20px 0;'>
    #     <h1 style='color: #2E86C1; display: flex; align-items: center; justify-content: center; gap: 10px; margin: 0;'>
    #         <span>🚀</span> ROTRIX Analysis Dashboard
    #     </h1>
    #     <p style='color: #666; margin: 10px 0 0 0;'>Advanced data analysis and visualization platform</p>
    # </div>
    # """, unsafe_allow_html=True)

    # Data Source Selection
    st.markdown("""
    <div style='padding: 0 0 10px 0;'>
        <div style='display: flex; align-items: center; gap: 8px;'>
            <span style='font-size: 1.2em;'>📊</span>
            <h2 style='color: #2E86C1; margin: 0; font-size: 1.2em;'>Data Assessment</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    data_source = st.radio(
        "Choose your preferred data source for analysis",
        ["ROTRIX Account", "My Shared Files", "Other"],
        #help="Choose your preferred data source for analysis",
        horizontal=False
    )
    st.session_state.data_source = data_source

    if data_source == "ROTRIX Account":
        st.markdown("""
        <style>
        div[data-testid="stExpander"] {
            max-width: 600px;
        }
        div.stButton button {
            float: right;
            margin-right: -10px;
        }
        </style>
        """, unsafe_allow_html=True)
        with st.expander("ROTRIX Account Details", expanded=True):
            col1, col2 = st.columns([4, 0.5])
            with col1:
                st.text_input("Account ID", value="ROTRIX_123", disabled=True)
                st.text_input("Password", value="••••••••", type="password", disabled=True)
                login_col1, login_col2 = st.columns([1, 2])
                with login_col1:
                    if st.button("Login", type="primary", use_container_width=True):
                        st.success("Login successful!")
                        st.rerun()
            with col2:
                if st.button("❌", key="close_account", help="Close"):
                    st.rerun()
        st.info("🚧 This feature is currently under development. Please select 'Other' to proceed.")
        st.stop()
    elif data_source == "My Shared Files":
        st.markdown("""
        <style>
        div[data-testid="stExpander"] {
            max-width: 600px;
        }
        div.stButton button {
            float: right;
            margin-right: -10px;
        }
        </style>
        """, unsafe_allow_html=True)
        with st.expander("My Shared Files", expanded=True):
            col1, col2 = st.columns([4, 0.5])
            with col1:
                st.markdown("### Recent Files")
                st.markdown("- 📄 analysis_data_001.ulg")
                st.markdown("- 📄 benchmark_test.csv")
                st.markdown("- 📄 performance_log.ulg")
            with col2:
                if st.button("❌", key="close_shared", help="Close"):
                    st.rerun()
        st.info("🚧 This feature is currently under development. Please select 'Other' to proceed.")
        st.stop()
    elif data_source == "B":
        st.info("🚧 This feature is currently under development. Please select 'Other' to proceed.")
        st.stop()

    # st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

    # col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     analysis_type = st.radio(
    #         "Choose the type of analysis you want to perform",
    #         ["Single File Analysis", "Comparative Analysis"],
    #         #help="Choose the type of analysis you want to perform"
    #     )
    
    # with col2:
    #     st.markdown("""
    #     <div style='background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #e1e4e8; height: 100%;'>
    #         <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 10px;'>
    #             <span>📈</span>
    #             <h3 style='color: #2E86C1; margin: 0; font-size: 1em;'>Single File Analysis</h3>
    #         </div>
    #         <p style='color: #666; margin: 0 0 10px 0; font-size: 0.95em;'>Analyze individual data files</p>
    #         <ul style='color: #666; padding-left: 20px; margin: 0; font-size: 0.95em;'>
    #             <li>Data visualization</li>
    #             <li>Statistical analysis</li>
    #             <li>Anomaly detection</li>
    #             <li>Performance metrics</li>
    #         </ul>
    #     </div>
    #     """, unsafe_allow_html=True)

    # with col3:
    #     st.markdown("""
    #     <div style='background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #e1e4e8; height: 100%;'>
    #         <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 10px;'>
    #             <span>🔄</span>
    #             <h3 style='color: #2E86C1; margin: 0; font-size: 1em;'>Comparative Analysis</h3>
    #         </div>
    #         <p style='color: #666; margin: 0 0 10px 0; font-size: 0.95em;'>Compare two data files</p>
    #         <ul style='color: #666; padding-left: 20px; margin: 0; font-size: 0.95em;'>
    #             <li>Side-by-side comparison</li>
    #             <li>Difference analysis</li>
    #             <li>Performance benchmarking</li>
    #             <li>Trend visualization</li>
    #         </ul>
    #     </div>
    #     """, unsafe_allow_html=True)

    # st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
    analysis_type = st.radio(
        "Choose the type of analysis you want to perform",
        ["Single File Analysis", "Comparative Analysis"],
        #help="Choose the type of analysis you want to perform"
    )
    st.session_state.analysis_type = analysis_type

    # File upload section based on analysis type
    if analysis_type == "Single File Analysis":
        st.markdown("<h4 style='font-size:18px; color:#4B8BBE;'>🔼 Upload Analysis File</h4>", unsafe_allow_html=True)
        
        # File upload layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Initialize uploaded files list in session state
            if "uploaded_files" not in st.session_state:
                st.session_state.uploaded_files = []
            if "show_share_modal" not in st.session_state:
                st.session_state.show_share_modal = None

            # Upload file with hidden default display
            uploaded_file = st.file_uploader("", type=["csv", "ulg"], key="uploader", label_visibility="collapsed")

            # Add to session state list if it's a new file
            if uploaded_file and uploaded_file.name not in [f.name for f in st.session_state.uploaded_files]:
                st.session_state.uploaded_files.append(uploaded_file)

            # Display custom file list
            for i, file in enumerate(st.session_state.uploaded_files):
                col_name, col_share, col_remove = st.columns([8, 1, 1])
                with col_name:
                    st.markdown(f"📎 {file.name}")
                with col_share:
                    if st.button("🔗", key=f"share_{i}", help="Copy share link"):
                        st.session_state.show_share_modal = i
                with col_remove:
                    if st.button("🗑️", key=f"remove_{i}", help="Remove file"):
                        st.session_state.uploaded_files.pop(i)
                        st.rerun()
                
                # Show share modal for this file if selected
                if st.session_state.show_share_modal == i:
                    with st.expander("Share Options", expanded=True):
                        # st.markdown("### Share Options")
                        
                        # Create columns for share options
                        share_col1, share_col2, share_col3, share_col4 = st.columns([1, 1, 1, 0.5])
                        
                        with share_col1:
                            if st.button("📁 My Shared Files", key=f"shared_files_{i}", use_container_width=True):
                                st.toast("Opening My Shared Files...")
                                # Add shared files logic here
                        
                        with share_col2:
                            if st.button("👤 Rotrix Account", key=f"rotrix_account_{i}", use_container_width=True):
                                st.toast("Opening Rotrix Account...")
                                # Add Rotrix account logic here
                        
                        with share_col3:
                            share_link = f"http://localhost:8501/share?file={file.name}"
                            if st.button("📋 Copy Link", key=f"copy_link_{i}", use_container_width=True):
                                st.code(share_link, language="text")
                                st.toast("Link copied to clipboard!")
                        
                        with share_col4:
                            if st.button("❌", key=f"close_share_{i}", use_container_width=True):
                                st.session_state.show_share_modal = None
                                st.rerun()

            if not st.session_state.uploaded_files:
                st.info("📂 Upload File")
        
        with col2:
            selected_file = st.selectbox("Select File", ["None"] + [f.name for f in st.session_state.uploaded_files], key="file_selector")
            if selected_file == "None":
                st.info("📋 Select File")
            else:
                st.success(f"✅ {selected_file}")
                
            # Store selected file in session state
            st.session_state.selected_single_file = selected_file if selected_file != "None" else None
            if selected_file != "None" and st.session_state.uploaded_files:
                try:
                    file = [f for f in st.session_state.uploaded_files if f.name == selected_file][0]
                    # Store file content in session state
                    file.seek(0)
                    st.session_state.selected_single_file_content = file.read()
                    file.seek(0)  # Reset file pointer for future use
                except Exception as e:
                    st.error("Error loading file")
                
    else:  # Comparative Analysis
        st.markdown("<h4 style='font-size:18px; color:#4B8BBE;'>🔼 Upload Benchmark & Target Files</h4>", unsafe_allow_html=True)
        
        # File upload layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Initialize uploaded files list in session state
            if "benchmark_files" not in st.session_state:
                st.session_state.benchmark_files = []
            if "show_share_modal_bench" not in st.session_state:
                st.session_state.show_share_modal_bench = None

            benchmark_file = st.file_uploader("", type=["csv", "ulg"], key="benchmark_uploader", label_visibility="collapsed")
            if benchmark_file and benchmark_file.name not in [f.name for f in st.session_state.benchmark_files]:
                st.session_state.benchmark_files.append(benchmark_file)

            # Display custom benchmark file list
            for i, file in enumerate(st.session_state.benchmark_files):
                col_name, col_share, col_remove = st.columns([8, 1, 1])
                with col_name:
                    st.markdown(f"📎 {file.name}")
                with col_share:
                    if st.button("🔗", key=f"share_bench_{i}", help="Copy share link"):
                        st.session_state.show_share_modal_bench = i
                with col_remove:
                    if st.button("🗑️", key=f"remove_bench_{i}", help="Remove file"):
                        st.session_state.benchmark_files.pop(i)
                        st.rerun()
                
                # Show share modal for benchmark file
                if st.session_state.show_share_modal_bench == i:
                    with st.expander("Share Options", expanded=True):
                        share_col1, share_col2, share_col3, share_col4 = st.columns([1, 1, 1, 0.5])
                        
                        with share_col1:
                            if st.button("📁 My Shared Files", key=f"shared_files_bench_{i}", use_container_width=True):
                                st.toast("Opening My Shared Files...")
                                # Add shared files logic here
                        
                        with share_col2:
                            if st.button("👤 Rotrix Account", key=f"rotrix_account_bench_{i}", use_container_width=True):
                                st.toast("Opening Rotrix Account...")
                                # Add Rotrix account logic here
                        
                        with share_col3:
                            share_link = f"http://localhost:8501/share?file={file.name}"
                            if st.button("📋 Copy Link", key=f"copy_link_bench_{i}", use_container_width=True):
                                st.code(share_link, language="text")
                                st.toast("Link copied to clipboard!")
                        
                        with share_col4:
                            if st.button("❌", key=f"close_share_bench_{i}", use_container_width=True):
                                st.session_state.show_share_modal_bench = None
                                st.rerun()

            if not st.session_state.benchmark_files:
                st.info("📂 Upload Benchmark File")
        
        with col2:
            # Initialize uploaded files list in session state
            if "target_files" not in st.session_state:
                st.session_state.target_files = []
            if "show_share_modal_target" not in st.session_state:
                st.session_state.show_share_modal_target = None

            target_file = st.file_uploader("", type=["csv", "ulg"], key="target_uploader", label_visibility="collapsed")
            if target_file and target_file.name not in [f.name for f in st.session_state.target_files]:
                st.session_state.target_files.append(target_file)

            # Display custom target file list
            for i, file in enumerate(st.session_state.target_files):
                col_name, col_share, col_remove = st.columns([8, 1, 1])
                with col_name:
                    st.markdown(f"📎 {file.name}")
                with col_share:
                    if st.button("🔗", key=f"share_target_{i}", help="Copy share link"):
                        st.session_state.show_share_modal_target = i
                with col_remove:
                    if st.button("🗑️", key=f"remove_target_{i}", help="Remove file"):
                        st.session_state.target_files.pop(i)
                        st.rerun()
                
                # Show share modal for target file
                if st.session_state.show_share_modal_target == i:
                    with st.expander("Share Options", expanded=True):
                        share_col1, share_col2, share_col3, share_col4 = st.columns([1, 1, 1, 0.5])
                        
                        with share_col1:
                            if st.button("📁 My Shared Files", key=f"shared_files_target_{i}", use_container_width=True):
                                st.toast("Opening My Shared Files...")
                                # Add shared files logic here
                        
                        with share_col2:
                            if st.button("👤 Rotrix Account", key=f"rotrix_account_target_{i}", use_container_width=True):
                                st.toast("Opening Rotrix Account...")
                                # Add Rotrix account logic here
                        
                        with share_col3:
                            share_link = f"http://localhost:8501/share?file={file.name}"
                            if st.button("📋 Copy Link", key=f"copy_link_target_{i}", use_container_width=True):
                                st.code(share_link, language="text")
                                st.toast("Link copied to clipboard!")
                        
                        with share_col4:
                            if st.button("❌", key=f"close_share_target_{i}", use_container_width=True):
                                st.session_state.show_share_modal_target = None
                                st.rerun()

            if not st.session_state.target_files:
                st.info("📂 Upload Target File")
        
        with col3:
            selected_bench = st.selectbox("Select Benchmark File", ["None"] + [f.name for f in st.session_state.benchmark_files])
            if selected_bench == "None":
                st.info("📋 Select a Benchmark file to proceed")
            
            # Store benchmark selection in session state
            st.session_state.selected_bench = selected_bench if selected_bench != "None" else None
            if selected_bench != "None" and st.session_state.benchmark_files:
                try:
                    file = [f for f in st.session_state.benchmark_files if f.name == selected_bench][0]
                    # Store file content in session state
                    file.seek(0)
                    st.session_state.selected_bench_content = file.read()
                    file.seek(0)  # Reset file pointer for future use
                except Exception as e:
                    st.error("Error loading benchmark file")
        
        with col4:
            selected_val = st.selectbox("Select Target File", ["None"] + [f.name for f in st.session_state.target_files])
            if selected_val == "None":
                st.info("📋 Select a Target file to proceed")
            
            # Store validation selection in session state
            st.session_state.selected_val = selected_val if selected_val != "None" else None
            if selected_val != "None" and st.session_state.target_files:
                try:
                    file = [f for f in st.session_state.target_files if f.name == selected_val][0]
                    # Store file content in session state
                    file.seek(0)
                    st.session_state.selected_val_content = file.read()
                    file.seek(0)  # Reset file pointer for future use
                except Exception as e:
                    st.error("Error loading target file")

    # Topic selection for ULG files if needed
    if st.session_state.analysis_type == "Single File Analysis":
        selected_file = st.session_state.get('selected_single_file')
        if selected_file and selected_file.endswith('.ulg'):
            st.markdown("<h4 style='font-size:18px; color:#4B8BBE;'>📊 Select Topic</h4>", unsafe_allow_html=True)
            assessment_names = ["None"] + [a for _, a in TOPIC_ASSESSMENT_PAIRS]
            selected_assessment = st.selectbox("Select Topic", options=assessment_names)
            if selected_assessment == "None":
                st.info("Please select a topic to analyze")
            st.session_state.selected_assessment = selected_assessment
    else:  # Comparative Analysis 
        # For comparative analysis, only show topic selection if both files are ULG
        selected_bench = st.session_state.get('selected_bench')
        selected_val = st.session_state.get('selected_val')
        if (selected_bench and selected_val and 
            selected_bench.endswith('.ulg') and selected_val.endswith('.ulg')):
            st.markdown("<h4 style='font-size:18px; color:#4B8BBE;'>📊 Select Topic</h4>", unsafe_allow_html=True)
            assessment_names = ["None"] + [a for _, a in TOPIC_ASSESSMENT_PAIRS]
            selected_assessment = st.selectbox("Select Topic", options=assessment_names)
            if selected_assessment == "None":
                st.info("Please select a topic to analyze")
            st.session_state.selected_assessment = selected_assessment

    # Proceed Button
    proceed_disabled = False
    if st.session_state.analysis_type == "Single File Analysis":
        selected_file = st.session_state.get('selected_single_file')
        if not selected_file:
            proceed_disabled = True
        elif selected_file and selected_file.endswith('.ulg'):
            if not st.session_state.get('selected_assessment') or st.session_state.get('selected_assessment') == "None":
                proceed_disabled = True
    else:  # Comparative Analysis
        selected_bench = st.session_state.get('selected_bench')
        selected_val = st.session_state.get('selected_val')
        if not (selected_bench and selected_val):
            proceed_disabled = True
        elif (selected_bench and selected_val and 
              selected_bench.endswith('.ulg') and selected_val.endswith('.ulg')):
            if not st.session_state.get('selected_assessment') or st.session_state.get('selected_assessment') == "None":
                proceed_disabled = True
    
    if st.button("Proceed to Analysis", 
                 help="Click to continue with your selected analysis type",
                 type="primary",
                 use_container_width=True,
                 disabled=proceed_disabled):
        if st.session_state.analysis_type == "Single File Analysis":
            change_page('single_analysis')
        else:
            change_page('comparative_analysis')
        st.rerun()

# Single File Analysis Page
elif st.session_state.current_page == 'single_analysis':
    # Title and back button in the same line
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.markdown("<h3 style='font-size: 20px;'>🔍 Single File Analysis</h3>", unsafe_allow_html=True)
    with col2:
        if st.button("← Back to Home", use_container_width=True):
            change_page('home')
            st.rerun()
    
    # Initialize variables
    selected_file = st.session_state.get('selected_single_file', "None")
    selected_assessment = "None"
    df = None
    file_ext = None
    
    # Topic selection for ULG files
    if selected_file != "None":
        file_ext = os.path.splitext(selected_file)[-1].lower()
        if file_ext == ".ulg":
            assessment_names = ["None"] + [a for _, a in TOPIC_ASSESSMENT_PAIRS]
            assessment_to_topic = {a: t for t, a in TOPIC_ASSESSMENT_PAIRS}
            
            # Get the topic selected from home page
            default_assessment = st.session_state.get('selected_assessment')
            default_index = 0  # Default to "None"
            if isinstance(default_assessment, str) and default_assessment in assessment_names:
                default_index = assessment_names.index(default_assessment)
            
            selected_assessment = st.selectbox(
                "Change Topic", 
                options=assessment_names,
                index=default_index
            )
            
            # Update session state with new topic selection
            if selected_assessment != st.session_state.get('selected_assessment'):
                st.session_state.selected_assessment = selected_assessment
            
            if selected_assessment == "None":
                st.info("Please select a topic to analyze")
    
    # Process the selected file
    if selected_file != "None":
        file_content = st.session_state.get('selected_single_file_content')
        if file_content:
            # Get file extension from selected file name
            file_ext = os.path.splitext(selected_file)[-1].lower()
            df = None
            
            # Create a temporary file with the content
            tmp_file = None
            try:
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
                if isinstance(file_content, str):
                    tmp_file.write(file_content.encode('utf-8'))
                else:
                    tmp_file.write(file_content)
                tmp_file.flush()
                tmp_file.close()
                
                if file_ext == ".ulg":
                    dfs, topics = load_ulog(tmp_file.name)
                    # Get topic from session state or current selection
                    current_assessment = st.session_state.get('selected_assessment', selected_assessment)
                    if current_assessment and current_assessment != "None":
                        selected_topic = assessment_to_topic.get(str(current_assessment))
                        if selected_topic and selected_topic in dfs:
                            df = dfs[selected_topic]
                            df = ensure_seconds_column(df)
                            if 'Index' not in df.columns:
                                df.insert(0, 'Index', range(1, len(df) + 1))
                            if 'timestamp_seconds' not in df.columns:
                                df['timestamp_seconds'] = df.index
                else:
                    df, _ = load_data(tmp_file.name, file_ext, "")
                    df = ensure_seconds_column(df)
                    if 'Index' not in df.columns:
                        df.insert(0, 'Index', range(1, len(df) + 1))
                    if 'timestamp_seconds' not in df.columns:
                        df['timestamp_seconds'] = df.index
            finally:
                # Clean up temporary file
                if tmp_file:
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass
            
            if df is not None and isinstance(df, pd.DataFrame) and len(df.index) > 0:
                # Add Index column if it doesn't exist
                if 'Index' not in df.columns:
                    df.insert(0, "Index", range(1, len(df) + 1))
                
                # Analysis tabs
                tab1, tab2 = st.tabs(["📊 Plot", "📋 Data"])
                
                with tab1:
                    # st.markdown("<h3 style='font-size: 20px;'>📈 Plot Visualization</h3>", unsafe_allow_html=True)
                    col1, col2 = st.columns([0.2, 0.8])
                    
                    with col1:
                        # st.markdown("<h3 style='font-size: 20px;'>🔍 Single File Analysis</h3>", unsafe_allow_html=True)
                        st.markdown("<h4 style='font-size: 18px;'>📈 Parameters</h4>", unsafe_allow_html=True)
                        # Get selected assessment/topic
                        if file_ext == ".ulg" and selected_assessment and selected_assessment != "None":
                            allowed_y_axis = ASSESSMENT_Y_AXIS_MAP.get(selected_assessment, [])
                            allowed_y_axis = [col for col in allowed_y_axis if col in df.columns]
                            if not allowed_y_axis:
                                allowed_y_axis = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                        else:
                            allowed_y_axis = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                        
                        ALLOWED_X_AXIS = ["Index", "timestamp_seconds"] + [col for col in allowed_y_axis if col not in ["Index", "timestamp_seconds"]]
                        
                        # Set default x_axis based on file type
                        if file_ext == ".ulg":
                            default_x = 'timestamp_seconds' if 'timestamp_seconds' in ALLOWED_X_AXIS else 'Index'
                        else:
                            default_x = 'Index' if 'Index' in ALLOWED_X_AXIS else ALLOWED_X_AXIS[0]
                        
                        x_axis = st.selectbox("X-Axis", ALLOWED_X_AXIS, key="x_axis_single", index=ALLOWED_X_AXIS.index(default_x))
                        
                        # Set default y_axis based on file type and available columns
                        if file_ext == ".csv" and 'cD2detailpeak' in allowed_y_axis:
                            default_y = 'cD2detailpeak'
                        else:
                            default_y = allowed_y_axis[0] if allowed_y_axis else None
                        
                        y_axis = st.selectbox("Y-Axis", allowed_y_axis, key="y_axis_single", 
                                            index=allowed_y_axis.index(default_y) if default_y in allowed_y_axis else 0)
                        
                        if x_axis and y_axis:
                            z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, key="z-slider-single")
                            col_min, col_max = st.columns(2)
                            with col_min:
                                x_min = st.number_input("X min", value=float(df[x_axis].min()))
                                y_min = st.number_input("Y min", value=float(df[y_axis].min()))
                            with col_max:
                                x_max = st.number_input("X max", value=float(df[x_axis].max()))
                                y_max = st.number_input("Y max", value=float(df[y_axis].max()))

                    with col2:
                        if x_axis and y_axis:
                            try:
                                # Filter data
                                filtered_df = df[(df[x_axis] >= x_min) & (df[x_axis] <= x_max) &
                                               (df[y_axis] >= y_min) & (df[y_axis] <= y_max)]
                                if len(filtered_df.index) > 0:
                                    # Calculate statistics and detect abnormalities
                                    stats = filtered_df[y_axis].describe()
                                    abnormal_mask, z_scores = detect_abnormalities(filtered_df[y_axis], z_threshold)
                                    filtered_df["Z_Score"] = z_scores
                                    filtered_df["Abnormal"] = abnormal_mask
                                    abnormal_points = filtered_df[filtered_df["Abnormal"]]
                                    
                                    # Display metrics
                                    st.markdown("### 🎯 Metrics")
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
                                    
                                    # Create plot
                                    st.markdown("### 🧮 Plot Visualization")
                                    plot_container = st.container()
                                    with plot_container:
                                        fig = go.Figure()
                                        
                                        # Add main line plot
                                        fig.add_trace(go.Scatter(
                                            x=filtered_df[x_axis],
                                            y=filtered_df[y_axis],
                                            mode='lines',
                                            name='Data'
                                        ))
                                        
                                        # Add abnormal points
                                        if not abnormal_points.empty:
                                            fig.add_trace(go.Scatter(
                                                x=abnormal_points[x_axis],
                                                y=abnormal_points[y_axis],
                                                mode='markers',
                                                marker=dict(color='red', size=8),
                                                name='Abnormal Points'
                                            ))
                                        
                                        # Add mean line
                                        mean_value = filtered_df[y_axis].mean()
                                        fig.add_hline(y=mean_value, line_dash="dash", line_color="green",
                                                    annotation_text=f"Mean: {mean_value:.2f}")
                                        
                                        # Get timestamp ticks if needed
                                        if x_axis == 'timestamp_seconds':
                                            tick_vals, tick_texts = get_timestamp_ticks(filtered_df[x_axis])
                                            fig.update_xaxes(
                                                tickvals=tick_vals,
                                                ticktext=tick_texts,
                                                title_text=get_axis_title(x_axis),
                                                type='linear'
                                            )
                                        else:
                                            fig.update_xaxes(title_text=x_axis)
                                        
                                        fig.update_layout(
                                            height=450,  # Reduced from 900 to fit viewport
                                            showlegend=True,
                                            legend=dict(
                                                orientation="h",
                                                yanchor="bottom",
                                                y=1.02,  # Slightly adjusted to prevent overlap
                                                xanchor="center",
                                                x=0.5
                                            ),
                                            margin=dict(t=30, b=20, l=50, r=20),  # Optimized margins
                                            yaxis=dict(
                                                showticklabels=True,
                                                title=y_axis
                                            )
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                               st.error(f"Error creating plot: {str(e)}")
                    
                with tab2:
                    st.markdown("<h3 style='font-size: 20px;'>📋 Data Preview</h3>", unsafe_allow_html=True)
                    
                    # Create a 20-80 split layout
                    settings_col, data_col = st.columns([0.2, 0.8])
                    
                    with settings_col:
                        # Add dataset selector and column management
                        st.markdown("<h5 style='font-size: 16px;'>🔧 Column Management</h5>", unsafe_allow_html=True)
                        if isinstance(df, pd.DataFrame):
                            df = add_remove_column(df, "Dataset")
                    
                    with data_col:
                        if isinstance(df, pd.DataFrame) and len(df.index) > 0:
                            # For ULG files after topic selection, show only selected axes
                            if file_ext == ".ulg" and selected_assessment and selected_assessment != "None" and x_axis and y_axis:
                                # Only add Index column if it doesn't exist
                                if 'Index' not in df.columns:
                                    df.insert(0, "Index", range(1, len(df) + 1))
                                
                                # Create a list of columns to display in order
                                display_cols = []
                                # Always start with Index
                                if 'Index' in df.columns:
                                    display_cols.append('Index')
                                # Add timestamp_seconds next if available
                                if 'timestamp_seconds' in df.columns:
                                    display_cols.append('timestamp_seconds')
                                # Add selected axes if not already included
                                if x_axis not in display_cols:
                                    display_cols.append(x_axis)
                                if y_axis not in display_cols:
                                    display_cols.append(y_axis)
                                
                                # Add any numeric columns from the assessment map if available
                                if selected_assessment in ASSESSMENT_Y_AXIS_MAP:
                                    for col in ASSESSMENT_Y_AXIS_MAP[selected_assessment]:
                                        if col in df.columns and col not in display_cols and pd.api.types.is_numeric_dtype(df[col]):
                                            display_cols.append(col)
                                
                                # Display the filtered DataFrame
                                st.dataframe(
                                    df[list(dict.fromkeys(display_cols))],  # Remove duplicates while preserving order
                                    use_container_width=True,
                                    height=500  # Increased height for better visibility
                                )
                            else:
                                # For CSV files or when no topic/axes selected
                                display_cols = ['Index']
                                if 'timestamp_seconds' in df.columns:
                                    display_cols.append('timestamp_seconds')
                                
                                # Add all numeric columns
                                numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) 
                                              and col not in display_cols]
                                display_cols.extend(numeric_cols)
                                
                                st.dataframe(
                                    df[list(dict.fromkeys(display_cols))],  # Remove duplicates while preserving order
                                    use_container_width=True,
                                    height=500
                                )
                # except:
                #     st.error("Error loading file")
                #     df = None
    else:
        st.info("Please upload a valid data file to begin analysis.")

# Comparative Analysis Page
elif st.session_state.current_page == 'comparative_analysis':
    # Title and back button in the same line
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.markdown("<h3 style='font-size: 20px;'>🚀 Comparative Analysis</h3>", unsafe_allow_html=True)
    with col2:
        if st.button("← Back to Home", use_container_width=True):
            change_page('home')
            st.rerun()
    
    # Initialize variables
    b_df = None
    v_df = None
    b_file_ext = None
    v_file_ext = None
    b_dfs = {}
    v_dfs = {}
    selected_bench = st.session_state.get('selected_bench', "None")
    selected_val = st.session_state.get('selected_val', "None")

    # Process benchmark file
    if selected_bench != "None":
        try:
            b_content = st.session_state.get('selected_bench_content')
            if b_content:
                b_file_ext = os.path.splitext(selected_bench)[-1].lower()
                tmp_file = None
                try:
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=b_file_ext)
                    if isinstance(b_content, str):
                        tmp_file.write(b_content.encode('utf-8'))
                    else:
                        tmp_file.write(b_content)
                    tmp_file.flush()
                    tmp_file.close()
                    
                    if b_file_ext == ".ulg":
                        b_dfs, b_topics = load_ulog(tmp_file.name)
                    else:
                        df, _ = load_data(tmp_file.name, b_file_ext, key_suffix="bench")
                        if df is not None and isinstance(df, pd.DataFrame) and len(df.index) > 0:
                            b_df = df
                            st.session_state.b_df = df
                finally:
                    if tmp_file:
                        try:
                            os.unlink(tmp_file.name)
                        except:
                            pass
        except Exception as e:
            st.error(f"Error processing benchmark file: {str(e)}")

    # Process validation file
    if selected_val != "None":
        try:
            v_content = st.session_state.get('selected_val_content')
            if v_content:
                v_file_ext = os.path.splitext(selected_val)[-1].lower()
                tmp_file = None
                try:
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=v_file_ext)
                    if isinstance(v_content, str):
                        tmp_file.write(v_content.encode('utf-8'))
                    else:
                        tmp_file.write(v_content)
                    tmp_file.flush()
                    tmp_file.close()
                    
                    if v_file_ext == ".ulg":
                        v_dfs, v_topics = load_ulog(tmp_file.name)
                    else:
                        df, _ = load_data(tmp_file.name, v_file_ext, key_suffix="val")
                        if df is not None and isinstance(df, pd.DataFrame) and len(df.index) > 0:
                            v_df = df
                            st.session_state.v_df = df
                finally:
                    if tmp_file:
                        try:
                            os.unlink(tmp_file.name)
                        except:
                            pass
        except Exception as e:
            st.error(f"Error processing target file: {str(e)}")
            
    # Show topic selection only after both ULG files are selected
    if (selected_bench != "None" and selected_val != "None" and 
        b_file_ext == ".ulg" and v_file_ext == ".ulg"):
        
        assessment_names = ["None"] + [a for _, a in TOPIC_ASSESSMENT_PAIRS]
        assessment_to_topic = {a: t for t, a in TOPIC_ASSESSMENT_PAIRS}
        
        # Get the topic selected from home page
        default_assessment = st.session_state.get('selected_assessment')
        
        # If we have a valid topic from home page, use it directly
        if default_assessment and default_assessment != "None":
            selected_topic = assessment_to_topic.get(str(default_assessment))
            if selected_topic:
                if selected_topic in b_dfs and selected_topic in v_dfs:
                    b_df = b_dfs[selected_topic]
                    v_df = v_dfs[selected_topic]
                    st.session_state.b_df = b_dfs[selected_topic]
                    st.session_state.v_df = v_dfs[selected_topic]
                else:
                    st.warning(f"⚠️ Topic '{selected_topic}' not found in one or both files")
        
        # Show topic selection dropdown with default from home page
        # st.markdown("<h3 style='font-size: 20px;'>📊 Analysis Topic</h3>", unsafe_allow_html=True)
        
        # Handle default topic selection
        default_index = 0  # Default to "None"
        if isinstance(default_assessment, str) and default_assessment in assessment_names:
            default_index = assessment_names.index(default_assessment)
            
        selected_assessment = st.selectbox(
            "Change Topic", 
            options=assessment_names,
            index=default_index,
            key="common_topic"
        )
        
        # Update data if topic is changed
        if selected_assessment != "None" and selected_assessment != default_assessment:
            selected_topic = assessment_to_topic.get(str(selected_assessment))
            if selected_topic:
                if selected_topic in b_dfs and selected_topic in v_dfs:
                    b_df = b_dfs[selected_topic]
                    v_df = v_dfs[selected_topic]
                    st.session_state.b_df = b_dfs[selected_topic]
                    st.session_state.v_df = v_dfs[selected_topic]
                    # Update session state with new topic
                    st.session_state.selected_assessment = selected_assessment
                else:
                    st.warning(f"⚠️ Topic '{selected_topic}' not found in one or both files")
    elif selected_bench != "None" and selected_val != "None":
        # For non-ULG files, get data from session state
        b_df = st.session_state.get("b_df", None)
        v_df = st.session_state.get("v_df", None)

    # Show analysis tabs only if both files are loaded and selected
    if b_df is not None and v_df is not None:
        tab1, tab2 = st.tabs(["📊 Plot", "📋 Data"])
        
        # Data Tab
        with tab2:
            st.markdown("<h3 style='font-size: 20px;'>📋 Data Preview</h3>", unsafe_allow_html=True)
            
            # Create a 20-80 split layout
            settings_col, data_col = st.columns([0.2, 0.8])
            
            with settings_col:
                # Add dataset selector and column management
                st.markdown("<h5 style='font-size: 16px;'>🔧 Column Management</h5>", unsafe_allow_html=True)
                dataset_choice = st.selectbox(
                    "Select Dataset to Edit",
                    ["Benchmark", "Target", "Both"],
                    key="dataset_selector"
                )
                
                # Column management based on selection
                if isinstance(b_df, pd.DataFrame) and isinstance(v_df, pd.DataFrame):
                    if dataset_choice == "Benchmark":
                        st.markdown("**Editing Benchmark Dataset**")
                        b_df = add_remove_column(b_df, "Benchmark")
                    elif dataset_choice == "Target":
                        st.markdown("**Editing Target Dataset**")
                        v_df = add_remove_column(v_df, "Target")
                    else:  # Both
                        st.markdown("**Editing Both Datasets**")
                        with st.expander("Common Column Operations"):
                            b_df, v_df = add_remove_common_column(b_df, v_df)
            
            with data_col:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<h4 style='font-size: 18px;'>Benchmark Data</h4>", unsafe_allow_html=True)
                    if isinstance(b_df, pd.DataFrame):
                        # Add Index if not present
                        if 'Index' not in b_df.columns:
                            b_df.insert(0, 'Index', range(1, len(b_df) + 1))
                        
                        # Ensure timestamp_seconds is present
                        b_df = ensure_seconds_column(b_df)
                        
                        # Get display columns
                        display_cols = ['Index']
                        if 'timestamp_seconds' in b_df.columns:
                            display_cols.append('timestamp_seconds')
                        
                        # For ULG files with selected assessment
                        if b_file_ext == ".ulg" and selected_assessment and selected_assessment != "None":
                            # Add columns from assessment map
                            if selected_assessment in ASSESSMENT_Y_AXIS_MAP:
                                assessment_cols = ASSESSMENT_Y_AXIS_MAP[selected_assessment]
                                display_cols.extend([col for col in assessment_cols if col in b_df.columns])
                        else:
                            # Add all numeric columns
                            numeric_cols = [col for col in b_df.columns if pd.api.types.is_numeric_dtype(b_df[col]) 
                                          and col not in display_cols]
                            display_cols.extend(numeric_cols)
                        
                        # Display DataFrame with selected columns
                        st.dataframe(
                            b_df[list(dict.fromkeys(display_cols))],  # Remove duplicates while preserving order
                            use_container_width=True,
                            height=500
                        )
                    else:
                        st.warning("⚠️ Benchmark data not properly loaded")
                
                with col2:
                    st.markdown("<h4 style='font-size: 18px;'>Target Data</h4>", unsafe_allow_html=True)
                    if isinstance(v_df, pd.DataFrame):
                        # Add Index if not present
                        if 'Index' not in v_df.columns:
                            v_df.insert(0, 'Index', range(1, len(v_df) + 1))
                        
                        # Ensure timestamp_seconds is present
                        v_df = ensure_seconds_column(v_df)
                        
                        # Get display columns
                        display_cols = ['Index']
                        if 'timestamp_seconds' in v_df.columns:
                            display_cols.append('timestamp_seconds')
                        
                        # For ULG files with selected assessment
                        if v_file_ext == ".ulg" and selected_assessment and selected_assessment != "None":
                            # Add columns from assessment map
                            if selected_assessment in ASSESSMENT_Y_AXIS_MAP:
                                assessment_cols = ASSESSMENT_Y_AXIS_MAP[selected_assessment]
                                display_cols.extend([col for col in assessment_cols if col in v_df.columns])
                        else:
                            # Add all numeric columns
                            numeric_cols = [col for col in v_df.columns if pd.api.types.is_numeric_dtype(v_df[col]) 
                                          and col not in display_cols]
                            display_cols.extend(numeric_cols)
                        
                        # Display DataFrame with selected columns
                        st.dataframe(
                            v_df[list(dict.fromkeys(display_cols))],  # Remove duplicates while preserving order
                            use_container_width=True,
                            height=500
                        )
                    else:
                        st.warning("⚠️ Target data not properly loaded")
                    
        # Plot Tab
        with tab1:
            col1, col2 = st.columns([0.2, 0.8])
            
            with col1:
                # st.markdown("<h3 style='font-size: 20px;'>🎯 Comparative Analysis</h3>", unsafe_allow_html=True)
                st.markdown("<h4 style='font-size: 18px;'>📈 Parameters</h4>", unsafe_allow_html=True)
                # Get common columns
                b_df = st.session_state.get("b_df")
                v_df = st.session_state.get("v_df")
                
                if isinstance(b_df, pd.DataFrame) and isinstance(v_df, pd.DataFrame):
                    b_numeric = get_numeric_columns(b_df)
                    v_numeric = get_numeric_columns(v_df)
                    common_cols = list(set(b_numeric) & set(v_numeric))
                    ALLOWED_X_AXIS = ["Index", "timestamp_seconds"] + [col for col in common_cols if col not in ["Index", "timestamp_seconds"]]
                    
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
                    # Set default x_axis to 'Index' unless a topic is selected
                    if 'selected_assessment' in locals() and isinstance(selected_assessment, str) and selected_assessment != "None":
                        default_x = 'timestamp_seconds' if 'timestamp_seconds' in x_axis_options else ('Index' if 'Index' in x_axis_options else x_axis_options[0])
                    else:
                        default_x = 'Index' if 'Index' in x_axis_options else x_axis_options[0]
                    # If CSV and cD2detailpeak exists, use as default y if x is Index
                    if b_df is not None and v_df is not None and hasattr(b_df, 'columns') and hasattr(v_df, 'columns'):
                        if 'cD2detailpeak' in b_df.columns and 'cD2detailpeak' in v_df.columns:
                            default_y = 'cD2detailpeak' if default_x == 'Index' else (y_axis_options[0] if y_axis_options else None)
                        else:
                            default_y = y_axis_options[0] if y_axis_options else None
                    else:
                        default_y = y_axis_options[0] if y_axis_options else None
                    
                    # More compact layout with columns
                    col_axes, col_thres = st.columns(2)
                    with col_axes:
                        x_axis = st.selectbox("X-Axis", x_axis_options, key="x_axis_select", index=x_axis_options.index(default_x))
                        y_axis = st.selectbox("Y-Axis", y_axis_options, key="y_axis_select", index=y_axis_options.index(default_y) if default_y in y_axis_options else 0)
                    with col_thres:
                        z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, key="z-slider-comparative")

                    if not x_axis or not y_axis:
                        st.info("📌 Please select valid X and Y axes to compare.")
                    else:
                        # Check if x_axis exists in DataFrame before accessing
                        if x_axis in b_df.columns:
                            col_min, col_max = st.columns(2)
                            with col_min:
                                x_min = st.number_input("X min", value=float(b_df[x_axis].min()), key="x_min_param")
                                y_min = st.number_input("Y min", value=float(b_df[y_axis].min()), key="y_min_param")
                            with col_max:
                                x_max = st.number_input("X max", value=float(b_df[x_axis].max()), key="x_max_param")
                                y_max = st.number_input("Y max", value=float(b_df[y_axis].max()), key="y_max_param")
                        else:
                            st.error(f"Selected X-axis '{x_axis}' not found in data")
                            st.stop()
                else:
                    st.info("Please upload both Benchmark and Target files to begin analysis.")
            
            with col2:
                # Get the DataFrames and ensure they are valid
                b_df = st.session_state.get("b_df")
                v_df = st.session_state.get("v_df")
                
                if x_axis and y_axis and isinstance(b_df, pd.DataFrame) and isinstance(v_df, pd.DataFrame):
                    try:
                        # Filter data
                        b_filtered = b_df[(b_df[x_axis] >= x_min) & (b_df[x_axis] <= x_max) &
                                        (b_df[y_axis] >= y_min) & (b_df[y_axis] <= y_max)]
                        v_filtered = v_df[(v_df[x_axis] >= x_min) & (v_df[x_axis] <= x_max) &
                                        (v_df[y_axis] >= y_min) & (v_df[y_axis] <= y_max)]

                        if x_axis == 'timestamp_seconds':
                            b_filtered, v_filtered, common_time = resample_to_common_time(b_filtered, v_filtered)

                        # Calculate relative difference and combined statistics
                        merged = pd.DataFrame()
                        merged['benchmark'] = b_filtered[y_axis]
                        merged['target'] = v_filtered[y_axis]
                        
                        # Calculate absolute and relative differences
                        merged['abs_diff'] = abs(merged['target'] - merged['benchmark'])
                        merged['rel_diff'] = merged['abs_diff'] / (abs(merged['benchmark']) + 1e-10)
                        
                        # Calculate rolling statistics
                        window = min(50, max(20, len(merged) // 10))
                        merged['rolling_mean'] = merged['abs_diff'].rolling(window=window, center=True).mean()
                        merged['rolling_std'] = merged['abs_diff'].rolling(window=window, center=True).std()
                        
                        # Calculate metrics
                        rmse = np.sqrt(np.mean((merged['target'] - merged['benchmark']) ** 2))
                        bench_range = merged['benchmark'].max() - merged['benchmark'].min()
                        similarity = 1 - (rmse / bench_range) if bench_range != 0 else (1.0 if rmse == 0 else 0.0)
                        similarity_index = similarity * 100

                        # Unified abnormal points detection
                        merged["Difference"] = merged['target'] - merged['benchmark']
                        merged["Z_Score"] = (merged["Difference"] - merged["Difference"].mean()) / merged["Difference"].std()
                        abnormal_mask = abs(merged["Z_Score"]) > z_threshold
                        abnormal_points = v_filtered[abnormal_mask]
                        abnormal_count = int(abnormal_mask.sum())
                        
                        # Display metrics
                        st.markdown("<h3 style='font-size: 20px;'>🎯 Metrics</h3>", unsafe_allow_html=True)
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
                        
                        # Add Plot Visualization header and plot mode selection in the same line
                        col_viz1, col_viz2 = st.columns([0.7, 0.3])
                        with col_viz1:
                            st.markdown("<h3 style='font-size: 20px;'>🧮 Plot Visualization</h3>", unsafe_allow_html=True)
                        with col_viz2:
                            plot_mode = st.radio("Plot Mode", ["Superimposed", "Separate"], horizontal=True, key="comparative_plot_mode")
                        plot_container = st.container()
                        with plot_container:
                            try:
                                if plot_mode == "Superimposed":
                                    fig = go.Figure()
                                    # Add main line plot
                                    fig.add_trace(go.Scatter(
                                        x=b_filtered[x_axis],
                                        y=b_filtered[y_axis],
                                        mode='lines',
                                        name='Benchmark'
                                    ))
                                    # Add validation data (Target) in green
                                    fig.add_trace(go.Scatter(
                                        x=v_filtered[x_axis],
                                        y=v_filtered[y_axis],
                                        mode='lines',
                                        name='Target',
                                        line=dict(color='green')
                                    ))
                                    # Add abnormal points
                                    if not abnormal_points.empty:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=abnormal_points[x_axis], 
                                                y=abnormal_points[y_axis], 
                                                mode='markers', 
                                                marker=dict(color='red', size=8), 
                                                name='Abnormal Points'
                                            )
                                        )
                                    
                                    # Get timestamp ticks if needed
                                    if x_axis == 'timestamp_seconds':
                                        tick_vals, tick_texts = get_timestamp_ticks(b_filtered[x_axis])
                                        fig.update_xaxes(
                                            tickvals=tick_vals,
                                            ticktext=tick_texts,
                                            title_text=get_axis_title(x_axis),
                                            type='linear'
                                        )
                                    else:
                                        fig.update_xaxes(title_text=x_axis)
                                        
                                    fig.update_layout(
                                        height=450,  # Reduced height to fit viewport better
                                        showlegend=True,
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.02,
                                            xanchor="center",
                                            x=0.5
                                        ),
                                        margin=dict(t=30, b=20, l=50, r=20)  # Optimized margins
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:  # Separate plots
                                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                                      subplot_titles=("Benchmark", "Target"),
                                                      vertical_spacing=0.1)  # Reduced spacing
                                    
                                    fig.add_trace(go.Scatter(
                                        x=b_filtered[x_axis],
                                        y=b_filtered[y_axis],
                                        mode='lines',
                                        name='Benchmark',
                                        line=dict(color='blue')
                                    ), row=1, col=1)
                                    
                                    fig.add_trace(go.Scatter(
                                        x=v_filtered[x_axis],
                                        y=v_filtered[y_axis],
                                        mode='lines',
                                        name='Target',
                                        line=dict(color='green')
                                    ), row=2, col=1)
                                    
                                    # Add abnormal points
                                    if not abnormal_points.empty:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=abnormal_points[x_axis],
                                                y=abnormal_points[y_axis],
                                                mode='markers',
                                                marker=dict(color='red', size=8),
                                                name='Abnormal Points'
                                            ), row=2, col=1
                                        )
                                    
                                    # Get timestamp ticks if needed
                                    if x_axis == 'timestamp_seconds':
                                        tick_vals, tick_texts = get_timestamp_ticks(b_filtered[x_axis])
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
                                    else:
                                        fig.update_xaxes(title_text=x_axis, row=1, col=1)
                                        fig.update_xaxes(title_text=x_axis, row=2, col=1)
                                    
                                    fig.update_layout(
                                        height=450,  # Reduced height to fit viewport better
                                        showlegend=True,
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.02,
                                            xanchor="center",
                                            x=0.5
                                        ),
                                        margin=dict(t=30, b=20, l=50, r=20)  # Optimized margins
                                    )
                                    fig.update_yaxes(title_text=y_axis, row=1, col=1)
                                    fig.update_yaxes(title_text=y_axis, row=2, col=1)
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error during plotting: {str(e)}")
                    except Exception as e:
                        st.error(f"Error during plotting: {str(e)}")
    else:
        # Show appropriate guidance message
        if selected_bench != "None" and selected_val != "None" and b_file_ext == ".ulg" and v_file_ext == ".ulg":
            st.info("📋 Select a topic")
        elif b_df is None and v_df is None:
            st.warning("⚠️ Please upload and select both Benchmark and Target files to begin analysis")
        elif b_df is None:
            st.warning("⚠️ Please upload and select a Benchmark file to continue")
        else:
            st.warning("⚠️ Please upload and select a Target file to continue")
