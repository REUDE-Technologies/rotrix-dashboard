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
if 'previous_data_source' not in st.session_state:
    st.session_state.previous_data_source = None
if 'b_df' not in st.session_state:
    st.session_state.b_df = None
if 'v_df' not in st.session_state:
    st.session_state.v_df = None
if 'previous_analysis_type' not in st.session_state:
    st.session_state.previous_analysis_type = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'files_submitted' not in st.session_state:
    st.session_state.files_submitted = False
if 'show_upload_area' not in st.session_state:
    st.session_state.show_upload_area = True
if 'upload_opened_by_plus' not in st.session_state:
    st.session_state.upload_opened_by_plus = False
# Add new session state variables for file selections
if 'single_file_selection' not in st.session_state:
    st.session_state.single_file_selection = "None"
if 'benchmark_file_selection' not in st.session_state:
    st.session_state.benchmark_file_selection = "None"
if 'target_file_selection' not in st.session_state:
    st.session_state.target_file_selection = "None"

# Initialize global variables
b_df = None
v_df = None
selected_assessment = "None"
selected_bench = "None"  # Initialize selected_bench
selected_val = "None"    # Initialize selected_val

# Function to change page
def change_page(page):
    if page == 'home':
        # Store the current analysis type and data source before going back
        st.session_state.previous_analysis_type = st.session_state.analysis_type
        st.session_state.previous_data_source = st.session_state.data_source
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
    try:
        if filetype == ".csv":
            df_csv = load_csv(file)
            if df_csv is not None and not df_csv.empty:
                df_csv = convert_timestamps_to_seconds(df_csv)
                df_csv = ensure_seconds_column(df_csv)
            return df_csv, None
        elif filetype == ".ulg":
            df_ulog, topic_names = load_ulog(file, key_suffix)
            if isinstance(df_ulog, dict):
                # Handle each dataframe in the dictionary
                for topic in df_ulog:
                    if df_ulog[topic] is not None and not df_ulog[topic].empty:
                        df_ulog[topic] = convert_timestamps_to_seconds(df_ulog[topic])
                        df_ulog[topic] = ensure_seconds_column(df_ulog[topic])
            return df_ulog, topic_names
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
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
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
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
    
    # New Column Section with columns
    st.markdown("<p style='font-size: 12px; margin: 0;'>🧮 New Column</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        new_col_name = st.text_input("Column Name", key=f"{df_name}_add", label_visibility="collapsed", placeholder="Column Name")
    with col2:
        custom_formula = st.text_input("Formula", key=f"{df_name}_formula", label_visibility="collapsed", placeholder="Formula (e.g., x*y)")

    if st.button("Add Column", key=f"add_btn_{df_name}", use_container_width=True):
        try:
            if new_col_name and custom_formula:
                target_df[new_col_name] = target_df.eval(custom_formula)
                st.success(f"Added: {new_col_name}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Remove Column Section
    st.markdown("<p style='font-size: 12px; margin: 0;'>🗑 Remove Column</p>", unsafe_allow_html=True)
    columns_to_drop = st.multiselect("Select columns", target_df.columns, key=f"{df_name}_drop", label_visibility="collapsed")
    if st.button("Remove Selected", key=f"remove_btn_{df_name}", use_container_width=True):
        if columns_to_drop:
            target_df.drop(columns=columns_to_drop, inplace=True)
            st.success(f"Removed {len(columns_to_drop)} column(s)")
            
    # Rename Column Section
    st.markdown("<p style='font-size: 12px; margin: 0;'>✏ Rename Column</p>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        rename_col = st.selectbox("Select column", target_df.columns, key=f"{df_name}_rename_col", label_visibility="collapsed")
    with col4:
        new_name = st.text_input("New name", key=f"{df_name}_rename_input", label_visibility="collapsed", placeholder="New name")

    if st.button("Rename Column", key=f"rename_btn_{df_name}", use_container_width=True):
        if rename_col and new_name:
            target_df.rename(columns={rename_col: new_name}, inplace=True)
            st.success(f"Renamed: {rename_col} → {new_name}")

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
    # Add custom CSS for fixed header
    st.markdown("""
    <style>
    .fixed-header {
        position: fixed;
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
        max-width: 350px;
        border: 1px solid #e0e0e0;
    }
    .fixed-header h1 {
        color: #2E86C1;
        margin: 0 0 2px 0;
        font-size: 1.35rem;
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

    # Fixed header with improved structure
    st.markdown("""
    <div class="fixed-header">
        <h1><span class="rocket-icon">🚀</span> Data Assessment</h1>
        <!-- <p>Advanced data analysis and visualization platform</p> -->
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for file submission if not exists
    if 'files_submitted' not in st.session_state:
        st.session_state.files_submitted = False
    
    # --- Centered layout: left spacer, upload, icons, right spacer ---
    show_upload = st.session_state.show_upload_area or not st.session_state.files_submitted
    if show_upload:
        col_left, col_upload, col_icons, col_right = st.columns([4, 3, 1, 3])
        with col_upload:
            st.markdown("<h4 style='font-size:18px; color:#4B8BBE; text-align:center;'>🔼 Upload Files</h4>", unsafe_allow_html=True)
            uploaded_files = st.file_uploader("Upload your data files", type=["csv", "ulg"], key="uploader", label_visibility="collapsed", accept_multiple_files=True)
            if not st.session_state.uploaded_files:
                st.info("📂 Please upload your data files to begin analysis")
            if uploaded_files:
                new_files_added = False
                existing_names = [f.name for f in st.session_state.uploaded_files]
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in existing_names:
                        st.session_state.uploaded_files.append(uploaded_file)
                        new_files_added = True
                if new_files_added:
                    st.rerun()
            # Display uploaded files
            for i, file in enumerate(st.session_state.uploaded_files):
                col_name, col_remove = st.columns([11, 1])
                with col_name:
                    st.markdown(f"📎 {file.name}")
                with col_remove:
                    if st.button("🗑️", key=f"remove_{i}", help="Remove file"):
                        st.session_state.uploaded_files.pop(i)
                        st.rerun()
            # Submit button - show if files are uploaded (regardless of how upload area was opened)
            if st.session_state.uploaded_files:
                if st.button("Submit Files", type="primary", use_container_width=True):
                    st.session_state.files_submitted = True
                    st.session_state.show_upload_area = False
                    st.session_state.upload_opened_by_plus = False
                    st.rerun()
            # Show a 'Back' button when the upload area was opened via the plus button AND no files are present
            elif st.session_state.upload_opened_by_plus and not st.session_state.uploaded_files:
                st.markdown("""
                <style>
                .stButton>button#close_upload_area_btn {
                    background-color: #ff5252;
                    color: white;
                    width: 100%;
                    border-radius: 10px;
                    font-size: 1.1rem;
                    height: 48px;
                    margin-top: 8px;
                }
                </style>
                """, unsafe_allow_html=True)
                if st.button("Back", key="close_upload_area_btn"):
                    st.session_state.show_upload_area = False
                    st.session_state.upload_opened_by_plus = False
                    st.rerun()
        with col_icons:
            st.markdown("""
            <style>
            div[data-testid=\"column\"]:nth-of-type(3) button {
                border-radius: 50% !important;
                width: 44px !important;
                height: 44px !important;
                font-size: 1.3em !important;
                margin-bottom: 8px !important;
                margin-top: 0 !important;
                background: #f4f4f4;
                border: none;
                box-shadow: 0 2px 8px rgba(44,62,80,0.07);
                transition: background 0.2s, box-shadow 0.2s, transform 0.1s;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            div[data-testid=\"column\"]:nth-of-type(3) button:hover {
                background: #e0e7ef;
                box-shadow: 0 4px 16px rgba(44,62,80,0.13);
                transform: translateY(-2px) scale(1.04);
            }
            div[data-testid=\"column\"]:nth-of-type(3) button:last-child {
                margin-bottom: 0 !important;
            }
            div[data-testid=\"column\"]:nth-of-type(3) button:first-child {
                margin-top: 8px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            st.button("👤", key="rotrix_icon", help="Select files from ROTRIX Account")
            st.button("📁", key="shared_icon", help="Select files from My Shared Files")
    elif st.session_state.files_submitted:
        # Show a button to reveal the upload area again
        if st.button("Manage Files", type="secondary"):
            st.session_state.show_upload_area = True
            st.rerun()
    # --- End centered layout ---

    # Analysis Type Selection - only show after files are submitted
    if st.session_state.files_submitted:
        # Analysis type selection with plus button in the same row
        col1, col2, col3 = st.columns([6, 2, 0.5])
        with col1:
            analysis_type = st.radio(
                "Choose the type of analysis you want to perform",
                ["Single File Analysis", "Comparative Analysis"],
                index=0,
                horizontal=True
            )
        with col3:
            if st.button("➕", key="show_upload_area_btn", help="Add more files"):
                st.session_state.show_upload_area = True
                st.session_state.upload_opened_by_plus = True
                st.rerun()
        st.session_state.analysis_type = analysis_type

        # Reset variables when switching analysis type
        if 'previous_analysis_type' in st.session_state and st.session_state.previous_analysis_type != analysis_type:
            if analysis_type == "Single File Analysis":
                # Clear comparative analysis variables
                st.session_state.pop('selected_bench', None)
                st.session_state.pop('selected_val', None)
                st.session_state.pop('selected_bench_content', None)
                st.session_state.pop('selected_val_content', None)
                st.session_state.pop('b_df', None)
                st.session_state.pop('v_df', None)
            elif analysis_type == "Comparative Analysis":
                # Clear single file analysis variables
                st.session_state.pop('selected_single_file', None)
                st.session_state.pop('selected_single_file_content', None)
        st.session_state.previous_analysis_type = analysis_type

        if analysis_type == "Single File Analysis":
            # Single File Analysis
            col1, col2 = st.columns([1, 1])
            with col1:
                file_options = ["None"] + [f.name for f in st.session_state.uploaded_files]
                selected_file = st.selectbox(
                    "Select File", 
                    file_options,
                    key="file_selector",
                    index=file_options.index(st.session_state.single_file_selection) if st.session_state.single_file_selection in file_options else 0
                )
                st.session_state.single_file_selection = selected_file if selected_file != "None" else "None"
                st.session_state.selected_single_file = selected_file if selected_file != "None" else None
                if selected_file != "None" and st.session_state.uploaded_files:
                    try:
                        file = [f for f in st.session_state.uploaded_files if f.name == selected_file][0]
                        file.seek(0)
                        st.session_state.selected_single_file_content = file.read()
                        file.seek(0)
                    except Exception as e:
                        st.error("Error loading file")
            with col2:
                if selected_file != "None" and isinstance(selected_file, str):
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
                            "Select Topic", 
                            options=assessment_names,
                            index=default_index
                        )
                        
                        # Update session state with new topic selection
                        if selected_assessment != st.session_state.get('selected_assessment'):
                            st.session_state.selected_assessment = selected_assessment
        else:
            # Comparative Analysis
            col1, col2, col3 = st.columns([1, 1, 1])
            
            # Initialize variables
            benchmark_file = "None"
            target_file = "None"
            selected_assessment = st.session_state.get('selected_assessment', "None")
            
            # --- Comparative Analysis File Selection Row ---
            col1, col_swap, col2, col_topic = st.columns([5, .4, 5, 3])

            with col1:
                file_options = ["None"] + [f.name for f in st.session_state.uploaded_files if f.name != st.session_state.target_file_selection]
                benchmark_file = st.selectbox(
                    "Select Benchmark File", 
                    file_options,
                    key="benchmark_selector",
                    index=file_options.index(st.session_state.benchmark_file_selection) if st.session_state.benchmark_file_selection in file_options else 0
                )
                st.session_state.benchmark_file_selection = benchmark_file if benchmark_file != "None" else "None"
                if benchmark_file != "None":
                    try:
                        file = [f for f in st.session_state.uploaded_files if f.name == benchmark_file][0]
                        file.seek(0)
                        st.session_state.selected_bench_content = file.read()
                        file.seek(0)
                        st.session_state.selected_bench = benchmark_file
                    except Exception as e:
                        st.error("Error loading benchmark file")

            with col_swap:
                st.markdown("<br>", unsafe_allow_html=True)  # vertical align
                if st.button("⇄", key="swap_files", help="Swap Benchmark and Target"):
                    # Swap the selections
                    temp = st.session_state.benchmark_file_selection
                    st.session_state.benchmark_file_selection = st.session_state.target_file_selection
                    st.session_state.target_file_selection = temp
                    # Swap file contents if used
                    temp_content = st.session_state.get('selected_bench_content')
                    st.session_state['selected_bench_content'] = st.session_state.get('selected_val_content')
                    st.session_state['selected_val_content'] = temp_content
                    # Swap file names
                    temp_name = st.session_state.get('selected_bench')
                    st.session_state['selected_bench'] = st.session_state.get('selected_val')
                    st.session_state['selected_val'] = temp_name
                    # Swap topic if present
                    if 'selected_assessment' in st.session_state:
                        temp_topic = st.session_state.get('selected_assessment')
                        st.session_state['selected_assessment'] = st.session_state.get('selected_assessment')
                    st.rerun()

            with col2:
                file_options = ["None"] + [f.name for f in st.session_state.uploaded_files if f.name != st.session_state.benchmark_file_selection]
                target_file = st.selectbox(
                    "Select Target File", 
                    file_options,
                    key="target_selector",
                    index=file_options.index(st.session_state.target_file_selection) if st.session_state.target_file_selection in file_options else 0
                )
                st.session_state.target_file_selection = target_file if target_file != "None" else "None"
                if target_file != "None":
                    try:
                        file = [f for f in st.session_state.uploaded_files if f.name == target_file][0]
                        file.seek(0)
                        st.session_state.selected_val_content = file.read()
                        file.seek(0)
                        st.session_state.selected_val = target_file
                    except Exception as e:
                        st.error("Error loading target file")

            # If both files are .ulg, show topic selection in the same row
            if benchmark_file != "None" and target_file != "None" and \
               isinstance(benchmark_file, str) and isinstance(target_file, str) and \
               os.path.splitext(benchmark_file)[-1].lower() == ".ulg" and \
               os.path.splitext(target_file)[-1].lower() == ".ulg":
                with col_topic:
                    assessment_names = ["None"] + [a for _, a in TOPIC_ASSESSMENT_PAIRS]
                    assessment_to_topic = {a: t for t, a in TOPIC_ASSESSMENT_PAIRS}
                    default_assessment = st.session_state.get('selected_assessment')
                    default_index = 0  # Default to "None"
                    if isinstance(default_assessment, str) and default_assessment in assessment_names:
                        default_index = assessment_names.index(default_assessment)
                    selected_assessment = st.selectbox(
                        "Select Topic", 
                        options=assessment_names,
                        index=default_index,
                        key="comparative_topic"
                    )
                    if selected_assessment != st.session_state.get('selected_assessment'):
                        st.session_state.selected_assessment = selected_assessment
        
        # Show analysis output directly based on selection
        if analysis_type == "Single File Analysis":
            if st.session_state.get('selected_single_file'):
                # Single File Analysis content
                
                # Initialize variables
                selected_file = st.session_state.get('selected_single_file', "None")
                selected_assessment = st.session_state.get('selected_assessment', "None")
                file_ext = None
                df = None
                
                # Process the selected file
                if selected_file != "None" and isinstance(selected_file, str):
                    file_content = st.session_state.get('selected_single_file_content')
                    if file_content:
                        try:
                            # Get file extension from selected file name
                            file_ext = os.path.splitext(selected_file)[-1].lower()
                            
                            # Create a temporary file with the content
                            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                                if isinstance(file_content, str):
                                    tmp_file.write(file_content.encode('utf-8'))
                                else:
                                    tmp_file.write(file_content)
                                tmp_file.flush()
                                
                                try:
                                    if file_ext == ".ulg":
                                        dfs, topics = load_ulog(tmp_file.name)
                                        # Get topic from session state or current selection
                                        if selected_assessment and selected_assessment != "None":
                                            selected_topic = assessment_to_topic.get(str(selected_assessment))
                                            if selected_topic and selected_topic in dfs:
                                                df = dfs[selected_topic]
                                                df = ensure_seconds_column(df)
                                                if 'Index' not in df.columns:
                                                    df.insert(0, 'Index', range(1, len(df) + 1))
                                                if 'timestamp_seconds' not in df.columns:
                                                    df['timestamp_seconds'] = df.index
                                            else:
                                                df = None  # Do not set df to dict if no topic is selected
                                        else:
                                            df = None  # Do not set df to dict if no topic is selected
                                    else:
                                        df, _ = load_data(tmp_file.name, file_ext, "")
                                        if df is not None:
                                            df = ensure_seconds_column(df)
                                            if 'Index' not in df.columns:
                                                df.insert(0, 'Index', range(1, len(df) + 1))
                                            if 'timestamp_seconds' not in df.columns:
                                                df['timestamp_seconds'] = df.index
                                except Exception as e:
                                    st.error(f"Error processing file: {str(e)}")
                                    df = None
                        except Exception as e:
                            st.error(f"Error creating temporary file: {str(e)}")
                            df = None
                        finally:
                            # Clean up temporary file
                            try:
                                os.unlink(tmp_file.name)
                            except:
                                pass
                
                # Only proceed with analysis if df is properly loaded
                if df is not None and isinstance(df, pd.DataFrame) and len(df.index) > 0:
                    tab1, tab2 = st.tabs(["📊 Plot", "📋 Data"])
                    
                    if df is not None and isinstance(df, pd.DataFrame) and len(df.index) > 0:
                        # Add Index column if it doesn't exist
                        if 'Index' not in df.columns:
                            df.insert(0, "Index", range(1, len(df) + 1))
                        
                        with tab1:
                            # --- New Layout: Main columns for plot/metrics and parameters ---
                            main_col, param_col = st.columns([0.8, 0.2])
                            with param_col:
                                # st.markdown("<h4 style='font-size: 14px; margin: 0; padding: 0;'>📈 Parameters</h4>", unsafe_allow_html=True)
                                # Get selected assessment/topic and prepare axis options
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
                                # Set default y_axis based on file type and available columns
                                if file_ext == ".csv" and 'cD2detailpeak' in allowed_y_axis:
                                    default_y = 'cD2detailpeak'
                                else:
                                    default_y = allowed_y_axis[0] if allowed_y_axis else None
                                # Parameter controls
                                with param_col:
                                    st.markdown("""
                                    <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                                        <span style='font-size: 1.2rem;'>📝</span>
                                        <span style='font-size: 1.1rem; font-weight: 600;'>Parameters</span>
                                    </div>
                                    """, unsafe_allow_html=True)

                                    x_axis = st.selectbox("X-Axis", ALLOWED_X_AXIS, key="x_axis_single", index=ALLOWED_X_AXIS.index(default_x))
                                    y_axis = st.selectbox("Y-Axis", allowed_y_axis, key="y_axis_single", index=allowed_y_axis.index(default_y) if default_y in allowed_y_axis else 0)

                                    z_row = st.columns([0.7, 0.3])
                                    with z_row[0]:
                                        z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.01, key="z-slider-single")
                                    with z_row[1]:
                                        st.markdown(f"<div style='color: #e74c3c; font-weight: 600; text-align:right;'>{z_threshold:.2f}</div>", unsafe_allow_html=True)

                                    x_min_col, x_max_col = st.columns(2)
                                    with x_min_col:
                                        x_min = st.number_input("X min", value=float(df[x_axis].min()) if x_axis else 0.0, format="%.2f", key="x_min_single", step=1.0)
                                    with x_max_col:
                                        x_max = st.number_input("X max", value=float(df[x_axis].max()) if x_axis else 1.0, format="%.2f", key="x_max_single", step=1.0)

                                    y_min_col, y_max_col = st.columns(2)
                                    with y_min_col:
                                        y_min = st.number_input("Y min", value=float(df[y_axis].min()) if y_axis else 0.0, format="%.2f", key="y_min_single", step=1.0)
                                    with y_max_col:
                                        y_max = st.number_input("Y max", value=float(df[y_axis].max()) if y_axis else 1.0, format="%.2f", key="y_max_single", step=1.0)
                            with main_col:
                                # --- Metrics Row ---
                                filtered_df = df[(df[x_axis] >= x_min) & (df[x_axis] <= x_max) & (df[y_axis] >= y_min) & (df[y_axis] <= y_max)]
                                abnormal_mask, z_scores = detect_abnormalities(filtered_df[y_axis], z_threshold) if len(filtered_df.index) > 0 else (None, None)
                                abnormal_count = int(abnormal_mask.sum()) if abnormal_mask is not None else 0
                                stats = filtered_df[y_axis].describe() if len(filtered_df.index) > 0 else None
                                metrics_cols = st.columns(3)
                                with metrics_cols[0]:
                                    if stats is not None:
                                        fig1 = go.Figure(go.Indicator(
                                            mode="gauge+number",
                                            value=float(stats['mean']),
                                            title={'text': "Mean Value"},
                                            domain={'x': [0, 1], 'y': [0, 1]},
                                            gauge={
                                                'axis': {'range': [float(stats['min']), float(stats['max'])], 'tickformat': '.2f'},
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
                                with metrics_cols[1]:
                                    if stats is not None:
                                        fig2 = go.Figure(go.Indicator(
                                            mode="gauge+number",
                                            value=float(stats['std']),
                                            title={'text': "Standard Deviation"},
                                            domain={'x': [0, 1], 'y': [0, 1]},
                                            gauge={
                                                'axis': {'range': [0, float(stats['std'] * 2)], 'tickformat': '.2f'},
                                                'bar': {'color': "orange"},
                                                'steps': [
                                                    {'range': [0, float(stats['std'])], 'color': "#d4f0ff"},
                                                    {'range': [float(stats['std']), float(stats['std'] * 2)], 'color': "#ffeaa7"}
                                                ]
                                            }
                                        ))
                                        fig2.update_layout(width=200, height=120, margin=dict(t=50, b=10))
                                        st.plotly_chart(fig2)
                                with metrics_cols[2]:
                                    if stats is not None:
                                        fig3 = go.Figure(go.Indicator(
                                            mode="gauge+number",
                                            value=abnormal_count,
                                            title={'text': "Abnormal Points"},
                                            domain={'x': [0, 1], 'y': [0, 1]},
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
                                # --- Plot Visualization ---
                                st.markdown("### 🧮 Plot Visualization")
                                plot_container = st.container()
                                with plot_container:
                                    if x_axis and y_axis and len(filtered_df.index) > 0:
                                        abnormal_points = filtered_df[abnormal_mask] if abnormal_mask is not None else pd.DataFrame()
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=filtered_df[x_axis],
                                            y=filtered_df[y_axis],
                                            mode='lines',
                                            name='Data'
                                        ))
                                        if not abnormal_points.empty:
                                            fig.add_trace(go.Scatter(
                                                x=abnormal_points[x_axis],
                                                y=abnormal_points[y_axis],
                                                mode='markers',
                                                marker=dict(color='red', size=8),
                                                name='Abnormal Points'
                                            ))
                                        mean_value = filtered_df[y_axis].mean()
                                        fig.add_hline(y=mean_value, line_dash="dash", line_color="green",
                                                    annotation_text=f"Mean: {mean_value:.2f}")
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
                                            height=450,
                                            showlegend=True,
                                            legend=dict(
                                                orientation="h",
                                                yanchor="bottom",
                                                y=1.01,
                                                xanchor="center",
                                                x=0.5
                                            ),
                                            margin=dict(t=15, b=10, l=50, r=20),
                                            yaxis=dict(
                                                showticklabels=True,
                                                title=y_axis
                                            )
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                
                        with tab2:
                            # Create a 20-80 split layout
                            data_col, settings_col = st.columns([0.8, 0.2])
                            with settings_col:
                                # st.markdown("<h3 style='font-size: 20px;'>📋 Data Preview</h3>", unsafe_allow_html=True)
                                st.markdown("<h5 style='font-size: 14px; margin-bottom: 5px;'>🔧 Column Management</h5>", unsafe_allow_html=True)
                                if isinstance(df, pd.DataFrame):
                                    df = add_remove_column(df, "Dataset")
                            with data_col:
                                if file_ext == ".ulg" and selected_assessment and selected_assessment != "None":
                                    # If topic is selected but df is None, show warning
                                    if df is None:
                                        st.warning(f"⚠️ Topic '{selected_assessment}' not found in the file or has no data.")
                                    elif isinstance(df, pd.DataFrame) and len(df.index) > 0:
                                        display_cols = []
                                        if 'Index' in df.columns:
                                            display_cols.append('Index')
                                        if 'timestamp_seconds' in df.columns:
                                            display_cols.append('timestamp_seconds')
                                        if x_axis not in display_cols:
                                            display_cols.append(x_axis)
                                        if y_axis not in display_cols:
                                            display_cols.append(y_axis)
                                        if selected_assessment in ASSESSMENT_Y_AXIS_MAP:
                                            for col in ASSESSMENT_Y_AXIS_MAP[selected_assessment]:
                                                if col in df.columns and col not in display_cols and pd.api.types.is_numeric_dtype(df[col]):
                                                    display_cols.append(col)
                                        st.dataframe(df[display_cols], use_container_width=True, height=600)
                                    else:
                                        st.warning("⚠️ Dataset is empty or not loaded.")
                                elif isinstance(df, pd.DataFrame) and len(df.index) > 0:
                                    display_cols = []
                                    if 'Index' in df.columns:
                                        display_cols.append('Index')
                                    if 'timestamp_seconds' in df.columns:
                                        display_cols.append('timestamp_seconds')
                                    try:
                                        numeric_cols = []
                                        for col in list(df.columns):
                                            if col not in display_cols and pd.api.types.is_numeric_dtype(df[col]):
                                                numeric_cols.append(col)
                                        display_cols.extend(numeric_cols)
                                    except Exception as e:
                                        st.error(f"Error processing numeric columns: {str(e)}")
                                    st.dataframe(df[display_cols], use_container_width=True, height=600)
                                else:
                                    st.warning("⚠️ Dataset is empty or not loaded.")
                                    st.info("📋 Please upload a valid data file to begin analysis")

        else:  # Comparative Analysis
            if st.session_state.get('selected_bench') and st.session_state.get('selected_val'):
                # Initialize variables
                b_df = None
                v_df = None
                b_file_ext = None
                v_file_ext = None
                b_dfs = {}
                v_dfs = {}
                selected_bench = st.session_state.get('selected_bench', "None")
                selected_val = st.session_state.get('selected_val', "None")
                selected_assessment = st.session_state.get('selected_assessment', "None")

                # Initialize benchmark variables
                selected_bench = st.session_state.get('selected_bench', "None")
                b_content = st.session_state.get('selected_bench_content')
                
                # Process benchmark file
                if selected_bench != "None" and b_content:
                    b_file_ext = None
                    tmp_file = None
                    try:
                        b_file_ext = os.path.splitext(selected_bench)[-1].lower()
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
                    except Exception as e:
                        st.error(f"Error processing benchmark file: {str(e)}")
                    finally:
                        if tmp_file:
                            try:
                                os.unlink(tmp_file.name)
                            except:
                                pass

    # Initialize validation variables
    selected_val = st.session_state.get('selected_val', "None")
    v_content = st.session_state.get('selected_val_content')
    
    # Process validation file
    if selected_val != "None" and v_content:
        v_file_ext = None
        tmp_file = None
        try:
            v_file_ext = os.path.splitext(selected_val)[-1].lower()
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
        except Exception as e:
            st.error(f"Error processing target file: {str(e)}")
        finally:
            if tmp_file:
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass
            
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
    if st.session_state.get('selected_bench') and st.session_state.get('selected_val'):
        tab1, tab2 = st.tabs(["📊 Plot", "📋 Data"])
        
        # Data Tab
        with tab2:
            # Create a 20-80 split layout
            data_col, settings_col = st.columns([0.8, 0.2])
            
            with settings_col:
                # Add dataset selector and column management
                # st.markdown("<h3 style='font-size: 20px;'>📋 Data Preview</h3>", unsafe_allow_html=True)
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
                            height=600
                        )
                    elif isinstance(b_df, dict):
                        # Handle dictionary of DataFrames (ULog case)
                        if selected_assessment and selected_assessment != "None":
                            selected_topic = assessment_to_topic.get(str(selected_assessment))
                            if selected_topic and selected_topic in b_df:
                                df_to_display = b_df[selected_topic]
                                if isinstance(df_to_display, pd.DataFrame):
                                    # Add Index if not present
                                    if 'Index' not in df_to_display.columns:
                                        df_to_display.insert(0, 'Index', range(1, len(df_to_display) + 1))
                                    
                                    # Ensure timestamp_seconds is present
                                    df_to_display = ensure_seconds_column(df_to_display)
                                    
                                    # Get display columns
                                    display_cols = ['Index']
                                    if 'timestamp_seconds' in df_to_display.columns:
                                        display_cols.append('timestamp_seconds')
                                    
                                    # Add columns from assessment map
                                    if selected_assessment in ASSESSMENT_Y_AXIS_MAP:
                                        assessment_cols = ASSESSMENT_Y_AXIS_MAP[selected_assessment]
                                        display_cols.extend([col for col in assessment_cols if col in df_to_display.columns])
                                    
                                    # Display DataFrame with selected columns
                                    st.dataframe(
                                        df_to_display[list(dict.fromkeys(display_cols))],
                                        use_container_width=True,
                                        height=600
                                    )
                                else:
                                    st.warning("⚠️ Selected topic data is not in the correct format")
                            else:
                                st.warning("⚠️ Selected topic not found in the data")
                        else:
                            st.info("📋 Please select a topic to view the data")
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
                            height=600
                        )
                    elif isinstance(v_df, dict):
                        # Handle dictionary of DataFrames (ULog case)
                        if selected_assessment and selected_assessment != "None":
                            selected_topic = assessment_to_topic.get(str(selected_assessment))
                            if selected_topic and selected_topic in v_df:
                                df_to_display = v_df[selected_topic]
                                if isinstance(df_to_display, pd.DataFrame):
                                    # Add Index if not present
                                    if 'Index' not in df_to_display.columns:
                                        df_to_display.insert(0, 'Index', range(1, len(df_to_display) + 1))
                                    
                                    # Ensure timestamp_seconds is present
                                    df_to_display = ensure_seconds_column(df_to_display)
                                    
                                    # Get display columns
                                    display_cols = ['Index']
                                    if 'timestamp_seconds' in df_to_display.columns:
                                        display_cols.append('timestamp_seconds')
                                    
                                    # Add columns from assessment map
                                    if selected_assessment in ASSESSMENT_Y_AXIS_MAP:
                                        assessment_cols = ASSESSMENT_Y_AXIS_MAP[selected_assessment]
                                        display_cols.extend([col for col in assessment_cols if col in df_to_display.columns])
                                    
                                    # Display DataFrame with selected columns
                                    st.dataframe(
                                        df_to_display[list(dict.fromkeys(display_cols))],
                                        use_container_width=True,
                                        height=600
                                    )
                                else:
                                    st.warning("⚠️ Selected topic data is not in the correct format")
                            else:
                                st.warning("⚠️ Selected topic not found in the data")
                        else:
                            st.info("📋 Please select a topic to view the data")
                    else:
                        st.warning("⚠️ Target data not properly loaded")
                    
        # Plot Tab
        with tab1:
            # --- Metrics Row (Full Width) ---
            b_df = st.session_state.get("b_df")
            v_df = st.session_state.get("v_df")
            metrics_ready = False
            x_axis = y_axis = z_threshold = x_min = x_max = y_min = y_max = None
            if isinstance(b_df, pd.DataFrame) and isinstance(v_df, pd.DataFrame):
                b_numeric = get_numeric_columns(b_df)
                v_numeric = get_numeric_columns(v_df)
                common_cols = list(set(b_numeric) & set(v_numeric))
                ALLOWED_X_AXIS = ["Index", "timestamp_seconds"] + [col for col in common_cols if col not in ["Index", "timestamp_seconds"]]
                if 'selected_assessment' in locals() and isinstance(selected_assessment, str) and selected_assessment != "None":
                    allowed_y_axis = ASSESSMENT_Y_AXIS_MAP.get(selected_assessment, [])
                    allowed_y_axis = [col for col in allowed_y_axis if col in b_df.columns]
                    if not allowed_y_axis:
                        allowed_y_axis = list(b_df.columns)
                    allowed_y_axis = [col for col in allowed_y_axis if pd.api.types.is_numeric_dtype(b_df[col])]
                    ALLOWED_X_AXIS = ["Index", "timestamp_seconds"] + [col for col in allowed_y_axis if col not in ["Index", "timestamp_seconds"]]
                else:
                    allowed_y_axis = [col for col in b_df.columns if pd.api.types.is_numeric_dtype(b_df[col])]
                    ALLOWED_X_AXIS = ["Index", "timestamp_seconds"] + [col for col in allowed_y_axis if col not in ["Index", "timestamp_seconds"]]
                x_axis_options = ALLOWED_X_AXIS
                y_axis_options = allowed_y_axis
                if 'selected_assessment' in locals() and isinstance(selected_assessment, str) and selected_assessment != "None":
                    default_x = 'timestamp_seconds' if 'timestamp_seconds' in x_axis_options else ('Index' if 'Index' in x_axis_options else x_axis_options[0])
                else:
                    default_x = 'Index' if 'Index' in x_axis_options else x_axis_options[0]
                if b_df is not None and v_df is not None and hasattr(b_df, 'columns') and hasattr(v_df, 'columns'):
                    if 'cD2detailpeak' in b_df.columns and 'cD2detailpeak' in v_df.columns:
                        default_y = 'cD2detailpeak' if default_x == 'Index' else (y_axis_options[0] if y_axis_options else None)
                    else:
                        default_y = y_axis_options[0] if y_axis_options else None
                else:
                    default_y = y_axis_options[0] if y_axis_options else None
                # Set up parameter defaults for metrics calculation
                x_axis = default_x
                y_axis = default_y
                z_threshold = 3.0
                x_min = x_max = y_min = y_max = None
                if b_df is not None and x_axis is not None and y_axis is not None and x_axis in b_df.columns and y_axis in b_df.columns:
                    x_min = float(b_df[x_axis].min())
                    x_max = float(b_df[x_axis].max())
                    y_min = float(b_df[y_axis].min())
                    y_max = float(b_df[y_axis].max())
                    metrics_ready = True
                else:
                    metrics_ready = False
            if metrics_ready and None not in (x_min, x_max, y_min, y_max) and b_df is not None and v_df is not None and x_axis in b_df.columns and y_axis in b_df.columns and x_axis in v_df.columns and y_axis in v_df.columns:
                # Compute metrics using full data range and defaults
                b_filtered = b_df[(b_df[x_axis] >= x_min) & (b_df[x_axis] <= x_max) & (b_df[y_axis] >= y_min) & (b_df[y_axis] <= y_max)]
                v_filtered = v_df[(v_df[x_axis] >= x_min) & (v_df[x_axis] <= x_max) & (v_df[y_axis] >= y_min) & (v_df[y_axis] <= y_max)]
                if x_axis == 'timestamp_seconds':
                    b_filtered, v_filtered, _ = resample_to_common_time(b_filtered, v_filtered)
                merged = pd.DataFrame()
                merged['benchmark'] = b_filtered[y_axis]
                merged['target'] = v_filtered[y_axis]
                rmse = np.sqrt(np.mean((merged['target'] - merged['benchmark']) ** 2))
                bench_range = merged['benchmark'].max() - merged['benchmark'].min()
                similarity = 1 - (rmse / bench_range) if bench_range != 0 else (1.0 if rmse == 0 else 0.0)
                similarity_index = similarity * 100
                merged["Difference"] = merged['target'] - merged['benchmark']
                merged["Z_Score"] = (merged["Difference"] - merged["Difference"].mean()) / merged["Difference"].std()
                abnormal_mask = abs(merged["Z_Score"]) > z_threshold
                abnormal_count = int(abnormal_mask.sum())
                # --- Metrics Row ---
            metrics_col, param_col = st.columns([0.8, 0.2])
            with param_col:
                st.markdown("""
                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                    <span style='font-size: 1.2rem;'>📝</span>
                    <span style='font-size: 1.1rem; font-weight: 600;'>Parameters</span>
                </div>
                """, unsafe_allow_html=True)

                x_axis = st.selectbox("X-Axis", x_axis_options, key="x_axis_comparative", index=x_axis_options.index(str(default_x)) if isinstance(default_x, str) and default_x in x_axis_options else 0)
                y_axis = st.selectbox("Y-Axis", y_axis_options, key="y_axis_comparative", index=y_axis_options.index(str(default_y)) if isinstance(default_y, str) and default_y in y_axis_options else 0)

                z_row = st.columns([0.7, 0.3])
                with z_row[0]:
                    z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.01, key="z-slider-comparative")
                with z_row[1]:
                    st.markdown(f"<div style='color: #e74c3c; font-weight: 600; text-align:right;'>{z_threshold:.2f}</div>", unsafe_allow_html=True)

                x_min_col, x_max_col = st.columns(2)
                with x_min_col:
                    x_min = st.number_input("X min", value=float(b_df[x_axis].min()) if b_df is not None and x_axis in b_df.columns else 0.0, format="%.2f", key="x_min_comparative", step=1.0)
                with x_max_col:
                    x_max = st.number_input("X max", value=float(b_df[x_axis].max()) if b_df is not None and x_axis in b_df.columns else 1.0, format="%.2f", key="x_max_comparative", step=1.0)

                y_min_col, y_max_col = st.columns(2)
                with y_min_col:
                    y_min = st.number_input("Y min", value=float(b_df[y_axis].min()) if b_df is not None and y_axis in b_df.columns else 0.0, format="%.2f", key="y_min_comparative", step=1.0)
                with y_max_col:
                    y_max = st.number_input("Y max", value=float(b_df[y_axis].max()) if b_df is not None and y_axis in b_df.columns else 1.0, format="%.2f", key="y_max_comparative", step=1.0)
            with metrics_col:
                metrics_cols = st.columns(3)
                with metrics_cols[0]:
                    fig1 = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=rmse,
                        title={'text': "RMSE"},
                        number={'valueformat': ',.2f'},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, max(rmse * 2, 1)], 'tickformat': ',.2f'},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, rmse], 'color': "lightgray"},
                                {'range': [rmse, max(rmse * 2, 1)], 'color': "gray"}
                            ]
                        }
                    ))
                    fig1.update_layout(width=200, height=120, margin=dict(t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig1, use_container_width=False)
                with metrics_cols[1]:
                    fig2 = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=similarity_index,
                        title={'text': "Similarity Index (%)"},
                        number={'valueformat': '.2f', 'suffix': '%'},
                        domain={'x': [0, 1], 'y': [0, 1]},
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
                    fig2.update_layout(width=200, height=120, margin=dict(t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig2, use_container_width=False)
                with metrics_cols[2]:
                    fig3 = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=abnormal_count,
                        title={'text': "Abnormal Points"},
                        domain={'x': [0, 1], 'y': [0, 1]},
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
                    fig3.update_layout(width=200, height=120, margin=dict(t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig3, use_container_width=False)
                # --- Main Content: Plot ---
                plot_container = st.container()
                with plot_container:
                    if not x_axis or not y_axis:
                        st.info("📊 Please select valid X and Y axes to compare")
                        st.stop()
                    if b_df is not None and hasattr(b_df, 'columns') and x_axis in b_df.columns:
                        pass
                    else:
                        st.error(f"Selected X-axis '{x_axis}' not found in data")
                        st.stop()
                    # Plot visualization (move all plot code here)
                    if x_axis and y_axis and isinstance(b_df, pd.DataFrame) and isinstance(v_df, pd.DataFrame):
                        try:
                            b_filtered = b_df[(b_df[x_axis] >= x_min) & (b_df[x_axis] <= x_max) & (b_df[y_axis] >= y_min) & (b_df[y_axis] <= y_max)]
                            v_filtered = v_df[(v_df[x_axis] >= x_min) & (v_df[x_axis] <= x_max) & (v_df[y_axis] >= y_min) & (v_df[y_axis] <= y_max)]
                            if x_axis == 'timestamp_seconds':
                                b_filtered, v_filtered, common_time = resample_to_common_time(b_filtered, v_filtered)
                            merged = pd.DataFrame()
                            merged['benchmark'] = b_filtered[y_axis]
                            merged['target'] = v_filtered[y_axis]
                            merged['abs_diff'] = abs(merged['target'] - merged['benchmark'])
                            merged['rel_diff'] = merged['abs_diff'] / (abs(merged['benchmark']) + 1e-10)
                            window = min(50, max(20, len(merged) // 10))
                            merged['rolling_mean'] = merged['abs_diff'].rolling(window=window, center=True).mean()
                            merged['rolling_std'] = merged['abs_diff'].rolling(window=window, center=True).std()
                            rmse = np.sqrt(np.mean((merged['target'] - merged['benchmark']) ** 2))
                            bench_range = merged['benchmark'].max() - merged['benchmark'].min()
                            similarity = 1 - (rmse / bench_range) if bench_range != 0 else (1.0 if rmse == 0 else 0.0)
                            similarity_index = similarity * 100
                            merged["Difference"] = merged['target'] - merged['benchmark']
                            merged["Z_Score"] = (merged["Difference"] - merged["Difference"].mean()) / merged["Difference"].std()
                            abnormal_mask = abs(merged["Z_Score"]) > z_threshold
                            abnormal_points = v_filtered[abnormal_mask]
                            abnormal_count = int(abnormal_mask.sum())
                            # --- Plot Visualization heading and Plot Mode selector in one row ---
                            heading_col, mode_col = st.columns([0.7, 0.3])
                            with heading_col:
                                st.markdown("### 🧮 Plot Visualization")
                            with mode_col:
                                plot_mode = st.radio("Plot Mode", ["Superimposed", "Separate"], horizontal=True, key="comparative_plot_mode")
                            plot_container = st.container()
                            with plot_container:
                                if plot_mode == "Superimposed":
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=b_filtered[x_axis],
                                        y=b_filtered[y_axis],
                                        mode='lines',
                                        name='Benchmark'
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=v_filtered[x_axis],
                                        y=v_filtered[y_axis],
                                        mode='lines',
                                        name='Target',
                                        line=dict(color='green')
                                    ))
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
                                        height=450,
                                        showlegend=True,
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.01,
                                            xanchor="center",
                                            x=0.5
                                        ),
                                        margin=dict(t=15, b=10, l=50, r=20),
                                        plot_bgcolor='white',
                                        yaxis=dict(
                                            showticklabels=True,
                                            title=y_axis
                                        )
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:  # Separate plots
                                    fig = make_subplots(rows=2, cols=1, 
                                                    shared_xaxes=True, 
                                                    subplot_titles=None,  
                                                    vertical_spacing=0.08,  
                                                    row_heights=[0.5, 0.5])
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
                                    if x_axis == 'timestamp_seconds':
                                        tick_vals, tick_texts = get_timestamp_ticks(b_filtered[x_axis])
                                        fig.update_xaxes(
                                            tickvals=tick_vals,
                                            ticktext=tick_texts,
                                            title_text="",  
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
                                        fig.update_xaxes(title_text="", row=1, col=1)
                                        fig.update_xaxes(title_text=x_axis, row=2, col=1)
                                    fig.update_layout(
                                        height=450,
                                        showlegend=True,
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.01,
                                            xanchor="center",
                                            x=0.5
                                        ),
                                        margin=dict(t=15, b=10, l=50, r=20),
                                        plot_bgcolor='white',
                                        yaxis=dict(
                                            showticklabels=True,
                                            title=y_axis
                                        )
                                    )
                                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                                    fig.update_yaxes(title_text=y_axis, row=1, col=1)
                                    fig.update_yaxes(title_text=y_axis, row=2, col=1)
                                    st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error during plotting: {str(e)}")
    else:
        # Show appropriate guidance message
        if selected_bench != "None" and selected_val != "None" and b_file_ext == ".ulg" and v_file_ext == ".ulg":
            st.info("📋 Please select a topic to begin analysis")
        elif b_df is None and v_df is None:
            pass  # Removed warning message
        elif b_df is None:
            pass  # Removed warning message
        else:
            pass  # Removed warning message

if (st.session_state.get('analysis_type') == 'Comparative Analysis' and len(st.session_state.uploaded_files) == 1):
    st.markdown("<div style='margin-top:2em; text-align:center;'>", unsafe_allow_html=True)
    if st.button('Upload More Files', key='upload_more_files_btn', use_container_width=True):
        st.session_state.show_upload_area = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
