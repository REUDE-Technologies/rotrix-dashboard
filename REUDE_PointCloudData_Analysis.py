import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
from io import StringIO, BytesIO
from PIL import Image
import base64
import tempfile
import os
import requests

st.cache_data.clear()  # Clear cache to ensure latest code is used

st.set_page_config(page_title="Point Cloud Data Dashboard", layout="wide")

# 🔹 Logo
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("Rotrix-Logo.png")
st.markdown(f"""
    <div style="display: flex; position: fixed; top:50px; left: 50px; z-index:50; justify-content: left; align-items: center; padding: 1px; background-color:white; border-radius:25px;">
        <a href="http://rotrixdemo.reude.tech/" target="_blank">
            <img src="data:image/png;base64,{logo_base64}" width="180" alt="Rotrix Logo">
        </a>
    </div>
""", unsafe_allow_html=True)

# Load logic
def load_data(file_content, filetype):
    if filetype == ".csv":
        return pd.read_csv(StringIO(file_content.decode("utf-8")))
    elif filetype == ".pcd":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pcd") as tmp:
            tmp.write(file_content)
            filepath = tmp.name
            with open(filepath, 'r') as f:
                lines = f.readlines()
            data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
            data_array = np.array([list(map(float, line.split())) for line in data_lines])
            df = pd.DataFrame(data_array[:, :3], columns=['X', 'Y', 'Z'])
        os.unlink(filepath)  # Clean up temporary file
        return df
    return None

# Fetch files from GitHub folder or load single file
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
                # Extract repo and folder path
                url_parts = base_url.split("/")
                if len(url_parts) < 5 or url_parts[2] != "github.com":
                    st.error("Unable to parse repository from URL.")
                    return None
                repo = f"{url_parts[3]}/{url_parts[4]}"  # e.g., username/repo
                folder_path = path.split('/', 1)[-1] if '/' in path else path  # Get path after branch
                api_url = f"https://api.github.com/repos/{repo}/contents/{folder_path}"
                st.write(f"Debug: API URL: {api_url}")  # Debug output
                response = requests.get(api_url, headers={"Accept": "application/vnd.github.v3+json"})
                if response.status_code == 200:
                    files = [item for item in response.json() if item['type'] == 'file' and item['name'].endswith(('.csv', '.pcd'))]
                    if not files:
                        st.warning("No .csv or .pcd files found in the folder.")
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
                if file_ext in [".csv", ".pcd"]:
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
                if file_ext in [".csv", ".pcd"]:
                    response = requests.get(raw_url)
                    if response.status_code == 200:
                        return {file_name: (response.content, file_ext)}
                    else:
                        st.error(f"Failed to download file. Status code: {response.status_code}, URL: {raw_url}")
                        return None
            # Handle direct file URL (e.g., https://github.com/username/repo/filename.csv)
            elif len(url.split("/")) > 4 and url.split("/")[4] not in ["tree", "blob", "raw"]:
                # Assume it's a file URL and convert to raw
                base_parts = url.split("/")
                repo = f"{base_parts[3]}/{base_parts[4]}"
                file_path = "/".join(base_parts[5:])  # Path including file name
                raw_url = f"https://raw.githubusercontent.com/{repo}/main/{file_path}"
                file_name = file_path.split("/")[-1]
                file_ext = os.path.splitext(file_name)[-1].lower()
                if file_ext in [".csv", ".pcd"]:
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

st.markdown("<h3 style='color:#c71585;'>🚀 Data Visionary</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='font-size:20px; color:#0068c9;'>🔼 Upload Benchmark & Target Files</h4>", unsafe_allow_html=True)

# URL Input and File Upload in Columns
col1, col2 = st.columns(2)
with col1:
    url = st.text_input("Enter GitHub Raw File or Folder URL")
    if url:
        result = process_url(url)
        if result:
            # Store fetched files in session_state for selection
            if isinstance(list(result.values())[0], dict):  # Folder case
                st.error("Unexpected folder structure in single file case.")
            else:
                st.session_state.uploaded_files = result
                for file_name, (file_content, file_ext) in result.items():
                    df = load_data(file_content, file_ext)
                    if df is not None:
                        st.session_state[file_name] = df
                        st.success(f"{file_ext[1:].upper()} file '{file_name}' loaded successfully!")
                        # st.dataframe(df)
        else:
            st.warning("The provided path is not a valid GitHub folder or raw file URL. Please upload files manually via drag and drop below.")

with col2:
    benchmark_files = st.file_uploader("📂 Upload Benchmark File", type=["csv", ".pcd"], accept_multiple_files=True)
    benchmark_names = [f.name for f in benchmark_files] if benchmark_files else list(st.session_state.get('uploaded_files', {}).keys())

# File Selection in Columns
col3, col4 = st.columns(2)
with col3:
    selected_bench = st.selectbox("Select Abirami File", ["None"] + benchmark_names)
    if selected_bench != "None":
        if selected_bench in st.session_state:
            st.session_state.b_df = st.session_state[selected_bench]
        elif benchmark_files:
            b_file = benchmark_files[benchmark_names.index(selected_bench)]
            b_file_ext = os.path.splitext(b_file.name)[-1].lower()
            st.session_state.b_df = load_data(b_file.read(), b_file_ext)

with col4:
    selected_val = st.selectbox("Select Keerthi File", ["None"] + benchmark_names)
    if selected_val != "None":
        if selected_val in st.session_state:
            st.session_state.v_df = st.session_state[selected_val]
        elif benchmark_files:
            v_file = benchmark_files[benchmark_names.index(selected_val)]
            v_file_ext = os.path.splitext(v_file.name)[-1].lower()
            st.session_state.v_df = load_data(v_file.read(), v_file_ext)

if "b_df" not in st.session_state:
    st.session_state.b_df = None
if "v_df" not in st.session_state:
    st.session_state.v_df = None

# Tabs for Plot and Data
tab1, tab2 = st.tabs(["📊 Plot", "📋 Data"])
with tab2:
    st.subheader("📁 Imported Data Preview")
    b_df = st.session_state.get("b_df")
    v_df = st.session_state.get("v_df")
    if b_df is not None and v_df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🧪 Abirami")
            st.dataframe(b_df)
        with col2:
            st.markdown("### 🔬 Keerthishree")
            st.dataframe(v_df)
    elif b_df is not None:
        st.markdown("### 🧪 Abirami")
        st.dataframe(b_df)
    elif v_df is not None:
        st.markdown("### 🔬 Keerthishree")
        st.dataframe(v_df)
    else:
        st.info("No data uploaded yet.")

with tab1:
    st.subheader("🔍 Comparative Analysis")
    b_df = st.session_state.get("b_df")
    v_df = st.session_state.get("v_df")

    if b_df is not None and v_df is not None:
        # Add Index column only if it doesn't exist
        if "Index" not in b_df.columns:
            b_df.insert(0, "Index", range(1, len(b_df) + 1))
        if "Index" not in v_df.columns:
            v_df.insert(0, "Index", range(1, len(v_df) + 1))

        # Find common columns
        common_cols = list(set(b_df.columns) & set(v_df.columns))
        if common_cols:
            col1, col2 = st.columns([0.20, 0.80])
            with col1:
                st.markdown("#### 📈 Parameters")
                x_axis = st.selectbox("X-Axis", ["None"] + common_cols, key="x_axis_select")
                y_axis = st.selectbox("Y-Axis", ["None"] + common_cols, key="y_axis_select")
                sample_size = st.slider("Sample Size (for large datasets)", 1000, 100000, 10000, 1000, 
                                       help="Reduce the number of points to plot for performance. Set to a lower value for large datasets.")
                z_threshold = st.slider("Z-Score Threshold for Abnormal Points", 1.0, 5.0, 3.0, 0.1,
                                       help="Points with Z-score above this threshold will be marked as abnormal.")

                if x_axis == "None" or y_axis == "None":
                    st.info("📌 Please select a valid X-axis and Y-axis to compare.")
                else:
                    x_min = st.number_input("X min", value=float(b_df[x_axis].min()), key="x_min")
                    x_max = st.number_input("X max", value=float(b_df[x_axis].max()), key="x_max")
                    y_min = st.number_input("Y min", value=float(b_df[y_axis].min()), key="y_min")
                    y_max = st.number_input("Y max", value=float(b_df[y_axis].max()), key="y_max")

                    # Filter data
                    b_filtered = b_df[(b_df[x_axis] >= x_min) & (b_df[x_axis] <= x_max) &
                                    (b_df[y_axis] >= y_min) & (b_df[y_axis] <= y_max)]
                    v_filtered = v_df[(v_df[x_axis] >= x_min) & (v_df[x_axis] <= x_max) &
                                    (v_df[y_axis] >= y_min) & (v_df[y_axis] <= y_max)]

                    # Sample the data if too large
                    if len(b_filtered) > sample_size:
                        b_filtered = b_filtered.sample(n=sample_size, random_state=42)
                    if len(v_filtered) > sample_size:
                        v_filtered = v_filtered.sample(n=sample_size, random_state=42)

                    merged = pd.merge(b_filtered, v_filtered, on=x_axis, suffixes=('_benchmark', '_validation'), how='inner')

                    # Debug: Check merged DataFrame
                    st.write("Debug: Merged DataFrame columns:", merged.columns.tolist())
                    st.write("Debug: Merged DataFrame shape:", merged.shape)

                    # Calculate abnormal points and stats for validation data
                    if not merged.empty and f"{y_axis}_validation" in merged.columns:
                        val_col = f"{y_axis}_validation"
                        if pd.api.types.is_numeric_dtype(merged[val_col]):
                            mean_val = merged[val_col].mean()
                            std_val = merged[val_col].std()
                            z_scores = np.abs((merged[val_col] - mean_val) / std_val)
                            abnormal_mask = z_scores > z_threshold
                            abnormal_points = merged[abnormal_mask]
                        else:
                            st.warning(f"Column {val_col} contains non-numeric data. Skipping abnormality detection.")
                            abnormal_points = pd.DataFrame()
                    else:
                        st.warning("No valid data for abnormality detection.")
                        abnormal_points = pd.DataFrame()

            with col2:
                if x_axis == "None" or y_axis == "None":
                    st.info("📌 Please select a valid X-axis and Y-axis to compare.")
                elif merged.empty:
                    st.warning("No data to plot. Check filters or column selections.")
                else:
                    st.markdown("<div style='min-height: 10px'>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("### 🧮 Plot Visualization")
                    fig = make_subplots(rows=2, cols=1, subplot_titles=["Abirami", "Keerthishree"], shared_yaxes=True)

                    # Abirami plot
                    fig.add_trace(go.Scatter(x=merged[x_axis], y=merged[f"{y_axis}_benchmark"], name="Abirami", mode="lines"), row=1, col=1)

                    # Keerthishree plot with mean, std, and abnormal points
                    fig.add_trace(go.Scatter(x=merged[x_axis], y=merged[f"{y_axis}_validation"], name="Keerthishree", mode="lines"), row=2, col=1)
                    if not abnormal_points.empty:
                        fig.add_trace(go.Scatter(x=abnormal_points[x_axis], y=abnormal_points[f"{y_axis}_validation"],
                                               mode='markers', marker=dict(color='red', size=6),
                                               name="Abnormal"), row=2, col=1)

                    # Add mean line
                    if f"{y_axis}_validation" in merged.columns and pd.api.types.is_numeric_dtype(merged[f"{y_axis}_validation"]):
                        fig.add_trace(go.Scatter(x=[merged[x_axis].min(), merged[x_axis].max()],
                                               y=[mean_val, mean_val],
                                               mode='lines',
                                               line=dict(color='green', dash='dash'),
                                               name=f"Mean ({mean_val:.2f})"), row=2, col=1)

                        # Add ±1 std deviation bands
                        fig.add_trace(go.Scatter(x=[merged[x_axis].min(), merged[x_axis].max()],
                                               y=[mean_val + std_val, mean_val + std_val],
                                               mode='lines',
                                               line=dict(color='blue', dash='dot'),
                                               name=f"+1 SD ({mean_val + std_val:.2f})"), row=2, col=1)
                        fig.add_trace(go.Scatter(x=[merged[x_axis].min(), merged[x_axis].max()],
                                               y=[mean_val - std_val, mean_val - std_val],
                                               mode='lines',
                                               line=dict(color='blue', dash='dot'),
                                               name=f"-1 SD ({mean_val - std_val:.2f})"), row=2, col=1)

                    fig.update_layout(height=700, width=1000, title_text="Abirami vs Keerthishree Subplot",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No common columns to compare between Abirami and Keerthishree.")
    else:
        st.info("Please upload both Abirami and Keerthishree files or pre-converted CSVs.")

# Helper function for abnormality detection (included for reference, but implemented inline)
def detect_abnormalities(series, threshold=3.0):
    if series.empty or not pd.api.types.is_numeric_dtype(series):
        return pd.Series([False]), pd.Series([0.0])
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold, z_scores
