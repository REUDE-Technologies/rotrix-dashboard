import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
from io import StringIO, BytesIO
from PIL import Image
import base64
import tempfile
import os
import requests

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
        if "tree" in url:  # Folder URL
            parts = url.split("tree/")
            if len(parts) < 2:
                return None
            base_url = parts[0].rstrip('/')
            path = parts[1].lstrip('/').replace('%20', ' ')  # Handle spaces and remove leading slash
            repo = path.split('/')[0] + '/' + path.split('/')[1]
            folder_path = '/'.join(path.split('/')[2:])  # Get the full folder path including subfolders
            api_url = f"https://api.github.com/repos/{repo}/contents/{folder_path}"
            st.write(f"Debug: API URL: {api_url}")  # Debug output
            response = requests.get(api_url, headers={"Accept": "application/vnd.github.v3+json"})
            if response.status_code == 200:
                files = [item for item in response.json() if item['type'] == 'file' and item['name'].endswith(('.csv', '.pcd'))]
                file_data = {}
                for file in files:
                    file_response = requests.get(file['download_url'])
                    if file_response.status_code == 200:
                        file_ext = os.path.splitext(file['name'])[-1].lower()
                        file_data[file['name']] = (file_response.content, file_ext)
                return file_data if file_data else None
            else:
                st.error(f"Failed to fetch folder contents. Status code: {response.status_code}, API URL: {api_url}")
                return None
        elif "/blob/" in url:  # Blob URL
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
        elif "raw.githubusercontent.com" in url:  # Raw file URL
            file_name = url.split("/")[-1]
            file_ext = os.path.splitext(file_name)[-1].lower()
            if file_ext in [".csv", ".pcd"]:
                response = requests.get(url)
                if response.status_code == 200:
                    return {file_name: (response.content, file_ext)}
                else:
                    st.error(f"Failed to download file. Status code: {response.status_code}, URL: {url}")
                    return None
    return None

st.markdown("<h3 style='color:#c71585;'>🚀 Data Visionary</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='font-size:20px; color:#0068c9;'>🔼 Upload Benchmark & Target Files</h4>", unsafe_allow_html=True)

# URL Input Handling
url = st.text_input("Enter GitHub Raw File or Folder URL or Local Path")
if url:
    result = process_url(url)
    if result:
        # Store fetched files in session_state for selection
        if isinstance(list(result.values())[0], dict):  # Folder case (should not happen with current logic)
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

# File Upload Handling
benchmark_files = st.file_uploader("📂 Upload Benchmark File", type=["csv", "pcd", "ulg"], accept_multiple_files=True)
benchmark_names = [f.name for f in benchmark_files] if benchmark_files else list(st.session_state.get('uploaded_files', {}).keys())

if benchmark_files or st.session_state.get('uploaded_files'):
    selected_bench = st.selectbox("Select Abirami File", ["None"] + benchmark_names)
    if selected_bench != "None":
        if selected_bench in st.session_state:
            st.session_state.b_df = st.session_state[selected_bench]
        elif benchmark_files:
            b_file = benchmark_files[benchmark_names.index(selected_bench)]
            b_file_ext = os.path.splitext(b_file.name)[-1].lower()
            st.session_state.b_df = load_data(b_file.read(), b_file_ext)

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
    
b_df = st.session_state.get("b_df")
v_df = st.session_state.get("v_df")

# Next Step Placeholder (e.g., display selected data)
if b_df is not None or v_df is not None:
    st.markdown("<h4 style='color:#0068c9;'>Next Step: Analyze Data</h4>", unsafe_allow_html=True)
    if b_df is not None:
        st.write("Abirami File Data:", b_df)
    if v_df is not None:
        st.write("Keerthi File Data:", v_df)
