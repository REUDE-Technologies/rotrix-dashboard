# Modified script to support benchmark-only mode or benchmark+validation mode with all functionality toggled accordingly

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
import tempfile
import os


st.set_page_config(page_title="Point Cloud Data Dashboard", layout="wide")


logo_base64 = get_base64_image("Rotrix-Logo.png")
# st.logo(logo_base64, *, size="medium", link=None, icon_image=None)
st.markdown(f"""
    <div style="display: flex; position: fixed; top:50px; left: 50px; z-index:50; justify-content: left; align-items: center; padding: 1px; background-color:white; border-radius:25px;">
        <a href="http://rotrixdemo.reude.tech/" target="_blank">
            <img src="data:image/png;base64,{logo_base64}" width="180" alt="Rotrix Logo">
        </a>
    </div>
""", unsafe_allow_html=True)

# Load logic
def load_data(file, filetype, key_suffix):
    if filetype == ".csv":
        df_csv = load_csv(file)
        return df_csv
    elif filetype == ".pcd":
        df_pcd = load_pcd(file)
    return df_pcd

# Loaders
def load_csv(file):
    file.seek(0)
    return pd.read_csv(StringIO(file.read().decode("utf-8")))

def load_pcd(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcd") as tmp:
        tmp.write(file.read())
        filepath = tmp.name
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
        data_array = np.array([list(map(float, line.split())) for line in data_lines])
        df = pd.DataFrame(data_array[:, :3], columns=['X', 'Y', 'Z'])
        
#         pcd = o3d.io.read_point_cloud(tmp.name, format='xyz')
#         points = np.asarray(pcd.points)
#         df = pd.DataFrame(points, columns=["X", "Y", "Z"])
#         if len(np.asarray(pcd.colors)) > 0:
#             df["Temperature"] = np.mean(np.asarray(pcd.colors), axis=1)
    return df


st.markdown("### 🚀 Data Visualization")

# st.markdown("#### 🔼 Upload Benchmark & Validation Files")
st.markdown("<h4 style='font-size:20px; color:#0068c9;'>🔼 Upload Benchmark & Target Files</h4>", unsafe_allow_html=True)

# Simulate a topbar with two upload sections
top_col1, top_col3, top_col4 = st.columns(3)

with top_col1:
    benchmark_files = st.file_uploader("📂 Upload Benchmark File", type=["csv", "pcd", "ulg"], accept_multiple_files=True)
    benchmark_names =  [f.name for f in benchmark_files]
    
# with top_col2:
#     validation_files = st.file_uploader("📂 Upload Target File", type=["csv", "pcd", "ulg"], accept_multiple_files=True)
#     validation_names =  [f.name for f in validation_files]
    
with top_col3:
    b_df = None
    if benchmark_files:
        selected_bench = st.selectbox("Select Abirami File", ["None"] + benchmark_names)
        
        if selected_bench != "None":
            b_file = benchmark_files[benchmark_names.index(selected_bench)]
            b_file_ext = os.path.splitext(b_file.name)[-1].lower()
            st.session_state.b_df = load_data(b_file, b_file_ext, key_suffix="bench")

with top_col4:
    v_df = None
    if benchmark_files:
        selected_val = st.selectbox("Select Keerthi File", ["None"] + benchmark_names)
     
        if selected_val != "None":
            v_file = benchmark_files[benchmark_names.index(selected_val)]
            v_file_ext = os.path.splitext(v_file.name)[-1].lower()
            st.session_state.v_df = load_data(v_file, v_file_ext, key_suffix="val")

if "b_df" not in st.session_state:
    st.session_state.b_df = None
if "v_df" not in st.session_state:
    st.session_state.v_df = None
    
b_df = st.session_state.get("b_df")
v_df = st.session_state.get("v_df")




