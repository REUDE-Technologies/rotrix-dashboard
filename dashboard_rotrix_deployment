# Modified script to support both single file analysis and comparative assessment modes

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

def load_ulog(file, key_suffix=""):
    ulog = ULog(file)
    extracted_dfs = {msg.name: pd.DataFrame(msg.data) for msg in ulog.data_list}
    topic_names = ["None"] + list(extracted_dfs.keys())
    
    if not topic_names:
        st.warning("No extractable topics found in ULOG file.")
        return pd.DataFrame()
    
    select_key = f"ulog_topic_{key_suffix}" if key_suffix else None
    selected_topic = st.selectbox("Select a topic from extracted CSVs", topic_names, key=select_key)

    df = extracted_dfs.get(selected_topic, pd.DataFrame())
    if df.empty:
        st.warning(f"Topic `{selected_topic}` has no data.")
    
    return df

def detect_trend(series):
    if series.iloc[-1] > series.iloc[0]:
        return "increasing"
    elif series.iloc[-1] < series.iloc[0]:
        return "decreasing"
    return "flat"

def detect_abnormalities(series, threshold=3.0):
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold, z_scores

# Load logic
def load_data(file, filetype, key_suffix):
    if filetype == ".csv":
        df_csv = load_csv(file)
        return df_csv
    elif filetype == ".pcd":
        df_pcd = load_pcd(file)
        return df_pcd
    elif filetype == ".ulg":
        df_ulog = load_ulog(file, key_suffix)
    return df_ulog

def add_remove_common_column(b_df, v_df):
    if b_df is None or v_df is None or b_df.empty or v_df.empty:
        st.warning("⚠️ Both Benchmark and Target data must be loaded.")
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
            st.success(f"✅ Added `{new_col['name']}` using `{new_col['formula']}` to both Benchmark and Target.")
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
                st.experimental_rerun()

    with col2:
        st.markdown("###### 🗑️ Remove Column")
        common_cols = list(set(b_df.columns) & set(v_df.columns))
        cols_to_drop = st.multiselect("Select column(s) to drop", common_cols, key="common_drop")

        if st.button("Remove Columns"):
            if cols_to_drop:
                st.session_state.b_df.drop(columns=cols_to_drop, inplace=True)
                st.session_state.v_df.drop(columns=cols_to_drop, inplace=True)
                st.success(f"🗑️ Removed columns: {', '.join(cols_to_drop)} from both Benchmark and Target.")
                st.experimental_rerun()

    return st.session_state.b_df, st.session_state.v_df


def add_remove_column(target_df, df_name="DataFrame"):
     # CREATE COLUMN
    if target_df is None or target_df.empty:
        st.warning(f"⚠️ {df_name} is empty or not loaded.")
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
                st.success(f"✅ Added column `{new_col_name}` to {selected_df} using: `{custom_formula}`")
        except Exception as e:
            st.error(f"❌ Error creating column: {e}")
#     with col2:
    # REMOVE COLUMN
    st.markdown("##### 🗑️ Remove Column")
    columns_to_drop = st.multiselect("Select columns to drop", target_df.columns, key=f"{df_name}_drop")

    if st.button(f"Remove Column to {df_name}"):
        if columns_to_drop:
            target_df.drop(columns=columns_to_drop, inplace=True)
            st.success(f"🗑️ Removed columns: {', '.join(columns_to_drop)} from {selected_df}")
            
    st.markdown("##### ✏️ Rename Column")
    rename_col = st.selectbox("Select column to rename", target_df.columns, key=f"{df_name}_rename_col")
    new_name = st.text_input("New column name", key=f"{df_name}_rename_input")

    if st.button(f"Rename Column in {df_name}", key=f"{df_name}_rename_button"):
        if rename_col and new_name:
            target_df.rename(columns={rename_col: new_name}, inplace=True)
            st.success(f"✏️ Renamed column `{rename_col}` to `{new_name}` in {df_name}")

    return target_df

# st.markdown("#### 🔼 Upload Benchmark & Validation Files")
st.markdown("<h4 style='font-size:20px; color:#FFFF00;'>🔼 Upload Benchmark & Target Files</h4>", unsafe_allow_html=True)

# Simulate a topbar with two upload sections
top_col1, top_col2, top_col3, top_col4 = st.columns(4)

with top_col1:

    benchmark_files = st.file_uploader("📂 Upload Benchmark File", type=["csv", "pcd", "ulg"], accept_multiple_files=True)
    benchmark_names =  [f.name for f in benchmark_files]
    
with top_col3:
    b_df = None
    if benchmark_files:
        selected_bench = st.selectbox("Select Benchmark File", ["None"] + benchmark_names)
        if selected_bench != "None":
            b_file = benchmark_files[benchmark_names.index(selected_bench)]
            b_file_ext = os.path.splitext(b_file.name)[-1].lower()
            st.session_state.b_df = load_data(b_file, b_file_ext, key_suffix="bench")
        
with top_col2:
    validation_files = st.file_uploader("📂 Upload Target File", type=["csv", "pcd", "ulg"], accept_multiple_files=True)
    validation_names =  [f.name for f in validation_files]
    
with top_col4:
    v_df = None
    if validation_files:
        selected_val = st.selectbox("Select Target File", ["None"] + validation_names)
        if selected_val != "None":
            v_file = validation_files[validation_names.index(selected_val)]
            v_file_ext = os.path.splitext(v_file.name)[-1].lower()
            st.session_state.v_df = load_data(v_file, v_file_ext, key_suffix="val")
        
if "b_df" not in st.session_state:
    st.session_state.b_df = None
if "v_df" not in st.session_state:
    st.session_state.v_df = None
    
b_df = st.session_state.get("b_df")
v_df = st.session_state.get("v_df")

# st.markdown("<h4 style='font-size:18px; color:#0099ff;'>🔧 Data Analysis Settings</h4>", unsafe_allow_html=True)
# selected_df = st.multiselect("Select DataFrame to Modify", ["Benchmark", "Target", "Both"], key="data_enable")

# #         if len(selected_df) > 0:
# for param in selected_df:
#     if param == "Both":
#         st.session_state.b_df, st.session_state.v_df = add_remove_common_column(st.session_state.b_df, st.session_state.v_df)
# #                 b_df, v_df = add_remove_common_column(b_df, v_df)
# #                     b_df = add_remove_column(b_df, df_name="Benchmark")
# #                     v_df = add_remove_column(v_df, df_name="Target")

#     elif param == "Benchmark":
#         st.session_state.b_df = add_remove_column(st.session_state.b_df, df_name="Benchmark")

#     elif param == "Target":
#         st.session_state.v_df = add_remove_column(st.session_state.v_df, df_name="Target")

col_main1, col_main2 = st.columns([0.25, 0.75])
with col_main1:
    st.markdown("<h4 style='font-size:18px; color:#0099ff;'>🔧 Data Analysis Settings</h4>", unsafe_allow_html=True)
    selected_df = st.multiselect("Select DataFrame to Modify", ["Benchmark", "Target", "Both"], key='data_analysis')

#         if len(selected_df) > 0:
    for param in selected_df:
        if param == "Both":
            st.session_state.b_df, st.session_state.v_df = add_remove_common_column(st.session_state.b_df, st.session_state.v_df)
#                 b_df, v_df = add_remove_common_column(b_df, v_df)
#                     b_df = add_remove_column(b_df, df_name="Benchmark")
#                     v_df = add_remove_column(v_df, df_name="Target")

        elif param == "Benchmark":
            st.session_state.b_df = add_remove_column(st.session_state.b_df, df_name="Benchmark")

        elif param == "Target":
            st.session_state.v_df = add_remove_column(st.session_state.v_df, df_name="Target")
    
with col_main2:
    tab1, tab2 = st.tabs(["📊 Plot", "📋 Data"])
    with tab2:
        st.subheader("📁 Imported Data Preview")
    
        # Create working copies for modification
    #     mod_b_df = b_df.copy()
    #     mod_v_df = v_df.copy()
        # Toggle to enable/disable modification tools
    #     modify_enabled = st.checkbox("🧰 Enable DataFrame Modification")
    
    #     if modify_enabled:
    #     colt1, colt2 = st.columns([0.25, 0.75])
    #     with colt1:
    #         st.markdown("<h4 style='font-size:18px; color:#0099ff;'>🔧 Data Analysis Settings</h4>", unsafe_allow_html=True)
    #         selected_df = st.multiselect("Select DataFrame to Modify", ["Benchmark", "Target", "Both"])
    
    # #         if len(selected_df) > 0:
    #         for param in selected_df:
    #             if param == "Both":
    #                 st.session_state.b_df, st.session_state.v_df = add_remove_common_column(st.session_state.b_df, st.session_state.v_df)
    # #                 b_df, v_df = add_remove_common_column(b_df, v_df)
    # #                     b_df = add_remove_column(b_df, df_name="Benchmark")
    # #                     v_df = add_remove_column(v_df, df_name="Target")
    
    #             elif param == "Benchmark":
    #                 st.session_state.b_df = add_remove_column(st.session_state.b_df, df_name="Benchmark")
    
    #             elif param == "Target":
    #                 st.session_state.v_df = add_remove_column(st.session_state.v_df, df_name="Target")
    
        # with colt2:
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
        
#     else: 
#         if b_df is not None and v_df is not None:
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.markdown("### 🧪 Benchmark Data")
#                 st.dataframe(b_df)
#             with col2:
#                 st.markdown("### 🔬 Target Data")
#                 st.dataframe(v_df)

#         elif b_df is not None:
#             st.markdown("### 🧪 Benchmark Data")
#             st.dataframe(b_df)

#         elif v_df is not None:
#             st.markdown("### 🔬 Target Data")
#             st.dataframe(v_df)

#         else:
#             st.info("No data uploaded yet.")
    
    with tab1:
    
        st.subheader(" 🔍 Comparative Analysis")
        if st.session_state.b_df is not None and st.session_state.v_df is not None:
            st.session_state.b_df.insert(0, "Index", range(1, len(st.session_state.b_df) + 1))
            st.session_state.v_df.insert(0, "Index", range(1, len(st.session_state.v_df) + 1))
    
            common_cols = list(set(st.session_state.b_df.columns) & set(st.session_state.v_df.columns))
            if common_cols:
    #             st.markdown("### 🎯 Similarity Metrics")
                col1, col2, col3 = st.columns([0.20, 0.60, 0.20])
                with col1:
                    st.markdown("#### 📈 Parameters")
                    x_axis = st.selectbox("X-Axis", ["None"] + common_cols, key="x_axis_select")
                    y_axis = st.selectbox("Y-Axis", ["None"] + common_cols, key="y_axis_select")
    #                 y_axis = st.multiselect("Select Y-axis (multiple allowed)", [col for col in common_cols if col != x_axis])
    #                 color_axis = st.selectbox("Color by (Optional)", ["None"] + common_cols)
                    z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, key="z-slider")
        
                    if x_axis == "None" and y_axis == "None":
                        st.info("📌 Please upload both files and select a valid X-axis to compare.")
                    elif x_axis is not None and y_axis is not None:
                        x_min = st.number_input("X min", value=float(b_df[x_axis].min()))
                        x_max = st.number_input("X max", value=float(b_df[x_axis].max()))
                        y_min = st.number_input("Y min", value=float(b_df[y_axis].min()))
                        y_max = st.number_input("Y max", value=float(b_df[y_axis].max()))
    
                        # Filter data
                        b_filtered = b_df[(b_df[x_axis] >= x_min) & (b_df[x_axis] <= x_max) &
                                          (b_df[y_axis] >= y_min) & (b_df[y_axis] <= y_max)]
                        v_filtered = v_df[(v_df[x_axis] >= x_min) & (v_df[x_axis] <= x_max) &
                                          (v_df[y_axis] >= y_min) & (v_df[y_axis] <= y_max)]
    
                        merged = pd.merge(b_filtered, v_filtered, on=x_axis, suffixes=('_benchmark', '_validation'))
                        
                        val_col = f"{y_axis}_validation"
                        bench_col = f"{y_axis}_benchmark"
                        abnormal_mask, z_scores = detect_abnormalities(merged[val_col], z_threshold)
                        merged["Z_Score"] = z_scores
                        merged["Abnormal"] = abnormal_mask
                        abnormal_points = merged[merged["Abnormal"]]
                        
                with col2:
                    if x_axis == "None" and y_axis == "None":
                        st.info("📌 Please upload both files and select a valid X-axis to compare.")
                    else:
                        st.markdown("<div style='min-height: 10px'>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        # Plot layout
                        st.markdown("### 🧮 Plot Visualization")
                        fig = go.Figure()
                        fig = make_subplots(rows=2, cols=1, subplot_titles=["Benchmark", "Target"], shared_yaxes=True)
    
                        fig.add_trace(go.Scatter(x=merged[x_axis], y=merged[f"{y_axis}_benchmark"], name="Benchmark", mode="lines"), row=1, col=1)
                        fig.add_trace(go.Scatter(x=merged[x_axis], y=merged[f"{y_axis}_validation"], name="Target", mode="lines"), row=2, col=1)
                        fig.add_trace(go.Scatter(x=abnormal_points[x_axis], y=abnormal_points[f"{y_axis}_validation"],
                                                 mode='markers', marker=dict(color='red', size=6),
                                                 name="Abnormal"), row=2, col=1)
    
                        fig.update_layout(height=700, width=1000, title_text="Benchmark vs Target Subplot")
                        st.plotly_chart(fig, use_container_width=True)
    
            #             st.markdown("### 🎯 Similarity Metrics")
            #             col1, col2, col3 = st.columns(3)
        #                 col1.metric("Similarity Index", f"{similarity*100:.1f}%")
        #                 col2.metric("RMSE", f"{rmse:.2f}")
        #                 col3.metric("Abnormal Points", f"{abnormal_mask.sum()}")
    
                with col3:
                    
                    if x_axis == "None" and y_axis == "None":
                            st.info("📌 Please upload both files and select a valid X-axis to compare.")
                    else:
                        st.markdown("<div style='min-height: 10px'>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("### 🎯 Metrics")
                        rmse = np.sqrt(mean_squared_error(merged[bench_col], merged[val_col]))
                        similarity = 1 - (rmse / (merged[bench_col].max() - merged[bench_col].min()))
    
                        similarity_index = similarity*100
                        
                        fig = make_subplots(
                            rows=3, cols=1,
                            specs=[[{"type": "indicator"}], [{"type": "indicator"}], [{"type": "indicator"}]],
                            vertical_spacing=0.05  # spacing between rows
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
    
        #             with col2:
                        
                        rmse_value = rmse
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
    
        #             with col3:
                        abnormal_points = abnormal_mask.sum()
                        fig.add_trace(go.Indicator(
                            mode="gauge+number",
                            value=abnormal_points,
                            title={'text': "Abnormal Points"},
                            gauge={
                                'axis': {'range': [0, max(10, abnormal_points * 2)]},
                                'bar': {'color': "crimson"},
                                'steps': [
                                    {'range': [0, 10], 'color': "#c8e6c9"},
                                    {'range': [10, 25], 'color': "#ffcc80"},
                                    {'range': [25, 100], 'color': "#ef5350"}
                                ]
                            }
                        ), row=3, col=1)
                        # Final layout
                        fig.update_layout(height=700, margin=dict(t=10, b=10))
                        # Display in Streamlit
                        st.plotly_chart(fig, use_container_width=True)    
    
    
            else:
                st.warning("No common columns to compare between Benchmark and Validation.")
        else:
            st.info("Please upload both benchmark and validation files or pre-converted CSVs.")
    
        
        

        
        
