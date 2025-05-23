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
import open3d as o3d
import tempfile
import os
from pyulog import ULog


st.set_page_config(page_title="ROTRIX Dashboard", layout="wide")
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            width: 310px;
            min-width: 310px;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 ROTRIX Comparative Assessment – Smart Mode")

# 🔹 Logo
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("Rotrix-Logo.png")
st.sidebar.markdown(f'''
    <div style="background-color:white; padding:1px; border-radius:25px; text-align:center;">
        <img src="data:image/png;base64,{logo_base64}" width="250">
    </div>
''', unsafe_allow_html=True)

# File Upload
st.sidebar.header("📁 Upload Test Files")
benchmark_files = st.sidebar.file_uploader("Upload Benchmark data(s)", type=["csv", "pcd", "ulg"], accept_multiple_files=True)
validation_files = st.sidebar.file_uploader("Upload Validation data(s)", type=["csv", "pcd", "ulg"], accept_multiple_files=True)

tab1, tab2 = st.tabs(["Plot", "Data"])
with tab1:
    st.subheader("Your Plot Here")
    # your plotly figure
with tab2:
    st.subheader("Preview Table")
    # your dataframe

def load_csv(file):
    file.seek(0)
    return pd.read_csv(StringIO(file.read().decode("utf-8")))

def detect_trend(series):
    if series.iloc[-1] > series.iloc[0]:
        return "increasing"
    elif series.iloc[-1] < series.iloc[0]:
        return "decreasing"
    return "flat"

def detect_abnormalities(series, threshold=3.0):
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold, z_scores

if benchmark_files and validation_files:
    benchmark_names =  [f.name for f in benchmark_files]
    validation_names =  [f.name for f in validation_files]

    selected_bench = st.sidebar.selectbox("Select Benchmark File", benchmark_names)
    selected_val = st.sidebar.selectbox("Select Validation File", validation_names)

    b_file = benchmark_files[benchmark_names.index(selected_bench)]
    v_file = validation_files[validation_names.index(selected_val)]

    b_df = load_csv(b_file)
    v_df = load_csv(v_file)

    common_cols = list(set(b_df.columns) & set(v_df.columns))
    if not common_cols:
        st.error("No common columns found for comparison.")
    else:
        x_axis = st.sidebar.selectbox("📌 Select X-axis", ["None"] + common_cols)
        y_axis = st.sidebar.selectbox("📌 Select Y-axis", ["None"] + common_cols)
        hue_axis = st.sidebar.selectbox("🎨 Select Hue (Optional)", ["None"] + common_cols)
        plot_type = st.sidebar.radio("📊 Choose Plot Type", ["Line", "Bar", "Scatter", "Heatmap"])

        st.markdown("### 🔍 **Analysis Visualization**")
        
        # Only run merge if all checks are valid
        if x_axis == "None" and y_axis == "None":
            st.info("📌 Please upload both files and select a valid X-axis to compare.")
            
        else:
            
            fig = go.Figure()
            abnormal_summary = []

            name = f"{selected_bench.split('.')[0]} vs {v_file.name.split('.')[0]}"
            merged = pd.merge(b_df, v_df, on=x_axis, suffixes=('_benchmark', '_validation'))

            if f"{y_axis}_benchmark" in merged.columns and f"{y_axis}_validation" in merged.columns:
                # Interactive plot
                fig = None

                x_val = merged[f"{y_axis}_benchmark"]
                y_val = merged[f"{y_axis}_validation"]
                rmse = np.sqrt(mean_squared_error(x_val, y_val))

                st.sidebar.header("🎯 Anomaly Settings")
                z_threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 3.0, 3.0, 0.1)

                abnormal_mask, z_scores = detect_abnormalities(y_val, threshold=z_threshold)

        #             threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 2.0, 3.0, 4.0)
        #             z_scores = np.abs((y_val - y_val.mean()) / y_val.std())

                merged["Z_Score"] = z_scores
                merged["Abnormal"] = abnormal_mask
                abnormal_points = merged[merged["Abnormal"]]
                trend = detect_trend(merged[f"{y_axis}_validation"])

                abnormal_summary.append(f"🔹 **{name}**:\n - RMSE: {rmse:.2f}\n - Trend: {trend}\n - Abnormal Points: {len(abnormal_points)}")

                if plot_type == "Line":
                    fig = px.line(merged, x=x_axis, y=[f"{y_axis}_benchmark", f"{y_axis}_validation"],
                                  title=f"{y_axis} - Line Plot", labels={x_axis: x_axis, "value": y_axis, "variable": "Source"})
                    fig.add_trace(go.Scatter(x=abnormal_points[x_axis], y=abnormal_points[f"{y_axis}_validation"],
                                         mode='markers', marker=dict(color='red', size=8),
                                         name=f"{name} - Abnormal"))

                elif plot_type == "Bar":
                    fig = px.bar(merged, x=x_axis, y=[f"{y_axis}_benchmark", f"{y_axis}_validation"],
                                 title=f"{y_axis} - Bar Plot", barmode='group')

                elif plot_type == "Scatter":
                    fig = px.scatter(merged, x=f"{y_axis}_benchmark", y=f"{y_axis}_validation",
                                     title=f"{y_axis} - Scatter Plot", labels={"x": "Benchmark", "y": "Validation"})
                    fig.add_trace(go.Scatter(x=abnormal_points[x_axis], y=abnormal_points[f"{y_axis}_validation"],
                                         mode='markers', marker=dict(color='red', size=8),
                                         name=f"{name} - Abnormal"))

                elif plot_type == "Heatmap":
                    st.subheader("📌 Similarity Heatmap")
                    sim_df = pd.DataFrame({
                        f"{y_axis}_benchmark": merged[f"{y_axis}_benchmark"],
                        f"{y_axis}_validation": merged[f"{y_axis}_validation"]
                    })
                    corr_matrix = sim_df.corr()
                    fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap")

                if fig:
                    fig.update_layout(hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

                # Abnormality Detection
                st.subheader("⚠️ Abnormality Detection")

                st.write(f"Detected {merged['Abnormal'].sum()} abnormal points.")
                st.dataframe(merged[merged["Abnormal"]][[x_axis, f"{y_axis}_benchmark", f"{y_axis}_validation", "Z_Score"]])

                # Recommendation Section
                st.subheader("💡 Recommendation")
                if merged["Abnormal"].sum() > 0:
                    st.markdown(f"- Significant abnormalities detected in **{y_axis}**.")
                    st.markdown("- Investigate sudden spikes or drops in validation data.")
                    st.markdown("- Re-test this parameter under controlled conditions.")
                else:
                    st.markdown(f"- No major abnormalities detected in **{y_axis}**.")
                    st.markdown("- You may proceed to compare other parameters or file sets.")

                st.subheader("🧠 Pattern-Based Recommendations")
                for rec in abnormal_summary:
                    st.markdown(rec)

            # Side-by-side comparison
            st.subheader("📊 Side-by-Side Multi-Parameter Comparison")
            multi_params = st.multiselect("Select 3 Parameters", [col for col in common_cols if col != x_axis], max_selections=3)


            fig_sub = make_subplots(rows=3, cols=1, shared_xaxes=False, subplot_titles=multi_params)
            v_df = load_csv(validation_files[0])  # Use first validation for subplot example
            merged = pd.merge(b_df, v_df, on=x_axis, suffixes=('_benchmark', '_validation'))

            if len(multi_params) > 0:
                for i, param in enumerate(multi_params, start=1):
                    if f"{param}_benchmark" in merged.columns and f"{param}_validation" in merged.columns:
                        fig_sub.add_trace(go.Scatter(x=merged[x_axis], y=merged[f"{param}_benchmark"],
                                                     name=f"{param} Benchmark", mode='lines'),
                                          row=i, col=1)
                        fig_sub.add_trace(go.Scatter(x=merged[x_axis], y=merged[f"{param}_validation"],
                                                     name=f"{param} Validation", mode='lines'),
                                          row=i, col=1)
                        # fig_sub.add_trace(go.Scatter(x=abnormal_points[x_axis], y=abnormal_points[f"{y_axis}_validation"],
                        #                      mode='markers', marker=dict(color='red', size=8),
                        #                      name=f"{name} - Abnormal"))
    
                fig_sub.update_layout(height=1200, showlegend=True, title_text="3-Parameter Subplot Comparison")
                st.plotly_chart(fig_sub, use_container_width=True)


elif benchmark_files:
    benchmark_names = [f.name for f in benchmark_files]
    selected_bench = st.sidebar.selectbox("Select Benchmark File", benchmark_names)
    b_file = benchmark_files[benchmark_names.index(selected_bench)]

    file_ext = os.path.splitext(b_file.name)[-1].lower()

    if file_ext == ".csv":
        b_df = load_csv(b_file)
        st.success("✅ CSV file loaded successfully.")

    elif file_ext == ".pcd":
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pcd") as tmp_file:
            tmp_file.write(b_file.read())
            tmp_file_path = tmp_file.name
    
         # Load point cloud with Open3D
        pcd = o3d.io.read_point_cloud(tmp_file_path, format='xyz')
        points = np.asarray(pcd.points)
    
        st.sidebar.header("🎯 Pyrometer XY PCD points")
        pcd_rotation = st.sidebar.slider("Rotation Angle", 0, 360, 16, 1)
        # pcd_filter = st.sidebar.slider("laser points", 0, 100000, 1000, 100)
    
        bit_val = 400/(pow(2,20) - 1)               #(mm)
        angle_radians = np.radians(pcd_rotation)
        sin_t = np.sin(angle_radians)
        cos_t = np.cos(angle_radians)
        
        x_rot     = [(d[1]*(-bit_val)*cos_t - d[0]*(bit_val)*sin_t)  for d in points]
        y_rot     = [(d[1]*(-bit_val)*sin_t + d[0]*(bit_val)*cos_t) for d in points]
        EM_rot    = [d[2]              for d in points]

        csv_data = {"X":x_rot, "Y":y_rot, "EM":EM_rot}
        csv_column = ["X", "Y", "EM"]
        
        b_df = pd.DataFrame(csv_data, columns=csv_column) 
        if hasattr(pcd, "colors"):
            b_df["Index"] = b_df.index

    elif file_ext == ".ulg":
        ulog = ULog(b_file)

        csv_append = []
        # Loop through all message types (topics)
        for msg_name, data in zip(ulog.data_list, ulog.data_list):
            b_df = pd.DataFrame(data.data)
            file_name = f"{msg_name.name}.csv"
            b_df.to_csv(file_name, index=False)
            csv_append.append(b_df)
        b_df = csv_append[70]
    

    # b_df = load_csv(b_file)

    common_cols = list(b_df.columns)

    st.sidebar.header("🔧 Select Parameters")
    x_axis = st.sidebar.selectbox("📌 X-Axis", ["None"] + common_cols)
    y_axis = st.sidebar.selectbox("📌 Y-Axis", ["None"] + common_cols)
    color_axis = st.sidebar.selectbox("Color by (Optional)", ["None"] + common_cols)
    plot_type = st.sidebar.radio("📊 Plot Type", ["Line", "Bar", "Scatter"])
    
    st.sidebar.header("🎯 Abnormality Settings")
    z_threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)

    st.subheader("🔍 data Visualization")
    if x_axis != "None" and y_axis != "None" and color_axis != "None":
        x_min = st.sidebar.number_input("X min", value=float(b_df[x_axis].min()))
        x_max = st.sidebar.number_input("X max", value=float(b_df[x_axis].max()))
        y_min = st.sidebar.number_input("Y min", value=float(b_df[y_axis].min()))
        y_max = st.sidebar.number_input("Y max", value=float(b_df[y_axis].max()))
        em_min = st.sidebar.number_input("Color min", value=float(b_df[color_axis].min()))
        em_max = st.sidebar.number_input("Color max", value=float(b_df[color_axis].max()))
    
        b_df = b_df[(b_df[x_axis] >= x_min) & (b_df[x_axis] <= x_max) & 
        (b_df[y_axis] >= y_min) & (b_df[y_axis] <= y_max) & 
        (b_df[color_axis] >= em_min) & (b_df[color_axis] <= em_max)]
        
        y_data = b_df[y_axis]
        abnormal_mask, z_scores = detect_abnormalities(y_data, threshold=z_threshold)
        b_df["Z_Score"] = z_scores
        b_df["Abnormal"] = abnormal_mask
        abnormal_points = b_df[b_df["Abnormal"]]

        fig = go.Figure()
        if plot_type == "Line":
            fig = px.line(b_df, x=x_axis, y=y_axis, title=f"{y_axis} - Line Plot")
            fig.add_trace(go.Scatter(x=abnormal_points[x_axis], y=abnormal_points[y_axis],
                                     mode='markers', marker=dict(color='red', size=8),
                                     name="Abnormal"))
        elif plot_type == "Bar":
            fig = px.bar(b_df, x=x_axis, y=y_axis, title=f"{y_axis} - Bar Plot")
        elif plot_type == "Scatter":
            if color_axis != "None":
                
                fig = px.scatter(b_df, x=x_axis, y=y_axis, color=b_df[color_axis], title=f"{y_axis} - Scatter Plot",
                                 color_continuous_scale="Turbo", width=1000, height=600)
    
            else:
                fig = px.scatter(b_df, x=x_axis, y=y_axis, title=f"{y_axis} - Scatter Plot")
                fig.add_trace(go.Scatter(x=abnormal_points[x_axis], y=abnormal_points[y_axis],
                                     mode='markers', marker=dict(color='red', size=8),
                                     name="Abnormal"))
        
        elif plot_type == "Heatmap":
             st.subheader("📌 Similarity Heatmap")
             sim_df = pd.DataFrame({f"{y_axis}_benchmark": merged[f"{y_axis}_benchmark"],
                                    f"{y_axis}_validation": merged[f"{y_axis}_validation"]
                                   })
             corr_matrix = sim_df.corr()
             fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap")

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("⚠️ Abnormality Detection")
        st.write(f"Detected {abnormal_mask.sum()} abnormal points.")
        st.dataframe(b_df[abnormal_mask][[x_axis, y_axis, "Z_Score"]])

        st.subheader("🧠 Trend Insight")
        st.markdown(f"📈 **{y_axis}** trend is **{detect_trend(y_data)}**")

        st.subheader("📊 Multi-Parameter Comparison (3-Subplots)")
        multi_params = st.multiselect("Select 3 Parameters", [col for col in common_cols if col != x_axis], max_selections=3)
        
        fig_sub = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=multi_params)
        for i, param in enumerate(multi_params, start=1):
            if f"{param}" in b_df.columns:
                fig_sub.add_trace(go.Scatter(x=b_df[x_axis], y=b_df[param], name=param, mode='lines'), row=i, col=1)
                # fig_sub.add_trace(go.Scatter(x=abnormal_points[x_axis], y=abnormal_points[y_axis],
                #                          mode='markers', marker=dict(color='red', size=8),
                #                          name="Abnormal"))
            fig_sub.update_layout(height=900, title_text="3-Parameter Subplot Comparison")
        st.plotly_chart(fig_sub, use_container_width=True)


    # # If validation files also exist
    # if validation_files:
    #     validation_names = [f.name for f in validation_files]
    #     selected_val = st.sidebar.selectbox("Select Validation File", validation_names)
    #     v_file = validation_files[validation_names.index(selected_val)]
    #     v_df = load_csv(v_file)

    #     if x_axis in b_df.columns and x_axis in v_df.columns and y_axis in b_df.columns and y_axis in v_df.columns:
    #         merged = pd.merge(b_df, v_df, on=x_axis, suffixes=('_benchmark', '_validation'))
    #         y_b = merged[f"{y_axis}_benchmark"]
    #         y_v = merged[f"{y_axis}_validation"]
    #         rmse = np.sqrt(mean_squared_error(y_b, y_v))
    #         abnormal_mask, z_scores = detect_abnormalities(y_v, z_threshold)
    #         merged["Z_Score"] = z_scores
    #         merged["Abnormal"] = abnormal_mask
    #         abnormal_points = merged[merged["Abnormal"]]

    #         st.subheader("📈 Benchmark vs Validation Comparison")
    #         fig_comp = px.line(merged, x=x_axis, y=[f"{y_axis}_benchmark", f"{y_axis}_validation"],
    #                            title=f"{y_axis} Benchmark vs Validation")
    #         fig_comp.add_trace(go.Scatter(x=abnormal_points[x_axis], y=abnormal_points[f"{y_axis}_validation"],
    #                                       mode='markers', marker=dict(color='red', size=8),
    #                                       name="Abnormal"))
    #         st.plotly_chart(fig_comp, use_container_width=True)

    #         st.subheader("💡 Recommendation")
    #         if abnormal_mask.sum() > 0:
    #             st.markdown(f"- **{abnormal_mask.sum()}** abnormal points detected in validation.")
    #             st.markdown("- Investigate spikes or irregularities.")
    #         else:
    #             st.markdown("- No major abnormalities found.")
