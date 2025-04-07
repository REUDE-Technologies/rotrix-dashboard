
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
from PIL import Image
import base64

st.set_page_config(page_title="ROTRIX Advanced Dashboard", layout="wide")

# Load logo and convert to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("Rotrix-Logo.png")

# Custom HTML with background and centered image
st.sidebar.markdown(f"""
    <div style="background-color:white; padding:1px; border-radius:25px; text-align:center;">
        <img src="data:image/png;base64,{logo_base64}" width="250">
    </div>
""", unsafe_allow_html=True)

# 🔥 Add ROTRIX logo at the top
# logo = Image.open("Rotrix-Logo.png")
st.title("🚀 ROTRIX Comparative Assessment – Advanced Dashboard")

# st.sidebar.image(logo)
st.sidebar.header("📁 Upload Test Files")
benchmark_files = st.sidebar.file_uploader("Upload Benchmark CSV(s)", type=["csv"], accept_multiple_files=True)
validation_files = st.sidebar.file_uploader("Upload Validation CSV(s)", type=["csv"], accept_multiple_files=True)

def load_csv(file):
    file.seek(0)
    return pd.read_csv(StringIO(file.read().decode("utf-8")))

def detect_trend(y):
    direction = "flat"
    if y.iloc[-1] > y.iloc[0]:
        direction = "increasing"
    elif y.iloc[-1] < y.iloc[0]:
        direction = "decreasing"
    return direction

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

                abnormal_summary.append(f"🔹 **{name}**:\n - RMSE: {rmse:.2f}\n - Trend: {trend}\n - Abnormal Points {len(abnormal_points)}")

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

            for i, param in enumerate(multi_params, start=1):
                if f"{param}_benchmark" in merged.columns and f"{param}_validation" in merged.columns:
                    fig_sub.add_trace(go.Scatter(x=merged[x_axis], y=merged[f"{param}_benchmark"],
                                                 name=f"{param} Benchmark", mode='lines'),
                                      row=i, col=1)
                    fig_sub.add_trace(go.Scatter(x=merged[x_axis], y=merged[f"{param}_validation"],
                                                 name=f"{param} Validation", mode='lines'),
                                      row=i, col=1)

                fig_sub.update_layout(height=1200, showlegend=True, title_text="3-Parameter Subplot Comparison")
                st.plotly_chart(fig_sub, use_container_width=True)

        #         elif len(multi_params) != 3 and multi_params:
        #             st.warning("Please select exactly 3 parameters for subplot display.")

    # else:
    #     st.info("Upload your Benchmark and Validation files to begin.")
