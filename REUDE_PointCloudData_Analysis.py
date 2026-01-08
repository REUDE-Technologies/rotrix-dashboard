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
import io
import os
import requests
import zipfile
import cv2
import shutil
import matplotlib.pyplot as plt
import time

# Function to optimize video generation with OpenCV
def create_video_from_frames(frame_paths, output_path, fps, quality="High (H.264)", progress_callback=None):
    """
    Create video from frame paths using OpenCV with optimized memory usage.
    
    Args:
        frame_paths: List of paths to frame images
        output_path: Output video path
        fps: Frames per second
        quality: Video quality setting ("High (H.264)" or "Standard (MP4V)")
        progress_callback: Optional callback function for progress updates
    """
    if not frame_paths:
        return False, "No frames provided"
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        return False, "Failed to read first frame"
    
    height, width, layers = first_frame.shape
    
    # Initialize video writer with MP4V codec (more reliable)
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            return False, "Failed to initialize video writer"
    except Exception as e:
        return False, f"Error initializing video writer: {e}"
    
    # Add frames to video with progress
    successful_frames = 0
    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is not None:
            video_writer.write(frame)
            successful_frames += 1
        
        # Call progress callback if provided
        if progress_callback:
            progress_callback(i + 1, len(frame_paths))
    
    # Release video writer
    video_writer.release()
    
    if successful_frames == 0:
        return False, "No frames were successfully processed"
    
    return True, f"Successfully processed {successful_frames}/{len(frame_paths)} frames"

# Function to load custom XYZ-like file using readlines
def load_custom_xyz_file(file_or_path):
    try:
        # Handle file-like object (e.g., UploadedGitHubFile)
        if hasattr(file_or_path, 'read'):
            content = file_or_path.read().decode('utf-8')  # Read and decode content
            lines = content.splitlines()  # Split into lines
        # Handle file path
        else:
            with open(file_or_path, 'r') as f:
                lines = f.readlines()

        # Filter out comment lines and empty lines
        data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]

        # Parse numeric lines with 3 or more numbers
        data = []
        for line in data_lines:
            numbers = list(map(float, line.split()))
            if len(numbers) >= 3:
                # Use first 3 for x, y, z, and 4th column for temperature if available
                if len(numbers) >= 4:
                    data.append(numbers[:4])  # x, y, z, temperature
                else:
                    # If no temperature data, use z value as temperature
                    data.append(numbers[:3] + [numbers[2]])

        if not data:
            raise ValueError("No valid 3D points found in file.")

        # Create DataFrame with temperature column
        if len(data[0]) == 4:
            df = pd.DataFrame(np.array(data), columns=["x", "y", "z", "temperature"])
        else:
            df = pd.DataFrame(np.array(data), columns=["x", "y", "z"])
            df["temperature"] = df["z"]  # Use z as temperature if no temperature data
        
        return df
    except Exception as e:
        raise ValueError(f"Failed to parse XYZ-like file: {e}")

# Function to plot the PCD files as 2D (x, y) with temperature as color
def plot_xyz_like_files(col, label, selected_files, x_axis, y_axis, x_min, x_max, y_min, y_max, z_threshold, color_axis="z", colorscale="plasma"):
    col.markdown(f"### {label} Layers")
    for path in selected_files:
        try:
            df = load_custom_xyz_file(path)

            # Calculate initial axis limits from unfiltered data with padding
            x_padding = (df[x_axis].max() - df[x_axis].min()) * 0.1
            y_padding = (df[y_axis].max() - df[y_axis].min()) * 0.1
            plot_x_min = df[x_axis].min() - x_padding
            plot_x_max = df[x_axis].max() + x_padding
            plot_y_min = df[y_axis].min() - y_padding
            plot_y_max = df[y_axis].max() + y_padding

            # Debug: Print the calculated ranges
            st.write(f"Debug - {os.path.basename(path)}: x_range=[{plot_x_min:.2f}, {plot_x_max:.2f}], y_range=[{plot_y_min:.2f}, {plot_y_max:.2f}]")

            # Filter data based on user parameters (using the provided limits as a starting point)
            df_filtered = df[(df[x_axis] >= plot_x_min) & (df[x_axis] <= plot_x_max) &
                             (df[y_axis] >= plot_y_min) & (df[y_axis] <= plot_y_max)]

            # Estimate density by 2D histogram and crop high-density area
            x_bins = np.linspace(df_filtered[x_axis].min(), df_filtered[x_axis].max(), 200)
            y_bins = np.linspace(df_filtered[y_axis].min(), df_filtered[y_axis].max(), 200)
            hist, xedges, yedges = np.histogram2d(df_filtered[x_axis], df_filtered[y_axis], bins=[x_bins, y_bins])

            # Threshold: find densest area
            idx = np.where(hist > np.percentile(hist, 95))
            if idx[0].size > 0:
                x_start, x_end = xedges[min(idx[0])], xedges[max(idx[0]) + 1]
                y_start, y_end = yedges[min(idx[1])], yedges[max(idx[1]) + 1]
                df_filtered = df_filtered[(df_filtered[x_axis] >= x_start) & (df_filtered[x_axis] <= x_end) &
                                         (df_filtered[y_axis] >= y_start) & (df_filtered[y_axis] <= y_end)]

            # Detect abnormal points based on z_threshold
            z_scores = np.abs((df_filtered["z"] - df_filtered["z"].mean()) / df_filtered["z"].std())
            abnormal_points = df_filtered[z_scores > z_threshold]

            # Use z-axis for coloring by default
            if color_axis == "z" and "z" in df_filtered.columns:
                color_data = df_filtered["z"]
                color_title = "Temp"
                marker_dict = dict(
                    color=color_data,
                    colorscale=colorscale,
                    size=4,
                    showscale=True,
                    colorbar=dict(title=color_title, tickfont=dict(size=12)),
                    cmin=650,
                    cmax=1000
                )
            else:
                # Default to z if available
                color_data = df_filtered["z"]
                color_title = "Z Height"
                marker_dict = dict(
                    color=color_data,
                    colorscale=colorscale,
                    size=4,
                    showscale=True,
                    colorbar=dict(title=color_title, tickfont=dict(size=12)),
                    cmin=650,
                    cmax=1000
                )
            
            # Create traces list with main data first, then abnormal points on top
            traces = []
            
            # Add main data trace
            traces.append(go.Scattergl(
                x=df_filtered[x_axis], y=df_filtered[y_axis],
                mode='markers',
                marker=marker_dict,
                name="Data Points"
            ))

            # Add abnormal points trace on top
            if not abnormal_points.empty:
                traces.append(go.Scattergl(
                    x=abnormal_points[x_axis], y=abnormal_points[y_axis],
                    mode='markers', 
                    marker=dict(color='red', size=8, opacity=0.8),
                    name="Abnormal"
                ))
            
            fig = go.Figure(data=traces)

            fig.update_layout(
                title=os.path.basename(path),
                height=400,
                plot_bgcolor='black',
                paper_bgcolor='black',
                xaxis=dict(
                    range=[plot_x_min, plot_x_max], 
                    showgrid=False, 
                    zeroline=False,
                    showticklabels=False,
                    title=""
                ),
                yaxis=dict(
                    range=[plot_y_min, plot_y_max], 
                    showgrid=False, 
                    zeroline=False,
                    showticklabels=False,
                    title=""
                ),
                margin=dict(l=10, r=10, b=50, t=30),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="white")
                )
            )
            col.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            col.warning(f"{os.path.basename(path)} - error: {e}")

# Function to plot two files as scatter plots side by side
def plot_two_layers(col1, col2, file1, file2, x_axis, y_axis, z_threshold, x_min, x_max, y_min, y_max, colorscale="plasma"):
    try:
        # Load and filter first file
        if hasattr(file1, 'read'):
            file1.seek(0)  # Reset pointer for multiple reads
        df1 = load_custom_xyz_file(file1)
        df1_filtered = df1[(df1[x_axis] >= x_min) & (df1[x_axis] <= x_max) &
                          (df1[y_axis] >= y_min) & (df1[y_axis] <= y_max)]
        z_scores1 = np.abs((df1_filtered["z"] - df1_filtered["z"].mean()) / df1_filtered["z"].std())
        abnormal_points1 = df1_filtered[z_scores1 > z_threshold]

        # Load and filter second file
        if hasattr(file2, 'read'):
            file2.seek(0)  # Reset pointer for multiple reads
        df2 = load_custom_xyz_file(file2)
        df2_filtered = df2[(df2[x_axis] >= x_min) & (df2[x_axis] <= x_max) &
                          (df2[y_axis] >= y_min) & (df2[y_axis] <= y_max)]
        z_scores2 = np.abs((df2_filtered["z"] - df2_filtered["z"].mean()) / df2_filtered["z"].std())
        abnormal_points2 = df2_filtered[z_scores2 > z_threshold]

        # Use z-axis for coloring
        color_data1 = df1_filtered["z"]
        color_title1 = "Temp"
        
        marker_dict1 = dict(
            color=color_data1,
            colorscale=colorscale,
            size=4,
            showscale=True,
            colorbar=dict(title=color_title1),
            cmin=650,
            cmax=1000
        )
        
        # Plot first file with traces in correct order
        traces1 = []
        traces1.append(go.Scattergl(
            x=df1_filtered[x_axis], y=df1_filtered[y_axis],
            mode='markers',
            marker=marker_dict1,
            name="Data Points"
        ))
        if not abnormal_points1.empty:
            traces1.append(go.Scattergl(
                x=abnormal_points1[x_axis], y=abnormal_points1[y_axis],
                mode='markers', 
                marker=dict(color='red', size=8, opacity=0.8), 
                name="Abnormal"
            ))
        fig1 = go.Figure(data=traces1)
        fig1.update_layout(
            title=os.path.basename(getattr(file1, 'name', 'File1')), 
            height=400,
            plot_bgcolor='black',
            paper_bgcolor='black',
            xaxis=dict(range=[x_min, x_max], showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(range=[y_min, y_max], showgrid=False, zeroline=False, showticklabels=False, title=""),
            margin=dict(l=10, r=10, b=50, t=30),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(color="white")
            )
        )
        col1.plotly_chart(fig1, use_container_width=True)

        # Use z-axis for coloring
        color_data2 = df2_filtered["z"]
        color_title2 = "Temp"
        
        marker_dict2 = dict(
            color=color_data2,
            colorscale=colorscale,
            size=4,
            showscale=True,
            colorbar=dict(title=color_title2),
            cmin=650,
            cmax=1000
        )
        
        # Plot second file with traces in correct order
        traces2 = []
        traces2.append(go.Scattergl(
            x=df2_filtered[x_axis], y=df2_filtered[y_axis],
            mode='markers',
            marker=marker_dict2,
            name="Data Points"
        ))
        if not abnormal_points2.empty:
            traces2.append(go.Scattergl(
                x=abnormal_points2[x_axis], y=abnormal_points2[y_axis],
                mode='markers', 
                marker=dict(color='red', size=8, opacity=0.8), 
                name="Abnormal"
            ))
        fig2 = go.Figure(data=traces2)
        fig2.update_layout(
            title=os.path.basename(getattr(file2, 'name', 'File2')), 
            height=400,
            plot_bgcolor='black',
            paper_bgcolor='black',
            xaxis=dict(range=[x_min, x_max], showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(range=[y_min, y_max], showgrid=False, zeroline=False, showticklabels=False, title=""),
            margin=dict(l=10, r=10, b=50, t=30),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(color="white")
            )
        )
        col2.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        col1.warning(f"{getattr(file1, 'name', 'File1')} - error: {e}")
        col2.warning(f"{getattr(file2, 'name', 'File2')} - error: {e}")

# --- Initialize Session States ---
if 'files_submitted' not in st.session_state:
    st.session_state.files_submitted = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'show_upload_area' not in st.session_state:
    st.session_state.show_upload_area = True
if 'file_rename_mode' not in st.session_state:
    st.session_state.file_rename_mode = {}
if 'file_share_mode' not in st.session_state:
    st.session_state.file_share_mode = {}
if 'share_all_mode' not in st.session_state:
    st.session_state.share_all_mode = False
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Single"
if "plot_ready_single_part" not in st.session_state:
    st.session_state.plot_ready_single_part = False

st.cache_data.clear()  # Clear cache to ensure latest code is used

st.set_page_config(page_title="Point Cloud Data Dashboard", layout="wide")

# Add custom CSS to improve layout
st.markdown("""
<style>
    /* Ensure plots stay within their columns */
    .stPlotlyChart {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Improve column spacing */
    .row-widget.stHorizontal > div {
        padding: 0 10px;
    }
    
    /* Better spacing for sidebar */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Ensure proper plot container sizing */
    .element-container {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# 🔹 Logo
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error("Logo file 'Rotrix-Logo.png' not found. Please ensure it exists in the working directory.")
        return ""

logo_col, upload_btn_col = st.columns([8, 1])
with logo_col:
    logo_base64 = get_base64_image("Rotrix-Logo.png")
    st.markdown(f"""
        <div style="display: flex; justify-content: left; align-items: center; padding: 1px; background-color:white; border-radius:25px;">
            <a href="http://rotrix.reude.tech/" target="_blank">
                <img src="data:image/png;base64,{logo_base64}" width="180" alt="Rotrix Logo">
            </a>
        </div>
    """, unsafe_allow_html=True)
with upload_btn_col:
    if not (not st.session_state.files_submitted or st.session_state.show_upload_area):
        st.markdown("""
        <style>
        .plus-upload-btn button {
            background: #2E86C1;
            color: white;
            font-weight: 700;
            font-size: 1.1rem;
            border-radius: 22px;
            padding: 8px 22px;
            margin-top: 18px;
            margin-left: 10px;
            box-shadow: 0 2px 8px rgba(44,62,80,0.07);
            transition: background 0.2s, box-shadow 0.2s, transform 0.1s;
        }
        .plus-upload-btn button:hover {
            background: #1B4F72;
            color: #fff;
            transform: translateY(-2px) scale(1.04);
        }
        </style>
        """, unsafe_allow_html=True)
        with st.container():
            if st.button("➕ Upload", key="plus_upload_btn", help="Upload or manage files", use_container_width=True):
                st.session_state.show_upload_area = True
                st.rerun()
st.markdown("<h3 style='color:#c71585;'>🚀 Data Visionary</h3>", unsafe_allow_html=True)

class UploadedGitHubFile:
    def __init__(self, content, name, filetype):
        from io import BytesIO
        self.file = BytesIO(content)
        self.name = name
        self.type = filetype
        self.size = len(content)
    def read(self, *args, **kwargs):
        return self.file.read(*args, **kwargs)
    def seek(self, *args, **kwargs):
        return self.file.seek(*args, **kwargs)

# Load logic
def load_data(file_content, filetype):
    if filetype == ".csv":
        try:
            df = pd.read_csv(StringIO(file_content.decode("utf-8")))
            return df
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return None
    elif filetype == ".pcd":
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pcd") as tmp:
                tmp.write(file_content)
                filepath = tmp.name
                with open(filepath, 'r') as f:
                    lines = f.readlines()
            data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
            data_array = np.array([list(map(float, line.split())) for line in data_lines])
            os.unlink(filepath)
            points = data_array[:, :3]
            # Store original points for rotation later
            df = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]})
            return df
        except Exception as e:
            st.error(f"Error loading PCD: {e}")
            return None
    return None

# Process local folder
def process_local_folder(folder_path):
    try:
        if not os.path.isdir(folder_path):
            st.error(f"The path '{folder_path}' is not a valid directory.")
            return None
        file_data = {}
        for file_name in os.listdir(folder_path):
            file_ext = os.path.splitext(file_name)[-1].lower()
            if file_ext in [".csv", ".pcd"]:
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'rb') as f:
                    file_data[file_name] = (f.read(), file_ext)
        if not file_data:
            st.warning(f"No .csv or .pcd files found in the folder: {folder_path}")
            return None
        return file_data
    except PermissionError:
        st.error(f"Permission denied accessing folder: {folder_path}")
        return None
    except Exception as e:
        st.error(f"Error processing local folder: {str(e)}")
        return None

# Process local file
def process_local_file(file_path):
    try:
        if not os.path.isfile(file_path):
            st.error(f"The path '{file_path}' is not a valid file.")
            return None
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[-1].lower()
        if file_ext in [".csv", ".pcd"]:
            with open(file_path, 'rb') as f:
                return {file_name: (f.read(), file_ext)}
        else:
            st.warning(f"Unsupported file type. Please provide a .csv or .pcd file.")
            return None
    except PermissionError:
        st.error(f"Permission denied accessing file: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error processing local file: {str(e)}")
        return None

# Fetch files from GitHub folder or load single file
def process_url(url):
    url = url.strip()
    # Check if the input is a local path (file or folder)
    if os.path.exists(url):
        if os.path.isdir(url):
            return process_local_folder(url)
        elif os.path.isfile(url):
            return process_local_file(url)
    # Handle GitHub URLs
    if "github.com" in url or "raw.githubusercontent.com" in url:
        try:
            # Handle folder URL (e.g., https://github.com/username/repo/tree/main/folder)
            if "tree" in url:
                parts = url.split("tree/")
                if len(parts) < 2:
                    st.error("Invalid GitHub folder URL format. Please include 'tree/' followed by the folder path.")
                    return None
                base_url = parts[0].rstrip('/')
                path = parts[1].lstrip('/')
                url_parts = base_url.split("/")
                if len(url_parts) < 5 or url_parts[2] != "github.com":
                    st.error("Unable to parse repository from URL.")
                    return None
                repo = f"{url_parts[3]}/{url_parts[4]}"
                folder_path = path.split('/', 1)[-1] if '/' in path else path
                api_url = f"https://api.github.com/repos/{repo}/contents/{folder_path}"
                response = requests.get(api_url, headers={"Accept": "application/vnd.github.v3+json"})
                if response.status_code == 200:
                    files = [item for item in response.json() if item['type'] == 'file' and item['name'].endswith(('.csv', '.pcd'))]
                    if not files:
                        st.warning("No .csv or .pcd files found in the GitHub folder.")
                        return None
                    file_data = {}
                    for file in files:
                        file_response = requests.get(file['download_url'])
                        if file_response.status_code == 200:
                            file_ext = os.path.splitext(file['name'])[-1].lower()
                            file_data[file['name']] = (file_response.content, file_ext)
                    return file_data if file_data else None
                else:
                    st.error(f"Failed to fetch GitHub folder contents. Status code: {response.status_code}")
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
                        st.error(f"Failed to download GitHub file. Status code: {response.status_code}")
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
                        st.error(f"Failed to download GitHub file. Status code: {response.status_code}")
                        return None
            # Handle direct file URL (e.g., https://github.com/username/repo/filename.csv)
            elif len(url.split("/")) > 4 and url.split("/")[4] not in ["tree", "blob", "raw"]:
                base_parts = url.split("/")
                repo = f"{base_parts[3]}/{base_parts[4]}"
                file_path = "/".join(base_parts[5:])
                raw_url = f"https://raw.githubusercontent.com/{repo}/main/{file_path}"
                file_name = file_path.split("/")[-1]
                file_ext = os.path.splitext(file_name)[-1].lower()
                if file_ext in [".csv", ".pcd"]:
                    response = requests.get(raw_url)
                    if response.status_code == 200:
                        return {file_name: (response.content, file_ext)}
                    else:
                        st.error(f"Failed to download GitHub file. Status code: {response.status_code}")
                        return None
            else:
                st.warning("Unsupported GitHub URL format. Please use a folder URL with 'tree/' or a raw/blob file URL.")
                return None
        except Exception as e:
            st.error(f"Error processing GitHub URL: {str(e)}")
            return None
    else:
        st.warning("The provided path is neither a valid GitHub URL nor a local folder/file. Please upload files manually or provide a valid path.")
        return None

if not st.session_state.files_submitted or st.session_state.show_upload_area:
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
    .main .block-container {
        padding-top: 40px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #2E86C1; margin-bottom: 30px;'>📁 File Upload & Management</h3>", unsafe_allow_html=True)
    github_col, upload_col = st.columns([1, 1])
    with github_col:
        st.markdown("""
        <div style='padding: 12px 16px; background: #f0f8ff; border-radius: 8px; border: 1.5px solid #b3d8fd; margin-bottom: 8px;'>
            <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 2px;'>
                <img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' width='22' style='margin-right: 4px;'/>
                <span style='font-size: 1.08rem; font-weight: 700; color: #24292f;'>GitHub</span>
                <span style='font-size: 0.98rem; color: #2980b9; margin-left: 6px;'>(<a style='color:#2980b9; text-decoration:underline; cursor:pointer;' href='#'>.csv, .pcd</a>)</span>
            </div>
            <div style='font-size: 0.98rem; color: #444;'>
                Paste a <b>GitHub <span style='font-weight:700;'>raw/blob/folder URL</span></b> to fetch files.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.get("clear_github_url_input", False):
            st.session_state.github_url_input = ""
            st.session_state.clear_github_url_input = False
        github_col, fetch_col = st.columns([5, 1])
        with fetch_col:
            fetch_github = st.button("Fetch", key="fetch_github_btn", use_container_width=True)
        with github_col:
            github_url = st.text_input("GitHub URL (raw, blob, or folder)", key="github_url_input", label_visibility="collapsed", placeholder="e.g. https://github.com/user/repo/blob/main/data.csv")
        if fetch_github and github_url:
            result = process_url(github_url)
            if result:
                existing_names = [f.name for f in st.session_state.uploaded_files]
                for file_name, (file_content, file_ext) in result.items():
                    if file_name not in existing_names:
                        filetype = "text/csv" if file_ext == ".csv" else "application/octet-stream"
                        file_like = UploadedGitHubFile(file_content, file_name, filetype)
                        st.session_state.uploaded_files.append(file_like)
                st.session_state.clear_github_url_input = True
                st.rerun()
            else:
                st.warning("No valid .csv or .pcd files found at the provided URL.")
    with upload_col:
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=["csv", "pcd"],
            key="desktop_uploader",
            label_visibility="collapsed",
            accept_multiple_files=True,
            help="Drag and drop files here or click to browse"
        )
    if uploaded_files:
        new_files_added = False
        existing_names = [f.name for f in st.session_state.uploaded_files]
        for uploaded_file in uploaded_files:
            # Manual size check for 1GB limit
            if uploaded_file.size > 1073741824:  # 1GB in bytes
                st.error(f"File '{uploaded_file.name}' exceeds the 1GB size limit and cannot be uploaded.")
            elif uploaded_file.name not in existing_names:
                st.session_state.uploaded_files.append(uploaded_file)
                new_files_added = True
        if new_files_added:
            st.rerun()

    if st.session_state.uploaded_files:
        st.markdown("""
        <style>
        .file-type-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .file-type-badge.csv {
            background: #d4edda;
            color: #155724;
        }
        .file-type-badge.pcd {
            background: #d1ecf1;
            color: #0c5460;
        }
        .file-stats {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .file-stats h6 {
            color: rgba(255,255,255,0.9);
            margin-bottom: 8px;
        }
        .file-stats .stat-value {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .file-stats .stat-label {
            font-size: 12px;
            opacity: 0.8;
        }
        </style>
        """, unsafe_allow_html=True)

        # Two-column layout for file preview
        preview_col, actions_col = st.columns([0.7, 0.3])
        with preview_col:
            st.markdown("<h4 style='margin-top: 0px; color: #495057;'>📋 File Preview & Management</h4>", unsafe_allow_html=True)
            st.markdown("<h5 style='color: #6c757d; margin-bottom: 15px;'>📎 Uploaded Files</h5>", unsafe_allow_html=True)
            for i, file in enumerate(st.session_state.uploaded_files):
                file_name = file.name
                file_ext = file.name.split('.')[-1].lower() if '.' in file.name else 'unknown'
                file_type_badge = f"<span class='file-type-badge {file_ext}'>{file_ext}</span>"
                file_size = file.size / (1024*1024)
                file_type = getattr(file, 'type', 'Unknown')
                file_cols = st.columns([12, 1, 1, 1, 1, 1])
                with file_cols[0]:
                    st.markdown(f"""
                    <div style="font-weight: 600; color: #495057;">
                        📄 {file_name}
                        <span style="font-size: 12px; color: #6c757d; margin-left: 10px;">
                            Size: {file_size:.1f} MB | Type: {file_type} {file_type_badge}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                with file_cols[1]:
                    if st.button("🔍", key=f"preview_btn_{i}", use_container_width=True, help="Quick Preview"):
                        st.session_state[f"preview_mode_{i}"] = not st.session_state.get(f"preview_mode_{i}", False)
                        st.rerun()
                with file_cols[2]:
                    if st.button("✏", key=f"rename_btn_{i}", use_container_width=True, help="Rename"):
                        st.session_state.file_rename_mode[i] = not st.session_state.file_rename_mode.get(i, False)
                        st.rerun()
                with file_cols[3]:
                    if st.button("➦", key=f"share_btn_{i}", use_container_width=True, help="Share"):
                        st.session_state.file_share_mode[i] = not st.session_state.file_share_mode.get(i, False)
                        st.rerun()
                with file_cols[4]:
                    if hasattr(file, 'read'):
                        file.seek(0)
                        st.download_button(
                            label="⬇",
                            data=file.read(),
                            file_name=file_name,
                            mime=file_type or "application/octet-stream",
                            key=f"download_btn_{i}",
                            use_container_width=True,
                            help="Download"
                        )
                        file.seek(0)
                    else:
                        st.download_button(
                            label="⬇",
                            data=file.read(),
                            file_name=file_name,
                            mime="application/octet-stream",
                            key=f"download_btn_{i}",
                            use_container_width=True,
                            help="Download"
                        )
                with file_cols[5]:
                    if st.button("🗑", key=f"remove_btn_{i}", use_container_width=True, help="Remove"):
                        st.session_state.uploaded_files.pop(i)
                        st.rerun()
                # Quick Preview Mode
                if st.session_state.get(f"preview_mode_{i}", False):
                    with st.expander("Quick Preview", expanded=True):
                        if hasattr(file, 'read'):
                            file.seek(0)
                            file_ext = file_name.split('.')[-1].lower() if '.' in file_name else 'unknown'
                            if file_ext == "csv":
                                try:
                                    import pandas as pd
                                    df = pd.read_csv(file, nrows=5)
                                    st.dataframe(df, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Could not preview CSV: {e}")
                            elif file_ext == "pcd":
                                st.write({"Name": file_name, "Size (MB)": f"{file_size:.2f} MB", "Type": file_type})
                            else:
                                st.warning("Preview not supported for this file type.")
                            file.seek(0)
                        else:
                            if file_ext == ".csv":
                                try:
                                    import pandas as pd
                                    from io import BytesIO
                                    df = pd.read_csv(BytesIO(file.read()), nrows=5)
                                    st.dataframe(df, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Could not preview CSV: {e}")
                            elif file_ext == "pcd":
                                st.info("PCD preview: Only file name, size, and type shown.")
                                st.write({"Name": file_name, "Size (MB)": f"{file_size:.2f} MB", "Type": file_type})
                            else:
                                st.warning("Preview not supported for this file type.")
                # Rename mode
                if st.session_state.file_rename_mode.get(i, False):
                    with st.container():
                        st.markdown("✏ Rename File")
                        col_rename1, col_rename2, col_rename3 = st.columns([2, 1, 1])
                        with col_rename1:
                            new_name = st.text_input(
                                "New file name:",
                                value=file_name,
                                key=f"rename_input_{i}"
                            )
                        with col_rename2:
                            if st.button("✅ Save", key=f"save_rename_{i}", use_container_width=True):
                                if new_name and new_name != file_name:
                                    if hasattr(file, 'name'):
                                        file.name = new_name
                                    else:
                                        st.session_state.uploaded_files[i] = UploadedGitHubFile(file.read(), new_name, file.type)
                                    st.success(f"File renamed to: {new_name}")
                                st.session_state.file_rename_mode[i] = False
                                st.rerun()
                        with col_rename3:
                            if st.button("❌ Cancel", key=f"cancel_rename_{i}", use_container_width=True):
                                st.session_state.file_rename_mode[i] = False
                                st.rerun()
                # Share mode
                if st.session_state.file_share_mode.get(i, False):
                    with st.container():
                        st.markdown("➦ Share File")
                        share_option = st.selectbox(
                            "Select sharing option:",
                            ["Public Link", "ROTRIX Team", "Email"],
                            key=f"share_option_{i}"
                        )
                        email_address = None
                        if share_option == "Email":
                            email_address = st.text_input(
                                "Recipient Email:",
                                key=f"email_input_{i}",
                                placeholder="Enter recipient email"
                            )
                            col_share1, col_share2 = st.columns([1, 1])
                            with col_share1:
                                if st.button("✅ Share", key=f"confirm_share_{i}", use_container_width=True):
                                    if share_option == "Email":
                                        if email_address:
                                            st.success(f"File '{file_name}' shared via Email to: {email_address}")
                                        else:
                                            st.warning("Please enter a recipient email address.")
                                            st.stop()
                                    else:
                                        st.info(f"🔧 {share_option} sharing is a Work in Progress.")
                                    st.session_state.file_share_mode[i] = False
                                    st.rerun()
                            with col_share2:
                                if st.button("❌ Cancel", key=f"cancel_share_{i}", use_container_width=True):
                                    st.session_state.file_share_mode[i] = False
                                    st.rerun()
                        elif share_option in ["Public Link", "ROTRIX Team"]:
                            st.info(f"🔧 {share_option} sharing is a Work in Progress.")
        with actions_col:
            total_files = len(st.session_state.uploaded_files)
            total_size = sum(f.size for f in st.session_state.uploaded_files) / (1024*1024) # MB
            st.markdown(f"""
            <div class="file-stats">
                <h6>📊 File Statistics</h6>
                <div class="stat-value">{total_files}</div>
                <div class="stat-label">Total Files</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="file-stats">
                <h6>💾 Storage</h6>
                <div class="stat-value">{total_size:.1f}</div>
                <div class="stat-label">Total Size (MB)</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("➦ Share All", use_container_width=True):
                st.session_state.share_all_mode = not st.session_state.get("share_all_mode", False)
            if st.session_state.get("share_all_mode", False):
                st.markdown("Share All Files")
                share_all_option = st.selectbox(
                    "Select sharing option for all files:",
                    ["Public Link", "ROTRIX Team", "Email"],
                    key="share_all_option"
                )
                share_all_email = None
                if share_all_option == "Email":
                    share_all_email = st.text_input(
                        "Recipient Email:",
                        key="share_all_email_input",
                        placeholder="Enter recipient email"
                    )
                    col_share_all1, col_share_all2 = st.columns([1, 1])
                    with col_share_all1:
                        if st.button("✅ Confirm Share All", key="confirm_share_all", use_container_width=True):
                            file_names = [f.name for f in st.session_state.uploaded_files]
                            if share_all_option == "Email":
                                if share_all_email:
                                    st.success(f"All files ({', '.join(file_names)}) shared via Email to: {share_all_email}")
                                else:
                                    st.warning("Please enter a recipient email address.")
                                    st.stop()
                            else:
                                st.info(f"🔧 {share_all_option} sharing is a Work in Progress.")
                            st.session_state.share_all_mode = False
                    with col_share_all2:
                        if st.button("❌ Cancel", key="cancel_share_all", use_container_width=True):
                            st.session_state.share_all_mode = False
                elif share_all_option in ["Public Link", "ROTRIX Team"]:
                    st.info(f"🔧 {share_option} sharing is a Work in Progress.")
            if st.button("🗑 Clear All", use_container_width=True):
                st.session_state.uploaded_files.clear()
                st.rerun()
            if st.button("✅ Submit Files for Analysis", type="primary", use_container_width=True):
                st.session_state.files_submitted = True
                st.session_state.show_upload_area = False
                st.rerun()

    else:
        st.markdown("""
        <div style='width: 100%; text-align: center; margin-top: 32px;'>
            <div style='font-size: 48px; margin-bottom: 16px;'>
                <span style='display: inline-block;'>
                    <img src='https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4c1.svg' alt='folder' width='48' style='vertical-align: middle;'/>
                </span>
            </div>
            <div style='font-size: 1.25rem; color: #495057; font-weight: 600; margin-bottom: 6px;'>No files uploaded yet</div>
            <div style='color: #adb5bd; font-size: 1.05rem; margin-bottom: 8px;'>Upload your CSV or PCD files to begin analysis</div>
            <div style='font-size: 12px; color: #ced4da;'>Supported formats: .csv, .pcd</div>
        </div>
        """, unsafe_allow_html=True)

if st.session_state.files_submitted and not st.session_state.show_upload_area:
    import streamlit.components.v1 as components
    tab_index = 0 if st.session_state.get("active_tab", "Single") == "Single" else 1
    components.html(f"""
    <script>
        const tabs = window.parent.document.querySelectorAll('button[kind="tab"]');
        if (tabs.length > {tab_index}) {{
            tabs[{tab_index}].click();
        }}
    </script>
    """, height=0)

    # Sidebar for Parameter Settings
    st.sidebar.markdown("""
        <style>
        [data-testid="stSidebar"] {
            width: 310px;
            min-width: 310px;
        }
        </style>
    """, unsafe_allow_html=True)
    st.sidebar.header("🔧 Select Parameters")

    # File upload
    if "uploaded_files" in st.session_state:
        single_files = st.session_state.uploaded_files
        single_names = [f.name for f in single_files]
    else:
        single_files = []
        single_names = []

    # layer_option = st.radio(
    #     "Select View Type",
    #     ["Single Layer", "Single Part", "Two Layers"],
    #     horizontal=True,
    #     key="layer_option_single"
    # )

    # # 1. Single Layer View
    # if layer_option == "Single Layer":
    selected_single = st.selectbox("Select File to Analyze", ["None"] + single_names, key="single_file_select")
    if selected_single != "None":
        s_file = single_files[single_names.index(selected_single)]
        s_file_ext = os.path.splitext(s_file.name)[-1].lower()
        if hasattr(s_file, 'seek'):
            s_file.seek(0)
        # Load original data
        original_df = load_data(s_file.read(), s_file_ext)
        
        # Initialize rotation angle in session state if not present
        if "rotation_angle_single" not in st.session_state:
            st.session_state.rotation_angle_single = 16
        
        # Store original data in session state if not already present
        if "original_data" not in st.session_state or st.session_state.get("current_file") != selected_single:
            st.session_state.original_data = original_df
            st.session_state.current_file = selected_single
        
        # Apply rotation to the original data
        bit_val = 400 / (2**20 - 1)
        angle_radians = np.radians(st.session_state.rotation_angle_single)
        sin_t, cos_t = np.sin(angle_radians), np.cos(angle_radians)
        
        x_rot = [(d[1] * -bit_val * cos_t - d[0] * bit_val * sin_t) for d in st.session_state.original_data[["x", "y"]].values]
        y_rot = [(d[1] * -bit_val * sin_t + d[0] * bit_val * cos_t) for d in st.session_state.original_data[["x", "y"]].values]
        z_rot = st.session_state.original_data["z"].values
        
        # Create rotated DataFrame
        s_df = pd.DataFrame({
            "x": x_rot,
            "y": y_rot,
            "z": z_rot
        })
        
        st.session_state.s_df = s_df

        st.success(f"✅ File loaded: {selected_single}")

        common_cols = s_df.columns.tolist()
        if common_cols:
            # Always use 'x' and 'y' columns for axes
            x_axis = "x"
            y_axis = "y"

            # Add rotation angle control
            rotation_angle = st.sidebar.number_input("Rotation Angle (°)", 0, 359, st.session_state.rotation_angle_single, 1, key="rotation_angle_input")
            if rotation_angle != st.session_state.rotation_angle_single:
                st.session_state.rotation_angle_single = rotation_angle
                st.rerun()

            colorscale = st.sidebar.selectbox("🎨 Colorscale", ["plasma", "hot", "viridis", "inferno", "turbo"], key="colorscale_single")
            z_threshold = st.sidebar.slider("Z-Score Threshold for Abnormal Points", 1.0, 5.0, 3.0, 0.1, key="z_threshold_single")

            if x_axis != "None" and y_axis != "None":
                    # Initialize axis limits in session state if not present
                    if "x_min_single_persistent" not in st.session_state:
                        x_mean = s_df[x_axis].mean()
                        x_std = s_df[x_axis].std()
                        y_mean = s_df[y_axis].mean()
                        y_std = s_df[y_axis].std()
                        st.session_state.x_min_single_persistent = float(x_mean - 3 * x_std)
                        st.session_state.x_max_single_persistent = float(x_mean + 3 * x_std)
                        st.session_state.y_min_single_persistent = float(y_mean - 3 * y_std)
                        st.session_state.y_max_single_persistent = float(y_mean + 3 * y_std)
                    
                    # X-axis controls in two columns
                    st.sidebar.markdown("### 📊 X-Axis Limits")
                    col_x_min, col_x_max = st.sidebar.columns(2)
                    with col_x_min:
                        x_min = st.number_input("X min", value=st.session_state.x_min_single_persistent, key="x_min_single")
                    with col_x_max:
                        x_max = st.number_input("X max", value=st.session_state.x_max_single_persistent, key="x_max_single")
                    
                    # Y-axis controls in two columns
                    st.sidebar.markdown("### 📊 Y-Axis Limits")
                    col_y_min, col_y_max = st.sidebar.columns(2)
                    with col_y_min:
                        y_min = st.number_input("Y min", value=st.session_state.y_min_single_persistent, key="y_min_single")
                    with col_y_max:
                        y_max = st.number_input("Y max", value=st.session_state.y_max_single_persistent, key="y_max_single")
                    
                    # Reset button
                    if st.sidebar.button("🔄 Reset Axis Limits", type="secondary"):
                        # Reset to default values (3 standard deviations from mean)
                        x_mean = s_df[x_axis].mean()
                        x_std = s_df[x_axis].std()
                        y_mean = s_df[y_axis].mean()
                        y_std = s_df[y_axis].std()
                        st.session_state.x_min_single_persistent = float(x_mean - 3 * x_std)
                        st.session_state.x_max_single_persistent = float(x_mean + 3 * x_std)
                        st.session_state.y_min_single_persistent = float(y_mean - 3 * y_std)
                        st.session_state.y_max_single_persistent = float(y_mean + 3 * y_std)
                        st.rerun()
                    
                    # Check if any axis limits changed and trigger rerun
                    if (x_min != st.session_state.x_min_single_persistent or 
                        x_max != st.session_state.x_max_single_persistent or
                        y_min != st.session_state.y_min_single_persistent or
                        y_max != st.session_state.y_max_single_persistent):
                        st.session_state.x_min_single_persistent = x_min
                        st.session_state.x_max_single_persistent = x_max
                        st.session_state.y_min_single_persistent = y_min
                        st.session_state.y_max_single_persistent = y_max
                        st.rerun()

                    # Video Generation Parameters
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("### 🎥 Video Generation")
                    
                    # Duration only - FPS adapts to data size for complete coverage
                    duration = st.sidebar.number_input("Duration (s)", min_value=1, max_value=60, value=10, key="video_duration")
                    
                    # Fixed video quality - Standard MP4
                    video_quality = "Standard (MP4V)"
                    
                    # FPS will be calculated after data is loaded to match data size
                    # This ensures total_frames = filtered_data_size
                    
                    # Generate Video Button
                    generate_video = st.sidebar.button("🎬 Generate Video", type="primary", key="generate_video")

                    # Detect abnormal points from the rotated data
                    mean_val = s_df["z"].mean()
                    std_val = s_df["z"].std()
                    z_scores = np.abs((s_df["z"] - mean_val) / std_val)
                    abnormal_points = s_df[z_scores > z_threshold]

                    filtered = s_df[(s_df[x_axis] >= x_min) & (s_df[x_axis] <= x_max) &
                                    (s_df[y_axis] >= y_min) & (s_df[y_axis] <= y_max)]

                    # Data is already rotated, use filtered data directly
                    rotated = filtered
                    
                    # Video Generation Logic
                    if generate_video:
                        # Calculate FPS based on filtered data size to ensure total_frames = data_size
                        filtered_data_size = len(rotated)
                        if filtered_data_size > 0:
                            # Calculate FPS so that total_frames = filtered_data_size
                            fps = filtered_data_size / duration
                            total_frames = filtered_data_size
                            
                            # Display calculated parameters
                            st.sidebar.info(f"Duration: {duration}s | Data Points: {filtered_data_size} | FPS: {fps:.1f} | Total frames: {total_frames}")
                            
                            # Progress bar
                        progress_bar = st.sidebar.progress(0)
                        status_text = st.sidebar.empty()
                        
                        with st.spinner("🎥 Generating video with user's plot data..."):
                            # Create temp directory for frames
                            temp_dir = tempfile.mkdtemp()
                            frame_paths = []
                            
                            # Use the actual plot data that user sees (rotated data)
                            plot_data = rotated  # This is the filtered data user sees
                            
                            # Calculate star movement path following actual PCD data points in order
                            # The Star will follow the laser scanning path of the data points
                            data_x = np.array(plot_data["x"])
                            data_y = np.array(plot_data["y"])
                            
                            # Ensure we have data to animate
                            if len(data_x) == 0:
                                st.error("No data points available for video generation. Please check your axis limits.")
                                # Cleanup and continue
                                progress_bar.empty()
                                status_text.empty()
                            else:
                                    # Create path that visits EVERY single data point - one frame per point
                                    # Since total_frames = data_size, each frame visits exactly one point
                                    num_points = len(data_x)
                                    
                                    # Each frame corresponds to one data point
                                    # Frame 0 = Point 0, Frame 1 = Point 1, etc.
                                    star_x_positions = data_x  # Direct mapping: frame index = data point index
                                    star_y_positions = data_y  # Direct mapping: frame index = data point index
                            
                                    # Generate frames with star movement
                                    for frame in range(total_frames):
                                        # Update progress
                                        progress = (frame + 1) / total_frames
                                        progress_bar.progress(progress)
                                        status_text.text(f"Generating frame {frame + 1}/{total_frames}")
                                        
                                        # Calculate current star position based on frame
                                        current_x = star_x_positions[frame]
                                        current_y = star_y_positions[frame]
                                    
                                        # Create frame using matplotlib (no plotly dependency)
                                        frame_path = os.path.join(temp_dir, f"frame_{frame:04d}.png")
                                        
                                        # Create matplotlib figure for this frame
                                        import matplotlib.pyplot as plt
                                        fig, ax = plt.subplots(figsize=(12, 8))
                                        fig.patch.set_facecolor('black')
                                        ax.set_facecolor('black')
                                    
                                        # Track revealed points across all frames (persistent trail effect)
                                        if frame == 0:
                                            # Initialize revealed points for first frame
                                            st.session_state.revealed_points = set()
                                            st.session_state.revealed_abnormal = set()
                                        
                                        # Calculate which data points should be revealed in this frame
                                        new_revealed_points = []
                                        new_revealed_abnormal = []
                                        
                                        # Check which points the star passes through in this frame
                                        for i in range(len(plot_data["x"])):
                                            point_x = plot_data["x"].iloc[i]
                                            point_y = plot_data["y"].iloc[i]
                                            
                                            # Point becomes revealed only if star is exactly at or very close to this specific point
                                            # Use a very small tolerance to ensure only exact matches
                                            if (abs(current_x - point_x) < 0.01 and abs(current_y - point_y) < 0.01) and i not in st.session_state.revealed_points:
                                                new_revealed_points.append(i)
                                                st.session_state.revealed_points.add(i)
                                        
                                        # Check abnormal points for revelation
                                        if not abnormal_points.empty:
                                            for i in range(len(abnormal_points["x"])):
                                                point_x = abnormal_points["x"].iloc[i]
                                                point_y = abnormal_points["y"].iloc[i]
                                                
                                                # Abnormal point becomes revealed only if star is exactly at this point
                                                if (abs(current_x - point_x) < 0.01 and abs(current_y - point_y) < 0.01) and i not in st.session_state.revealed_abnormal:
                                                    new_revealed_abnormal.append(i)
                                                    st.session_state.revealed_abnormal.add(i)
                                        
                                        # Plot all revealed data points
                                        if st.session_state.revealed_points:
                                            revealed_x = plot_data["x"].iloc[list(st.session_state.revealed_points)]
                                            revealed_y = plot_data["y"].iloc[list(st.session_state.revealed_points)]
                                            revealed_z = plot_data["z"].iloc[list(st.session_state.revealed_points)]
                                            
                                            scatter = ax.scatter(revealed_x, revealed_y, 
                                                               c=revealed_z, cmap='plasma', 
                                                               s=20, alpha=0.8, edgecolors='none')
                                        else:
                                            # Create empty scatter for colorbar
                                            scatter = ax.scatter([], [], c=[], cmap='plasma')
                                        
                                        # Plot all revealed abnormal points
                                        if st.session_state.revealed_abnormal and not abnormal_points.empty:
                                            revealed_abnormal_x = abnormal_points["x"].iloc[list(st.session_state.revealed_abnormal)]
                                            revealed_abnormal_y = abnormal_points["y"].iloc[list(st.session_state.revealed_abnormal)]
                                            ax.scatter(revealed_abnormal_x, revealed_abnormal_y, 
                                                      c='red', s=40, alpha=0.8, edgecolors='white', linewidth=1)
                                        
                                        # Plot star at current position
                                        ax.scatter([current_x], [current_y], c='yellow', s=400, 
                                                  marker='*', edgecolors='black', linewidth=2, alpha=1.0)
                                        
                                        # Set plot limits and styling
                                        ax.set_xlim(x_min, x_max)
                                        ax.set_ylim(y_min, y_max)
                                        # Removed axis labels for cleaner video
                                        ax.set_xlabel('')
                                        ax.set_ylabel('')
                                        ax.set_title(f'Frame {frame + 1}/{total_frames}', 
                                                   color='white', fontsize=14)
                                        # Hiding tick labels and marks
                                        ax.set_xticklabels([])
                                        ax.set_yticklabels([])
                                        ax.tick_params(axis='both', which='both', length=0)

                                        # Create colorbar but hide it visually (keeps colorscale mapping)
                                        cbar = plt.colorbar(scatter, ax=ax)
                                        cbar.set_label('Temperature', color='white')
                                        cbar.ax.tick_params(colors='white')
                                        # Hiding the colorbar visually while keeping the colorscale functionality
                                        cbar.remove()
                                        
                                        # Remove grid
                                        ax.grid(False)
                                        
                                        # Save frame
                                        plt.tight_layout()
                                        plt.savefig(frame_path, dpi=100, bbox_inches='tight', 
                                                   facecolor='black', edgecolor='none')
                                        plt.close()
                                        
                                        frame_paths.append(frame_path)
                            
                            # Create video using OpenCV with optimized function
                            video_path = os.path.join(temp_dir, "animation.mp4")
                            
                            # Progress callback for video creation
                            def video_progress_callback(current, total):
                                progress = current / total
                                progress_bar.progress(progress)
                                status_text.text(f"Creating video: {current}/{total} frames")
                            
                            # Create video using the optimized function
                            success, message = create_video_from_frames(
                                frame_paths, video_path, fps, video_quality, video_progress_callback
                            )
                            
                            if not success:
                                st.error(f"Video creation failed: {message}")
                                # Cleanup and continue
                                shutil.rmtree(temp_dir)
                                progress_bar.empty()
                                status_text.empty()
                            else:
                                # Clear progress indicators
                                progress_bar.empty()
                                status_text.empty()
                                
                                # Success message
                                st.sidebar.success("✅ Video generated successfully!")
                                
                                try:
                                    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                                        with open(video_path, 'rb') as f:
                                            video_bytes = f.read()
                                        
                                        # Download button in sidebar
                                        st.sidebar.download_button(
                                            label="📥 Download Video",
                                            data=video_bytes,
                                            file_name="point_cloud_animation.mp4",
                                            mime="video/mp4"
                                        )
                                    else:
                                        st.sidebar.error("Video file is empty or doesn't exist")
                                except Exception as e:
                                    st.sidebar.error(f"Error preparing download: {e}")
                                
                                # Cleanup
                                shutil.rmtree(temp_dir)

                    # Use z-axis for abnormality detection from the rotated data
                    mean_val = rotated["z"].mean()
                    std_val = rotated["z"].std()
                    z_scores = np.abs((rotated["z"] - mean_val) / std_val)
                    abnormal_points = rotated[z_scores > z_threshold]
                    st.write("Debug: Filtered data size:", len(filtered))
                    st.write("Debug: Abnormal points found:", len(abnormal_points))

                    if x_axis == "None" or y_axis == "None":
                        st.info("📌 Please select valid X and Y axes.")
                    elif filtered.empty:
                        st.warning("No data to plot. Check your filters or axis selections.")
                    else:
                        # 👉 Side-by-side layout for plot and abnormality table
                        col1, col2 = st.columns([3, 1])  # 3/4 plot, 1/4 table

                        with col1:
                            st.markdown("### 📊 Single File Scatter Plot")
                            
                            # Create a container for the plot to ensure proper sizing
                            plot_container = st.container()
                            
                            with plot_container:
                                # Create traces list with main data first, then abnormal points on top
                                traces = []
                            
                                # Add main data trace using go.Scattergl for better performance
                                traces.append(go.Scattergl(
                                x=rotated["x"], 
                                y=rotated["y"],
                                mode='markers',
                                marker=dict(
                                    color=rotated["z"],
                                    colorscale=colorscale,
                                    size=4,
                                    showscale=True,
                                    colorbar=dict(title="Temp", tickfont=dict(size=12)),
                                    cmin=650,
                                    cmax=1000
                                ),
                                name="Data Points"
                            ))
                            
                            # Add abnormal points trace on top
                            if not abnormal_points.empty:
                                traces.append(go.Scattergl(
                                    x=abnormal_points["x"],
                                    y=abnormal_points["y"],
                                    mode='markers',
                                    name="Abnormal",
                                    marker=dict(color='red', size=8, opacity=0.8)
                                ))
                            
                            fig = go.Figure(data=traces)
                            fig.update_layout(
                                height=600, 
                                title="",
                                plot_bgcolor='black',
                                paper_bgcolor='black',
                                xaxis=dict(
                                    showgrid=False,
                                    showticklabels=False,
                                    title="",
                                    zeroline=False
                                ),
                                yaxis=dict(
                                    showgrid=False,
                                    showticklabels=False,
                                    title="",
                                    zeroline=False
                                ),
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=-0.2,
                                    xanchor="center",
                                    x=0.5,
                                    font=dict(color="white")
                                )
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            st.subheader("⚠ Abnormality Detection")
                            st.write(f"Detected {len(abnormal_points)} abnormal points.")
                            st.dataframe(
                                abnormal_points[
                                ["x", "y", "z"]]
                            )
                        
                        # Add histogram below the plot
                        st.markdown("### 📊 PCD Data Distribution Histogram")
                        
                        # Create histogram from the filtered data
                        fig_hist = go.Figure(data=go.Histogram(
                            x=filtered["z"],
                            nbinsx=50,
                            marker_color='lightblue',
                            opacity=0.7
                        ))
                        
                        fig_hist.update_layout(
                            title="Z-Value Distribution",
                            xaxis_title="Z Values",
                            yaxis_title="Frequency",
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(color='black'),
                            xaxis=dict(
                                gridcolor='lightgray',
                                zerolinecolor='black'
                            ),
                            yaxis=dict(
                                gridcolor='lightgray',
                                zerolinecolor='black'
                            ),
                            height=300
                        )
                        # Using empty columns to set histogram width to 3/4 of the page
                        hist_col, empty_col = st.columns([3, 1])
                        with hist_col:
                            st.plotly_chart(fig_hist, use_container_width=True)

            else:
                st.warning("This file doesn't contain valid numeric columns.")

        # # 2. Single Part View
        # elif layer_option == "Single Part":
        #     st.markdown("#### 📁 Select Source for Part Files")

        #     source_type = st.radio("Choose Source Type", ["Local Folder", "Zip Upload", "GitHub URL"], horizontal=True)
        #     folder_path = st.text_input("📂 Enter Folder Path or URL", value="", key="folder_path_input")

        #     files = []
        #     if folder_path:
        #         if source_type == "Local Folder" and os.path.exists(folder_path):
        #             # Recursively find all .pcd files
        #             for root, _, filenames in os.walk(folder_path):
        #                 for f in filenames:
        #                     if f.endswith(".pcd"):
        #                         files.append(os.path.join(root, f))
        #         elif source_type == "Zip Upload" and folder_path:
        #             zip_file = st.file_uploader("📦 Upload a ZIP file", type=["zip"], key="zip_single_part")
        #             if zip_file:
        #                 zip_dir = "/mnt/data/unzipped_single_part"
        #                 os.makedirs(zip_dir, exist_ok=True)
        #                 with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        #                     zip_ref.extractall(zip_dir)
        #                 for root, _, filenames in os.walk(zip_dir):
        #                     for f in filenames:
        #                         if f.endswith(".pcd"):
        #                             files.append(os.path.join(root, f))
        #                 folder_path = zip_dir
        #         elif source_type == "GitHub URL":
        #             result = process_url(folder_path)
        #             if result:
        #                 files = [os.path.join(tempfile.gettempdir(), fname) for fname in result.keys()]
        #                 for fname, (content, ext) in result.items():
        #                     with open(os.path.join(tempfile.gettempdir(), fname), 'wb') as f:
        #                         f.write(content)

        #         if files:
        #             files.sort()
        #             total = len(files)
        #             one_third = total // 2
        #             bottom_files = files[:3]
        #             middle_files = files[one_third-2:one_third+1]
        #             top_files = files[-3:]

        #             # Initialize session state for parameters with fixed two-decimal-place defaults
        #             if "x_axis_single_part" not in st.session_state:
        #                 st.session_state.x_axis_single_part = "x"
        #             if "y_axis_single_part" not in st.session_state:
        #                 st.session_state.y_axis_single_part = "y"
        #             if "colorscale_single_part" not in st.session_state:
        #                 st.session_state.colorscale_single_part = "plasma"
        #             if "z_threshold_single_part" not in st.session_state:
        #                 st.session_state.z_threshold_single_part = 3.0
        #             if "x_min_single_part" not in st.session_state:
        #                 st.session_state.x_min_single_part = -55.20
        #             if "x_max_single_part" not in st.session_state:
        #                 st.session_state.x_max_single_part = -37.10
        #             if "y_min_single_part" not in st.session_state:
        #                 st.session_state.y_min_single_part = 22.65
        #             if "y_max_single_part" not in st.session_state:
        #                 st.session_state.y_max_single_part = 40.00

        #             # Always use 'x' and 'y' columns for axes
        #             x_axis = "x"
        #             y_axis = "y"
        #             colorscale = st.sidebar.selectbox("🎨 Colorscale", ["plasma", "hot", "viridis", "inferno", "turbo"], key="colorscale_single_part")
        #             z_threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 5.0, st.session_state.z_threshold_single_part, 0.1, key="z_threshold_single_part")
                    
        #             # X-axis controls in two columns
        #             st.sidebar.markdown("### 📊 X-Axis Limits")
        #             col_x_min, col_x_max = st.sidebar.columns(2)
        #             with col_x_min:
        #                 x_min = st.number_input("X min", value=st.session_state.x_min_single_part, format="%.2f", key="x_min_single_part")
        #             with col_x_max:
        #                 x_max = st.number_input("X max", value=st.session_state.x_max_single_part, format="%.2f", key="x_max_single_part")
                    
        #             # Y-axis controls in two columns
        #             st.sidebar.markdown("### 📊 Y-Axis Limits")
        #             col_y_min, col_y_max = st.sidebar.columns(2)
        #             with col_y_min:
        #                 y_min = st.number_input("Y min", value=st.session_state.y_min_single_part, format="%.2f", key="y_min_single_part")
        #             with col_y_max:
        #                 y_max = st.number_input("Y max", value=st.session_state.y_max_single_part, format="%.2f", key="y_max_single_part")
                    
        #             # Reset button
        #             if st.sidebar.button("🔄 Reset Axis Limits", type="secondary", key="reset_single_part"):
        #                 # Reset to default values
        #                 st.session_state.x_min_single_part = -55.20
        #                 st.session_state.x_max_single_part = -37.10
        #                 st.session_state.y_min_single_part = 22.65
        #                 st.session_state.y_max_single_part = 40.00
        #                 st.rerun()

        #             # Set plot_ready flag when both axes are selected
        #             if x_axis != "None" and y_axis != "None":
        #                 st.session_state.plot_ready_single_part = True
        #             else:
        #                 st.session_state.plot_ready_single_part = False

        #             col_btm, col_mid, col_top = st.columns(3)
        #             if st.session_state.plot_ready_single_part:
        #                 plot_xyz_like_files(col_btm, "Bottom", bottom_files,
        #                                 x_axis, y_axis, x_min, x_max, y_min, y_max, z_threshold, "z", colorscale)
        #                 plot_xyz_like_files(col_mid, "Middle", middle_files,
        #                                 x_axis, y_axis, x_min, x_max, y_min, y_max, z_threshold, "z", colorscale)
        #                 plot_xyz_like_files(col_top, "Top", top_files,
        #                                 x_axis, y_axis, x_min, x_max, y_min, y_max, z_threshold, "z", colorscale)
                        
        #             else:
        #                 col_btm.warning("Please select both X-Axis and Y-Axis in the parameters to view the plot.")
        #                 col_mid.warning("Please select both X-Axis and Y-Axis in the parameters to view the plot.")
        #                 col_top.warning("Please select both X-Axis and Y-Axis in the parameters to view the plot.")

        #         else:
        #             st.warning("No files found in the specified folder or URL.")

        #     elif folder_path and not files:
        #         st.warning("Folder or URL is accessible, but no .pcd files were found.")

        # # 3. Two Layers View
        # elif layer_option == "Two Layers":
        #     col1, col2 = st.columns(2)
        #     with col1:
        #         selected_file1 = st.selectbox("Select First File", ["None"] + single_names, key="file1_select")
        #     with col2:
        #         selected_file2 = st.selectbox("Select Second File", ["None"] + single_names, key="file2_select")

        #     if selected_file1 != "None" and selected_file2 != "None" and selected_file1 != selected_file2:
        #         file1 = single_files[single_names.index(selected_file1)]
        #         file2 = single_files[single_names.index(selected_file2)]

        #         if hasattr(file1, 'seek'):
        #             file1.seek(0)
        #         if hasattr(file2, 'seek'):
        #             file2.seek(0)

        #         # Load sample data to determine intelligent bounds
        #         sample_df1 = load_custom_xyz_file(file1) if file1 else pd.DataFrame(columns=["x", "y", "z"])
        #         sample_df2 = load_custom_xyz_file(file2) if file2 else pd.DataFrame(columns=["x", "y", "z"])
        #         if not sample_df1.empty and not sample_df2.empty:
        #             # Initialize session state with default axis values (always x and y)
        #             if "x_axis_two_layers" not in st.session_state:
        #                 st.session_state.x_axis_two_layers = "x"
        #             if "y_axis_two_layers" not in st.session_state:
        #                 st.session_state.y_axis_two_layers = "y"
        #             if "colorscale_two_layers" not in st.session_state:
        #                 st.session_state.colorscale_two_layers = "plasma"
        #             if "z_threshold_two_layers" not in st.session_state:
        #                 st.session_state.z_threshold_two_layers = 3.0
        #             # Always use 'x' and 'y' for initial bounds calculation
        #             default_x_axis = "x"
        #             default_y_axis = "y"
        #             if "x_min_two_layers" not in st.session_state:
        #                 st.session_state.x_min_two_layers = sample_df1[default_x_axis].min() - (sample_df1[default_x_axis].max() - sample_df1[default_x_axis].min()) * 0.1
        #             if "x_max_two_layers" not in st.session_state:
        #                 st.session_state.x_max_two_layers = sample_df1[default_x_axis].max() + (sample_df1[default_x_axis].max() - sample_df1[default_x_axis].min()) * 0.1
        #             if "y_min_two_layers" not in st.session_state:
        #                 st.session_state.y_min_two_layers = sample_df1[default_y_axis].min() - (sample_df1[default_y_axis].max() - sample_df1[default_y_axis].min()) * 0.1
        #             if "y_max_two_layers" not in st.session_state:
        #                 st.session_state.y_max_two_layers = sample_df1[default_y_axis].max() + (sample_df1[default_y_axis].max() - sample_df1[default_y_axis].min()) * 0.1

        #             # Always use 'x' and 'y' columns for axes
        #             x_axis = "x"
        #             y_axis = "y"
        #             colorscale = st.sidebar.selectbox("🎨 Colorscale", ["plasma", "hot", "viridis", "inferno", "turbo"], key="colorscale_two_layers")
        #             z_threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 5.0, st.session_state.z_threshold_two_layers, 0.1, key="z_threshold_two_layers")
                    
        #             # X-axis controls in two columns
        #             st.sidebar.markdown("### 📊 X-Axis Limits")
        #             col_x_min, col_x_max = st.sidebar.columns(2)
        #             with col_x_min:
        #                 x_min = st.number_input("X min", value=st.session_state.x_min_two_layers, format="%.2f", key="x_min_two_layers")
        #             with col_x_max:
        #                 x_max = st.number_input("X max", value=st.session_state.x_max_two_layers, format="%.2f", key="x_max_two_layers")
                    
        #             # Y-axis controls in two columns
        #             st.sidebar.markdown("### 📊 Y-Axis Limits")
        #             col_y_min, col_y_max = st.sidebar.columns(2)
        #             with col_y_min:
        #                 y_min = st.number_input("Y min", value=st.session_state.y_min_two_layers, format="%.2f", key="y_min_two_layers")
        #             with col_y_max:
        #                 y_max = st.number_input("Y max", value=st.session_state.y_max_two_layers, format="%.2f", key="y_max_two_layers")
                    
        #             # Reset button
        #             if st.sidebar.button("🔄 Reset Axis Limits", type="secondary", key="reset_two_layers"):
        #                 # Reset to default values based on sample data
        #                 st.session_state.x_min_two_layers = sample_df1[default_x_axis].min() - (sample_df1[default_x_axis].max() - sample_df1[default_x_axis].min()) * 0.1
        #                 st.session_state.x_max_two_layers = sample_df1[default_x_axis].max() + (sample_df1[default_x_axis].max() - sample_df1[default_x_axis].min()) * 0.1
        #                 st.session_state.y_min_two_layers = sample_df1[default_y_axis].min() - (sample_df1[default_y_axis].max() - sample_df1[default_y_axis].min()) * 0.1
        #                 st.session_state.y_max_two_layers = sample_df1[default_y_axis].max() + (sample_df1[default_y_axis].max() - sample_df1[default_y_axis].min()) * 0.1
        #                 st.rerun()
                    
        #             # Check if any axis limits changed and trigger rerun
        #             if (x_min != st.session_state.x_min_two_layers or 
        #                 x_max != st.session_state.x_max_two_layers or
        #                 y_min != st.session_state.y_min_two_layers or
        #                 y_max != st.session_state.y_max_two_layers):
        #                 st.session_state.x_min_two_layers = x_min
        #                 st.session_state.x_max_two_layers = x_max
        #                 st.session_state.y_min_two_layers = y_min
        #                 st.session_state.y_max_two_layers = y_max
        #                 st.rerun()

        #             if x_axis and y_axis:
        #                 plot_two_layers(col1, col2, file1, file2, x_axis, y_axis, z_threshold, x_min, x_max, y_min, y_max, colorscale)
        #             else:
        #                 col1.warning("Please select both X-Axis and Y-Axis in the parameters to view the plot.")
        #                 col2.warning("Please select both X-Axis and Y-Axis in the parameters to view the plot.")
        #         else:
        #             col1.warning("Error loading one or both files.")
        #             col2.warning("Error loading one or both files.")
        #     else:
        #         st.info("Please select two different files to compare.")
