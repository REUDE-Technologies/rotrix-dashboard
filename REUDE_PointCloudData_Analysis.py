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
import zipfile

# Function to load custom XYZ-like file using readlines
def load_custom_xyz_file(path):
    try:
        with open(path, 'r') as f:
            lines = f.readlines()

        # Filter out comment lines and empty lines
        data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]

        # Parse numeric lines with 3 or more numbers
        data = []
        for line in data_lines:
            numbers = list(map(float, line.split()))
            if len(numbers) >= 3:
                data.append(numbers[:3])  # Use only first 3 for x, y, z

        if not data:
            raise ValueError("No valid 3D points found in file.")

        df = pd.DataFrame(data, columns=["x", "y", "z"])
        return df
    except Exception as e:
        raise ValueError(f"Failed to parse XYZ-like file: {e}")

# Function to plot the PCD files as 2D (x, y) with z as color
def plot_xyz_like_files(col, label, selected_files, x_axis, y_axis, x_min, x_max, y_min, y_max, sample_size, z_threshold):
    col.markdown(f"### {label} Layers")
    for path in selected_files:
        try:
            df = load_custom_xyz_file(path)

            # Filter data based on user parameters
            df_filtered = df[(df[x_axis] >= x_min) & (df[x_axis] <= x_max) &
                             (df[y_axis] >= y_min) & (df[y_axis] <= y_max)]
            if len(df_filtered) > sample_size:
                df_filtered = df_filtered.sample(n=sample_size, random_state=42)

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

            fig = go.Figure(data=go.Scattergl(
                x=df_filtered[x_axis], y=df_filtered[y_axis],
                mode='markers',  # Changed to scatter plot
                marker=dict(
                    color=df_filtered["z"],
                    colorscale='Viridis',
                    size=2,
                    showscale=True,
                    colorbar=dict(title="Z Height")
                )
            ))

            if not abnormal_points.empty:
                fig.add_trace(go.Scattergl(x=abnormal_points[x_axis], y=abnormal_points[y_axis],
                                          mode='markers', marker=dict(color='red', size=6),
                                          name="Abnormal"))

            fig.update_layout(
                title=os.path.basename(path),
                height=400,
                xaxis=dict(showgrid=True, zeroline=False),
                yaxis=dict(showgrid=True, zeroline=False),
                margin=dict(l=10, r=10, b=10, t=30),
            )
            col.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            col.warning(f"{os.path.basename(path)} - error: {e}")

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
            help="Drag and drop files here or click to browse")
    if uploaded_files:
        new_files_added = False
        existing_names = [f.name for f in st.session_state.uploaded_files]
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in existing_names:
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
                                # st.info("PCD preview: Only file name, size, and type shown.")
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
                        st.markdown("✏ Rename File**")
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
                        st.markdown("*➦ Share File*")
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
                st.markdown("*Share All Files*")
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

    # === NOW move your full tab layout code here ===
    main_tab1, main_tab2 = st.tabs(["🧩 Single", "📊 Comparative"])
    with main_tab1:
        st.markdown("### 🧩 Single Part View")

        # File upload
        if "uploaded_files" in st.session_state:
            single_files = st.session_state.uploaded_files
            single_names = [f.name for f in single_files]
        else:
            single_files = []
            single_names = []

        layer_option = st.radio(
            "Select View Type",
            ["Single Layer", "Single Part", "Two Layer/Part"],
            horizontal=True,
            key="layer_option_single"
        )

        # 1. Single Layer View
        if layer_option == "Single Layer":
            selected_single = st.selectbox("Select File to Analyze", ["None"] + single_names, key="single_file_select")
            if selected_single != "None":
                s_file = single_files[single_names.index(selected_single)]
                s_file_ext = os.path.splitext(s_file.name)[-1].lower()
                if hasattr(s_file, 'seek'):
                    s_file.seek(0)
                st.session_state.s_df = load_data(s_file.read(), s_file_ext)
                s_df = st.session_state.s_df

                st.success(f"✅ File loaded: {selected_single}")

                common_cols = s_df.columns.tolist()
                if common_cols:
                    col1, col2 = st.columns([0.25, 0.75])
                    with col1:
                        st.markdown("#### 📈 Parameters")
                        x_axis = st.selectbox("X-Axis", ["None"] + common_cols, key="x_axis_single")
                        y_axis = st.selectbox("Y-Axis", ["None"] + common_cols, key="y_axis_single")
                        sample_size = st.slider("Sample Size (for large datasets)", 1000, 100000, 10000, 1000)
                        z_threshold = st.slider("Z-Score Threshold for Abnormal Points", 1.0, 5.0, 3.0, 0.1)

                        if x_axis != "None" and y_axis != "None":
                            x_min = st.number_input("X min", value=float(s_df[x_axis].min()), key="x_min_single")
                            x_max = st.number_input("X max", value=float(s_df[x_axis].max()), key="x_max_single")
                            y_min = st.number_input("Y min", value=float(s_df[y_axis].min()), key="y_min_single")
                            y_max = st.number_input("Y max", value=float(s_df[y_axis].max()), key="y_max_single")

                            filtered = s_df[(s_df[x_axis] >= x_min) & (s_df[x_axis] <= x_max) &
                                            (s_df[y_axis] >= y_min) & (s_df[y_axis] <= y_max)]
                            if len(filtered) > sample_size:
                                filtered = filtered.sample(n=sample_size, random_state=42)

                            mean_val = filtered[y_axis].mean()
                            std_val = filtered[y_axis].std()
                            z_scores = np.abs((filtered[y_axis] - mean_val) / std_val)
                            abnormal_points = filtered[z_scores > z_threshold]
                            st.write("Debug: Filtered data size:", len(filtered))  # Debug check
                        else:
                            st.session_state.single_plot_ready = False

                    with col2:
                        if x_axis == "None" or y_axis == "None":
                            st.info("📌 Please select valid X and Y axes.")
                        elif filtered.empty:
                            st.warning("No data to plot. Check your filters or axis selections.")
                        else:
                            st.markdown("### 📊 Single File Scatter Plot")
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=filtered[x_axis], y=filtered[y_axis],
                                                     mode='markers', name="Data", marker=dict(color='skyblue', size=5)))
                            fig.add_trace(go.Scatter(x=[filtered[x_axis].min(), filtered[x_axis].max()],
                                                     y=[mean_val, mean_val], mode='lines',
                                                     name=f"Mean ({mean_val:.2f})",
                                                     line=dict(color='green', dash='dash')))
                            fig.add_trace(go.Scatter(x=[filtered[x_axis].min(), filtered[x_axis].max()],
                                                     y=[mean_val + std_val, mean_val + std_val],
                                                     mode='lines', name=f"+1 SD",
                                                     line=dict(color='blue', dash='dot')))
                            fig.add_trace(go.Scatter(x=[filtered[x_axis].min(), filtered[x_axis].max()],
                                                     y=[mean_val - std_val, mean_val - std_val],
                                                     mode='lines', name=f"-1 SD",
                                                     line=dict(color='blue', dash='dot')))
                            if not abnormal_points.empty:
                                fig.add_trace(go.Scatter(x=abnormal_points[x_axis], y=abnormal_points[y_axis],
                                                         mode='markers', name="Abnormal",
                                                         marker=dict(color='red', size=6)))
                            fig.update_layout(height=600, width=1000, title="Scatter Plot with Statistics")
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("This file doesn't contain valid numeric columns.")

        # 2. Single Part View
        elif layer_option == "Single Part":
            st.markdown("#### 📁 Select Source for Part Files")

            source_type = st.radio("Choose Source Type", ["Local Folder", "Zip Upload", "GitHub URL"], horizontal=True)
            folder_path = st.text_input("📂 Enter Folder Path or URL", value="", key="folder_path_input")

            files = []
            if folder_path:
                if source_type == "Local Folder" and os.path.exists(folder_path):
                    # Recursively find all .pcd files
                    for root, _, filenames in os.walk(folder_path):
                        for f in filenames:
                            if f.endswith(".pcd"):
                                files.append(os.path.join(root, f))
                elif source_type == "Zip Upload" and folder_path:
                    zip_file = st.file_uploader("📦 Upload a ZIP file", type=["zip"], key="zip_single_part")
                    if zip_file:
                        zip_dir = "/mnt/data/unzipped_single_part"
                        os.makedirs(zip_dir, exist_ok=True)
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            zip_ref.extractall(zip_dir)
                        for root, _, filenames in os.walk(zip_dir):
                            for f in filenames:
                                if f.endswith(".pcd"):
                                    files.append(os.path.join(root, f))
                        folder_path = zip_dir
                elif source_type == "GitHub URL":
                    result = process_url(folder_path)
                    if result:
                        files = [os.path.join(tempfile.gettempdir(), fname) for fname in result.keys()]
                        for fname, (content, ext) in result.items():
                            with open(os.path.join(tempfile.gettempdir(), fname), 'wb') as f:
                                f.write(content)

                if files:
                    files.sort()
                    total = len(files)
                    one_third = total // 3
                    bottom_files = files[:one_third][:3]
                    middle_files = files[one_third:2 * one_third][:3]
                    top_files = files[2 * one_third:][:3]

                    # Initialize session state for parameters with finite defaults
                    if "x_axis_single_part" not in st.session_state:
                        st.session_state.x_axis_single_part = "None"
                    if "y_axis_single_part" not in st.session_state:
                        st.session_state.y_axis_single_part = "None"
                    if "sample_size_single_part" not in st.session_state:
                        st.session_state.sample_size_single_part = 10000
                    if "z_threshold_single_part" not in st.session_state:
                        st.session_state.z_threshold_single_part = 3.0
                    if "x_min_single_part" not in st.session_state:
                        st.session_state.x_min_single_part = -1e10  # Finite large negative value
                    if "x_max_single_part" not in st.session_state:
                        st.session_state.x_max_single_part = 1e10   # Finite large positive value
                    if "y_min_single_part" not in st.session_state:
                        st.session_state.y_min_single_part = -1e10  # Finite large negative value
                    if "y_max_single_part" not in st.session_state:
                        st.session_state.y_max_single_part = 1e10   # Finite large positive value

                    col_lhs, col_rhs = st.columns([0.75, 0.25])
                    with col_lhs:
                        col_btm, col_mid, col_top = st.columns(3)
                        # Only plot if both X and Y axes are selected and plot is ready
                        if st.session_state.plot_ready_single_part and st.session_state.x_axis_single_part != "None" and st.session_state.y_axis_single_part != "None":
                            plot_xyz_like_files(col_btm, "Bottom", bottom_files,
                                              st.session_state.x_axis_single_part, st.session_state.y_axis_single_part,
                                              st.session_state.x_min_single_part,
                                              st.session_state.x_max_single_part,
                                              st.session_state.y_min_single_part,
                                              st.session_state.y_max_single_part,
                                              st.session_state.sample_size_single_part, st.session_state.z_threshold_single_part)
                            plot_xyz_like_files(col_mid, "Middle", middle_files,
                                              st.session_state.x_axis_single_part, st.session_state.y_axis_single_part,
                                              st.session_state.x_min_single_part,
                                              st.session_state.x_max_single_part,
                                              st.session_state.y_min_single_part,
                                              st.session_state.y_max_single_part,
                                              st.session_state.sample_size_single_part, st.session_state.z_threshold_single_part)
                            plot_xyz_like_files(col_top, "Top", top_files,
                                              st.session_state.x_axis_single_part, st.session_state.y_axis_single_part,
                                              st.session_state.x_min_single_part,
                                              st.session_state.x_max_single_part,
                                              st.session_state.y_min_single_part,
                                              st.session_state.y_max_single_part,
                                              st.session_state.sample_size_single_part, st.session_state.z_threshold_single_part)
                        else:
                            col_btm.warning("Please select both X-Axis and Y-Axis in the parameters to view the plot.")
                            col_mid.warning("Please select both X-Axis and Y-Axis in the parameters to view the plot.")
                            col_top.warning("Please select both X-Axis and Y-Axis in the parameters to view the plot.")

                    with col_rhs:
                        st.markdown("#### 📈 Parameters")
                        x_axis = st.selectbox("X-Axis", ["None", "x", "y", "z"], key="x_axis_single_part", index=0)
                        y_axis = st.selectbox("Y-Axis", ["None", "x", "y", "z"], key="y_axis_single_part", index=0)
                        sample_size = st.slider("Sample Size", 1000, 100000, st.session_state.sample_size_single_part, 1000, key="sample_size_single_part")
                        z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, st.session_state.z_threshold_single_part, 0.1, key="z_threshold_single_part")
                        x_min = st.number_input("X min", value=st.session_state.x_min_single_part, key="x_min_single_part")
                        x_max = st.number_input("X max", value=st.session_state.x_max_single_part, key="x_max_single_part")
                        y_min = st.number_input("Y min", value=st.session_state.y_min_single_part, key="y_min_single_part")
                        y_max = st.number_input("Y max", value=st.session_state.y_max_single_part, key="y_max_single_part")

                        # Set plot_ready flag when both axes are selected
                        if st.session_state.x_axis_single_part != "None" and st.session_state.y_axis_single_part != "None":
                            st.session_state.plot_ready_single_part = True
                        elif st.session_state.x_axis_single_part == "None" or st.session_state.y_axis_single_part == "None":
                            st.session_state.plot_ready_single_part = False

                elif folder_path and not files:
                    st.warning("Folder or URL is accessible, but no .pcd files were found.")

        # 3. Two Layer/Part View
        elif layer_option == "Two Layer/Part":
            st.warning("Two Layer/Part view is not implemented yet.")

    with main_tab2:
        st.markdown("### 📊 Comparative Analysis")

        benchmark_files = st.session_state.uploaded_files
        benchmark_names = [f.name for f in benchmark_files]

        col3, col4 = st.columns(2)
        with col3:
            selected_bench = st.selectbox("Select Abirami File", ["None"] + benchmark_names, key="bench_select")
            if selected_bench != "None":
                b_file = benchmark_files[benchmark_names.index(selected_bench)]
                b_file_ext = os.path.splitext(b_file.name)[-1].lower()
                if hasattr(b_file, 'seek'):
                    b_file.seek(0)
                st.session_state.b_df = load_data(b_file.read(), b_file_ext)
            else:
                st.session_state.b_df = None

        with col4:
            selected_val = st.selectbox("Select Keerthi File", ["None"] + benchmark_names, key="val_select")
            if selected_val != "None":
                v_file = benchmark_files[benchmark_names.index(selected_val)]
                v_file_ext = os.path.splitext(v_file.name)[-1].lower()
                if hasattr(v_file, 'seek'):
                    v_file.seek(0)
                st.session_state.v_df = load_data(v_file.read(), v_file_ext)
            else:
                st.session_state.v_df = None

        b_df = st.session_state.get("b_df")
        v_df = st.session_state.get("v_df")

        if b_df is not None and v_df is not None:
            common_cols = list(set(b_df.columns) & set(v_df.columns))
            if common_cols:
                col1, col2 = st.columns([0.25, 0.75])
                with col1:
                    st.markdown("#### 📈 Parameters")
                    x_axis = st.selectbox("X-Axis", ["None"] + common_cols, key="x_axis_select")
                    y_axis = st.selectbox("Y-Axis", ["None"] + common_cols, key="y_axis_select")
                    sample_size = st.slider("Sample Size (for large datasets)", 1000, 100000, 10000, 1000,
                                            help="Reduce the number of points to plot for performance.")
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

                        merged = pd.merge(b_filtered, v_filtered, on=x_axis, suffixes=('_benchmark', '_validation'),
                                          how='inner')
                        st.write("Debug: Merged data size:", len(merged))  # Debug check

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
                                st.warning(
                                    f"Column {val_col} contains non-numeric data. Skipping abnormality detection.")
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
                        st.markdown("### 🧮 Plot Visualization")
                        fig = make_subplots(rows=2, cols=1, subplot_titles=["Abirami", "Keerthishree"],
                                            shared_yaxes=True)

                        # Abirami plot
                        fig.add_trace(
                            go.Scatter(x=merged[x_axis], y=merged[f"{y_axis}_benchmark"], mode='markers', name="Abirami",
                                       marker=dict(color='skyblue', size=5)),
                            row=1, col=1)

                        # Keerthishree plot with mean, std, and abnormal points
                        fig.add_trace(
                            go.Scatter(x=merged[x_axis], y=merged[f"{y_axis}_validation"], mode='markers', name="Keerthishree",
                                       marker=dict(color='orange', size=5)),
                            row=2, col=1)
                        if not abnormal_points.empty:
                            fig.add_trace(
                                go.Scatter(x=abnormal_points[x_axis], y=abnormal_points[f"{y_axis}_validation"],
                                           mode='markers', marker=dict(color='red', size=6),
                                           name="Abnormal"), row=2, col=1)

                        # Add mean line
                        if f"{y_axis}_validation" in merged.columns and pd.api.types.is_numeric_dtype(
                                merged[f"{y_axis}_validation"]):
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

# Helper function for abnormality detection
def detect_abnormalities(series, threshold=3.0):
    if series.empty or not pd.api.types.is_numeric_dtype(series):
        return pd.Series([False]), pd.Series([0.0])
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold, z_scores
