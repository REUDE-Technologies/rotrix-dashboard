#type: ignore
"""
GitHub file download helpers: UploadedGitHubFile wrapper and process_url().
"""
import os
import streamlit as st
import requests
from io import BytesIO


class UploadedGitHubFile:
    def __init__(self, content, name, filetype):
        self.file = BytesIO(content)
        self.name = name
        self.type = filetype
        self.size = len(content)

    def read(self, *args, **kwargs):
        return self.file.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self.file.seek(*args, **kwargs)


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
                url_parts = base_url.split("/")
                if len(url_parts) < 5 or url_parts[2] != "github.com":
                    st.error("Unable to parse repository from URL.")
                    return None
                repo = f"{url_parts[3]}/{url_parts[4]}"
                folder_path = path.split('/', 1)[-1] if '/' in path else path
                api_url = f"https://api.github.com/repos/{repo}/contents/{folder_path}"
                response = requests.get(api_url, headers={"Accept": "application/vnd.github.v3+json"})
                if response.status_code == 200:
                    files = [item for item in response.json() if item['type'] == 'file' and item['name'].endswith(('.csv', '.ulg'))]
                    if not files:
                        st.warning("No supported data files found in the folder.")
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
            # Handle raw file URL
            elif "raw.githubusercontent.com" in url:
                file_name = url.split("/")[-1]
                file_ext = os.path.splitext(file_name)[-1].lower()
                if file_ext in [".csv", ".ulg"]:
                    response = requests.get(url)
                    if response.status_code == 200:
                        return {file_name: (response.content, file_ext)}
                    else:
                        st.error(f"Failed to download file. Status code: {response.status_code}, URL: {url}")
                        return None
            # Handle blob URL
            elif "/blob/" in url:
                raw_url = url.replace("/blob/", "/raw/")
                file_name = raw_url.split("/")[-1]
                file_ext = os.path.splitext(file_name)[-1].lower()
                if file_ext in [".csv", ".ulg"]:
                    response = requests.get(raw_url)
                    if response.status_code == 200:
                        return {file_name: (response.content, file_ext)}
                    else:
                        st.error(f"Failed to download file. Status code: {response.status_code}, URL: {raw_url}")
                        return None
            # Handle direct file URL
            elif len(url.split("/")) > 4 and url.split("/")[4] not in ["tree", "blob", "raw"]:
                base_parts = url.split("/")
                repo = f"{base_parts[3]}/{base_parts[4]}"
                file_path = "/".join(base_parts[5:])
                raw_url = f"https://raw.githubusercontent.com/{repo}/main/{file_path}"
                file_name = file_path.split("/")[-1]
                file_ext = os.path.splitext(file_name)[-1].lower()
                if file_ext in [".csv", ".ulg"]:
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
