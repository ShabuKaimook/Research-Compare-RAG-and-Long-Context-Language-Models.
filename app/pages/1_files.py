import streamlit as st
import requests
from pathlib import Path

FASTAPI_URL = "http://0.0.0.0:8000"

st.set_page_config(page_title="File Management", page_icon="📂")

st.title("📂 File Management")

st.markdown("""
Manage your documents here. Supported formats:
- **PDF** (.pdf)
- **Text** (.txt)
""")

# File list section
st.subheader("📋 Uploaded Files")
response = requests.get(f"{FASTAPI_URL}/files")
if response.status_code == 200:
    files = response.json().get("files", [])
    if files:
        for file in files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 {file['name']} ({file['size_kb']} KB)")
            with col2:
                if st.button("Delete", key=file['name']):
                    delete_response = requests.delete(f"{FASTAPI_URL}/files/{file['name']}")
                    if delete_response.status_code == 200:
                        st.rerun()
                        st.success(f"Deleted {file['name']}")
                    else:
                        st.error(f"Failed to delete {file['name']}")
    else:
        st.info("No files uploaded yet.")
else:
    st.error("Failed to fetch file list.")

# File upload section
st.subheader("📤 Upload New File")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

if uploaded_file is not None:
    st.write(f"**Filename:** {uploaded_file.name}")
    st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")

    if st.button("Upload & Process"):
        with st.spinner("Uploading and processing file..."):
            try:
                # Save file temporarily
                temp_file = Path("temp") / uploaded_file.name
                temp_file.parent.mkdir(exist_ok=True)
                temp_file.write_bytes(uploaded_file.getvalue())

                # Send file to API
                with open(temp_file, "rb") as f:
                    response = requests.post(f"{FASTAPI_URL}/files/upload", files={"file": f})

                if response.status_code == 200:
                    st.rerun()
                    st.success("File uploaded and processed successfully!")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

                # Clean up temporary file
                temp_file.unlink()

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

