import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
from googleapiclient.discovery import build
import json
import time
import re
import pandas as pd
from datetime import datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 1. Vertex AI & Google API Auth
# ==========================================
@st.cache_resource
def get_credentials():
    try:
        info = dict(st.secrets["gcp_service_account"])
        return service_account.Credentials.from_service_account_info(
            info, 
            scopes=["https://www.googleapis.com/auth/cloud-platform", 
                    "https://www.googleapis.com/auth/drive", 
                    "https://www.googleapis.com/auth/spreadsheets"]
        )
    except Exception as e:
        st.error(f"Credentials Error: {e}")
        return None

def init_vertex_ai(creds):
    try:
        info = dict(st.secrets["gcp_service_account"])
        vertexai.init(project=info["project_id"], location="us-central1", credentials=creds)
        return True
    except Exception as e:
        st.error(f"Vertex AI Init Error: {e}")
        return False

# ==========================================
# 2. Google Drive / Sheets Functions
# ==========================================
def save_to_google_sheets(creds, folder_id, data_dict):
    try:
        drive_service = build('drive', 'v3', credentials=creds)
        sheets_service = build('sheets', 'v4', credentials=creds)
        
        # 1. Create Spreadsheet
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        file_metadata = {
            'name': f'Drawing_Analysis_{date_str}',
            'mimeType': 'application/vnd.google-apps.spreadsheet',
            'parents': [folder_id] if folder_id else []
        }
        spreadsheet = drive_service.files().create(body=file_metadata, fields='id').execute()
        spreadsheet_id = spreadsheet.get('id')
        
        # 2. Prepare Data (Header & Values)
        headers = list(data_dict.keys())
        values = [list(data_dict.values())]
        body = {'values': [headers] + values}
        
        # 3. Write to Sheet
        sheets_service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range="Sheet1!A1",
            valueInputOption="RAW",
            body=body
        ).execute()
        
        return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# 3. AI Analysis Logic
# ==========================================
def call_gemini_vertex(file_bytes, mime_type, target_instructions, customer_info, component_info):
    model = GenerativeModel("gemini-2.5-pro")
    prompt = f"""
    Context: {customer_info}, {component_info}
    Extract drawing data as JSON:
    {target_instructions}
    - Return ONLY valid JSON.
    """
    doc_part = Part.from_data(data=file_bytes, mime_type=mime_type)
    response = model.generate_content([doc_part, prompt])
    return response.text

# ==========================================
# 4. Main UI
# ==========================================
st.set_page_config(page_title="Drawing Analyzer Pro", layout="wide")
creds = get_credentials()

if creds and init_vertex_ai(creds):
    st.title("📄 AI Drawing Data Structurizer")

    # --- Settings ---
    st.subheader("1. Extraction & Storage Settings")
    col1, col2 = st.columns(2)
    with col1:
        customer_overview = st.text_input("Customer Overview")
        drive_folder_id = st.text_input("Google Drive Folder ID (Optional)", help="Enter the ID from the URL of your Drive folder")
    with col2:
        component_details = st.text_input("Component Type")

    # Dynamic Rows
    if 'rows' not in st.session_state:
        st.session_state.rows = [{"item": "Part Number", "guide": "Bottom right block"}]
    
    col_b1, col_b2, _ = st.columns([1,1,4])
    if col_b1.button("➕ Add Row"): st.session_state.rows.append({"item": "", "guide": ""})
    if col_b2.button("➖ Remove Row"): st.session_state.rows.pop()

    extracted_instructions = []
    for i, row in enumerate(st.session_state.rows):
        c1, c2 = st.columns([1, 2])
        item = c1.text_input(f"Item {i+1}", value=row['item'], key=f"it_{i}")
        gd = c2.text_input(f"Guide {i+1}", value=row['guide'], key=f"gd_{i}")
        if item: extracted_instructions.append(f"- {item}: {gd}")

    # --- Upload ---
    st.subheader("2. Upload Drawing")
    uploaded_file = st.file_uploader("Upload", type=["pdf", "png", "jpg", "jpeg", "tif"])

    # --- Execution ---
    if st.button("🚀 Run Analysis") and uploaded_file:
        progress = st.progress(0)
        file_bytes = uploaded_file.getvalue()
        
        with ThreadPoolExecutor() as executor:
            future = executor.submit(call_gemini_vertex, file_bytes, uploaded_file.type, "\n".join(extracted_instructions), customer_overview, component_details)
            progress.progress(50)
            result_text = future.result()
        
        progress.progress(90)
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        
        if json_match:
            data_dict = json.loads(json_match.group(0))
            st.session_state.last_result = data_dict
            st.success("Analysis Complete!")
            st.table([data_dict])
            
            # --- Export Options ---
            st.divider()
            st.subheader("📦 Export Results")
            
            # Local Excel Download
            df = pd.DataFrame([data_dict])
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Result')
            
            st.download_button(
                label="📥 Download as Excel (Local)",
                data=output.getvalue(),
                file_name=f"Drawing_Analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Google Drive Auto Save
            if st.button("☁️ Save to Google Drive (Spreadsheet)"):
                with st.spinner("Uploading to Drive..."):
                    sheet_url = save_to_google_sheets(creds, drive_folder_id, data_dict)
                    if "https" in sheet_url:
                        st.success(f"File created! [Open Spreadsheet]({sheet_url})")
                    else:
                        st.error(f"Failed to save to Drive: {sheet_url}")
        
        progress.progress(100)
