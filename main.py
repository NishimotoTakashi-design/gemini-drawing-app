import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import json
import time
import re
import pandas as pd
from datetime import datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. Auth & Vertex AI Setup
# ==========================================
@st.cache_resource
def get_credentials():
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("Secrets 'gcp_service_account' not found.")
            return None
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
# 2. Google API Helpers
# ==========================================
def list_files_in_folder(creds, folder_id):
    service = build('drive', 'v3', credentials=creds)
    query = f"'{folder_id}' in parents and trashed = false and (mimeType contains 'image/' or mimeType = 'application/pdf' or mimeType contains 'tiff')"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    return results.get('files', [])

def download_file(creds, file_id):
    service = build('drive', 'v3', credentials=creds)
    request = service.files().get_media(fileId=file_id)
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()

def create_multi_sheet_spreadsheet(creds, folder_id, result_df, evidence_df):
    drive_service = build('drive', 'v3', credentials=creds)
    sheets_service = build('sheets', 'v4', credentials=creds)
    name = f"Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M')}"
    meta = {'name': name, 'mimeType': 'application/vnd.google-apps.spreadsheet', 'parents': [folder_id] if folder_id else []}
    ss = drive_service.files().create(body=meta, fields='id').execute()
    ss_id = ss.get('id')
    sheets_service.spreadsheets().batchUpdate(spreadsheetId=ss_id, body={'requests': [{'addSheet': {'properties': {'title': 'Evidence'}}}]}).execute()
    
    def upload(df, r):
        vals = [df.columns.tolist()] + df.values.tolist()
        sheets_service.spreadsheets().values().update(spreadsheetId=ss_id, range=r, valueInputOption="RAW", body={'values': vals}).execute()
    
    upload(result_df, "Sheet1!A1")
    upload(evidence_df, "Evidence!A1")
    return f"https://docs.google.com/spreadsheets/d/{ss_id}/edit"

# ==========================================
# 3. AI Analysis Worker (Evidence in English)
# ==========================================
def process_single_file(creds, file_content, file_name, mime_type, target_inst, customer, component):
    try:
        # Vertex AI Gemini 2.5 Pro
        model = GenerativeModel("gemini-2.5-pro")
        
        # 指示を明確化（英語での根拠出力を必須に）
        prompt = f"""
        Context: {customer}, {component}
        Task: Analyze the attached drawing and extract technical data.
        
        Extraction Items and Guides:
        {target_inst}
        
        Output Rules:
        1. Results: Extract the actual values for each item.
        2. Evidence: For EACH item, describe specifically WHERE in the drawing the information was found. 
        3. Language for Evidence: **Must be written in ENGLISH.**
        4. JSON Format: Return ONLY a valid JSON object with "results" and "evidence" keys.
        
        Example for Evidence: "Found in the title block at the bottom right corner" or "Extracted from general notes section item #4."
        """
        
        doc = Part.from_data(data=file_content, mime_type=mime_type)
        response = model.generate_content([doc, prompt])
        
        # JSON抽出のロジックを強化
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not json_match:
            raise ValueError("AI did not return valid JSON.")
            
        data = json.loads(json_match.group(0))
        
        res = {"File Name": file_name}
        res.update(data.get("results", {}))
        
        ev = {"File Name": file_name}
        ev.update(data.get("evidence", {}))
        
        return res, ev
    except Exception as e:
        err_msg = f"Error: {str(e)}"
        return {"File Name": file_name, "Error": err_msg}, {"File Name": file_name, "Error": err_msg}

# ==========================================
# 4. Streamlit UI
# ==========================================
st.set_page_config(page_title="AI Drawing Analyzer Pro", layout="wide")
creds = get_credentials()

if creds and init_vertex_ai(creds):
    st.title("📄 AI Drawing Data Structurizer")
    st.caption("Engine: Gemini 2.5 Pro (Vertex AI) | Parallel Processing & Multi-Sheet Export")

    # 1. Extraction Settings
    st.subheader("1. Extraction Settings")
    c1, c2 = st.columns(2)
    with c1: customer = st.text_input("Customer Overview", placeholder="e.g., Automotive OEM")
    with c2: component = st.text_input("Component Type", placeholder="e.g., Wire Harness")

    if 'rows' not in st.session_state:
        st.session_state.rows = [{"item": "Part Number", "guide": "Bottom right title block"}]
    
    col_a1, col_a2, _ = st.columns([1,1,4])
    if col_a1.button("➕ Add Item"): st.session_state.rows.append({"item": "", "guide": ""})
    if col_a2.button("➖ Remove Last"): st.session_state.rows.pop()
    
    inst_list = []
    for i, row in enumerate(st.session_state.rows):
        r1, r2 = st.columns([1, 2])
        it = r1.text_input(f"Item {i+1}", value=row['item'], key=f"it_{i}")
        gd = r2.text_input(f"Guide {i+1}", value=row['guide'], key=f"gd_{i}")
        if it: inst_list.append(f"- {it}: {gd}")
    target_inst = "\n".join(inst_list)

    # 2. Input Method
    st.subheader("2. Select Input Method")
    input_type = st.radio("Input Source", ("Local Upload", "Google Drive Folder"), horizontal=True)

    all_res, all_ev = [], []

    # --- Case A: Local Upload ---
    if input_type == "Local Upload":
        uploaded_file = st.file_uploader("Upload Drawing", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])
        if st.button("🚀 Run Local Analysis") and uploaded_file:
            with st.spinner("AI is analyzing file..."):
                res, ev = process_single_file(creds, uploaded_file.getvalue(), uploaded_file.name, uploaded_file.type, target_inst, customer, component)
                all_res.append(res); all_ev.append(ev)

    # --- Case B: Google Drive ---
    else:
        folder_id = st.text_input("Google Drive Folder ID")
        if st.button("🚀 Run Batch Analysis") and folder_id:
            files = list_files_in_folder(creds, folder_id)
            if files:
                st.info(f"Found {len(files)} files. Starting parallel processing (Max 5 concurrent)...")
                progress = st.progress(0)
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {executor.submit(process_single_file, creds, download_file(creds, f['id']), f['name'], f['mimeType'], target_inst, customer, component): f for f in files}
                    for i, future in enumerate(as_completed(futures)):
                        res, ev = future.result()
                        all_res.append(res); all_ev.append(ev)
                        progress.progress((i + 1) / len(files))
            else:
                st.warning("No valid files found in this folder.")

    # 4. Display & Export (Common Results UI)
    if all_res:
        df_res = pd.DataFrame(all_res)
        df_ev = pd.DataFrame(all_ev)
        
        st.success("Analysis Complete!")
        
        st.write("### 📊 Result Sheet Preview")
        st.table(df_res)
        
        st.write("### 🔍 Evidence Sheet Preview (Location Guide)")
        st.table(df_ev)
        
        st.divider()
        e1, e2 = st.columns(2)
        with e1:
            # Excel Generation with 2 Sheets
            out = BytesIO()
            with pd.ExcelWriter(out, engine='openpyxl') as writer:
                df_res.to_excel(writer, index=False, sheet_name='Analysis_Results')
                df_ev.to_excel(writer, index=False, sheet_name='Evidence_English')
            st.download_button("📥 Download Excel (2 Sheets)", out.getvalue(), f"Analysis_Report_{datetime.now().strftime('%Y%m%d')}.xlsx")
        
        with e2:
            if input_type == "Google Drive Folder" and st.button("☁️ Auto-Save to Google Drive"):
                with st.spinner("Creating Spreadsheet..."):
                    url = create_multi_sheet_spreadsheet(creds, folder_id, df_res, df_ev)
                    st.success(f"Successfully saved in the same folder! [Open Spreadsheet]({url})")
