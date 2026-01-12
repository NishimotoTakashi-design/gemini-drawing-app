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
# 2. Google Drive / Sheets Advanced Logic
# ==========================================
def list_files_in_folder(creds, folder_id):
    """フォルダ内の解析対象ファイル一覧を取得"""
    service = build('drive', 'v3', credentials=creds)
    query = f"'{folder_id}' in parents and trashed = false and (mimeType contains 'image/' or mimeType = 'application/pdf' or mimeType contains 'tiff')"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    return results.get('files', [])

def download_file(creds, file_id):
    """ファイルをバイナリとしてダウンロード"""
    service = build('drive', 'v3', credentials=creds)
    request = service.files().get_media(fileId=file_id)
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()

def create_multi_sheet_spreadsheet(creds, folder_id, result_df, evidence_df):
    """解析完了後に一つのスプレッドシートにまとめて保存"""
    drive_service = build('drive', 'v3', credentials=creds)
    sheets_service = build('sheets', 'v4', credentials=creds)
    
    name = f"Batch_Analysis_{datetime.now().strftime('%Y%m%d_%H%M')}"
    meta = {'name': name, 'mimeType': 'application/vnd.google-apps.spreadsheet', 'parents': [folder_id]}
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
# 3. Parallel Worker
# ==========================================
def process_single_file(creds, file_info, target_inst, customer, component):
    """単一ファイルのダウンロードと解析を行うワーカー"""
    try:
        file_id = file_info['id']
        file_name = file_info['name']
        mime_type = file_info['mimeType']
        
        # Download
        content = download_file(creds, file_id)
        
        # AI Analysis
        model = GenerativeModel("gemini-2.5-pro")
        prompt = f"""
        Context: {customer}, {component}
        Task: Extract data as JSON with evidence in ENGLISH.
        Items: {target_inst}
        
        Rules for 'evidence':
        - Describe WHERE in the drawing you found the info in English.
        - Example: "Extracted from the 'Revision' block at the top right corner."
        - Return ONLY valid JSON: {{"results": {{...}}, "evidence": {{...}}}}
        """
        doc = Part.from_data(data=content, mime_type=mime_type)
        response = model.generate_content([doc, prompt])
        
        data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group(0))
        
        res = {"File Name": file_name}
        res.update(data.get("results", {}))
        
        ev = {"File Name": file_name}
        ev.update(data.get("evidence", {}))
        
        return res, ev
    except Exception as e:
        return {"File Name": file_info['name'], "Error": str(e)}, {"File Name": file_info['name'], "Error": str(e)}

# ==========================================
# 4. Main UI
# ==========================================
st.set_page_config(page_title="Batch Drawing Analyzer", layout="wide")
creds = get_credentials()

if creds and init_vertex_ai(creds):
    st.title("📄 AI Batch Drawing Analyzer")
    st.caption("Engine: Gemini 2.5 Pro (Vertex AI) | Parallel Processing Enabled")

    # 1. Settings
    st.subheader("1. Extraction Settings")
    c1, c2 = st.columns(2)
    with c1: customer = st.text_input("Customer Overview")
    with c2: component = st.text_input("Component Type")

    if 'rows' not in st.session_state: st.session_state.rows = [{"item": "Part Number", "guide": "Title block"}]
    
    if st.button("➕ Add Item"): st.session_state.rows.append({"item": "", "guide": ""})
    
    inst_list = []
    for i, row in enumerate(st.session_state.rows):
        r1, r2 = st.columns([1, 2])
        it = r1.text_input(f"Item {i+1}", value=row['item'], key=f"it_{i}")
        gd = r2.text_input(f"Guide {i+1}", value=row['guide'], key=f"gd_{i}")
        if it: inst_list.append(f"- {it}: {gd}")

    # 2. Input
    st.subheader("2. Google Drive Batch Input")
    folder_id = st.text_input("Google Drive Folder ID", placeholder="Enter the folder ID where drawings are stored")

    # 3. Execution
    if st.button("🚀 Run Batch Analysis") and folder_id:
        files = list_files_in_folder(creds, folder_id)
        if not files:
            st.warning("No valid drawing files found in this folder.")
        else:
            st.info(f"Found {len(files)} files. Starting parallel analysis...")
            progress = st.progress(0)
            all_results, all_evidence = [], []
            
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(process_single_file, creds, f, "\n".join(inst_list), customer, component) for f in files]
                
                for i, future in enumerate(as_completed(futures)):
                    res, ev = future.result()
                    all_results.append(res)
                    all_evidence.append(ev)
                    progress.progress((i + 1) / len(files))
            
            # Combine to DataFrames
            df_res = pd.DataFrame(all_results)
            df_ev = pd.DataFrame(all_evidence)
            
            st.success(f"Batch analysis complete in {int(time.time() - start_time)}s")
            st.write("### Preview: Analysis Results")
            st.table(df_res)
            
            # Save to Sheet
            with st.spinner("Saving results to Google Sheets in the same folder..."):
                url = create_multi_sheet_spreadsheet(creds, folder_id, df_res, df_ev)
                st.success(f"Successfully saved to Google Sheets! [Open File]({url})")
            
            # Local Download
            out = BytesIO()
            with pd.ExcelWriter(out, engine='openpyxl') as writer:
                df_res.to_excel(writer, index=False, sheet_name='Results')
                df_ev.to_excel(writer, index=False, sheet_name='Evidence')
            st.download_button("📥 Download Excel (Local)", out.getvalue(), "Batch_Analysis.xlsx")
