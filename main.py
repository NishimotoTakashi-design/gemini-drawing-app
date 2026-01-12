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
# 1. 認証とVertex AIの初期化 (Scopesの統合)
# ==========================================
@st.cache_resource
def get_unified_credentials():
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("Streamlit Secrets 'gcp_service_account' is missing.")
            return None
            
        info = dict(st.secrets["gcp_service_account"])
        # 改行コードの処理を参考コードに合わせて実装
        if "private_key" in info:
            info["private_key"] = info["private_key"].replace('\\n', '\n')
            
        # 参考コードにある scopes をそのまま適用
        scopes = [
            'https://www.googleapis.com/auth/cloud-platform', 
            'https://www.googleapis.com/auth/drive', 
            'https://www.googleapis.com/auth/spreadsheets'
        ]
        
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
        
        # Vertex AIの初期化
        vertexai.init(project=info["project_id"], location="us-central1", credentials=creds)
        
        return creds
    except Exception as e:
        st.error(f"🚨 Authentication Error: {e}")
        return None

# ==========================================
# 2. Google Drive Helpers (API経由)
# ==========================================
def list_files_in_folder(creds, folder_id):
    """フォルダ内の図面ファイルをリストアップ"""
    try:
        service = build('drive', 'v3', credentials=creds)
        # IDからURLパラメータ(ths=true等)を除去
        clean_id = folder_id.split('/')[-1].split('?')[0]
        query = f"'{clean_id}' in parents and trashed = false and (mimeType contains 'image/' or mimeType = 'application/pdf' or mimeType contains 'tiff')"
        results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
        return results.get('files', [])
    except Exception as e:
        st.error(f"🚨 Google Drive Access Error: {str(e)}")
        return []

def download_file(creds, file_id):
    """API経由でファイルをバイナリとしてダウンロード"""
    service = build('drive', 'v3', credentials=creds)
    request = service.files().get_media(fileId=file_id)
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()

def create_multi_sheet_spreadsheet(creds, folder_id, result_df, evidence_df):
    """結果を2シートのスプレッドシートとして保存"""
    drive_service = build('drive', 'v3', credentials=creds)
    sheets_service = build('sheets', 'v4', credentials=creds)
    clean_id = folder_id.split('/')[-1].split('?')[0]
    
    name = f"Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M')}"
    meta = {'name': name, 'mimeType': 'application/vnd.google-apps.spreadsheet', 'parents': [clean_id]}
    ss = drive_service.files().create(body=meta, fields='id').execute()
    ss_id = ss.get('id')
    
    # 2シート構成にする
    sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=ss_id, 
        body={'requests': [{'addSheet': {'properties': {'title': 'Evidence'}}}]}
    ).execute()
    
    def upload(df, r):
        df_clean = df.fillna("")
        vals = [df_clean.columns.tolist()] + df_clean.values.tolist()
        sheets_service.spreadsheets().values().update(spreadsheetId=ss_id, range=r, valueInputOption="RAW", body={'values': vals}).execute()
    
    upload(result_df, "Sheet1!A1")
    upload(evidence_df, "Evidence!A1")
    return f"https://docs.google.com/spreadsheets/d/{ss_id}/edit"

# ==========================================
# 3. AI Analysis logic
# ==========================================
def process_single_file(creds, file_content, file_name, mime_type, target_inst, customer, component):
    try:
        model = GenerativeModel("gemini-2.5-pro")
        prompt = f"""
        Context: {customer}, {component}
        Task: Extract data as JSON with evidence in ENGLISH.
        Items: {target_inst}
        Evidence Rule: Describe WHERE the info was found in English.
        """
        doc = Part.from_data(data=file_content, mime_type=mime_type)
        response = model.generate_content([doc, prompt])
        data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group(0))
        
        res = {"File Name": file_name}; res.update(data.get("results", {}))
        ev = {"File Name": file_name}; ev.update(data.get("evidence", {}))
        return res, ev, None
    except Exception as e:
        return None, None, f"{file_name}: {str(e)}"

# ==========================================
# 4. Streamlit UI
# ==========================================
st.set_page_config(page_title="AI Batch Drawing Analyzer", layout="wide")
creds = get_unified_credentials()

if creds:
    st.title("📄 AI Drawing Data Structurizer")
    
    st.subheader("1. Extraction Settings")
    c1, c2 = st.columns(2)
    with c1: customer = st.text_input("Customer Overview")
    with c2: component = st.text_input("Component Type")
    
    if 'rows' not in st.session_state: st.session_state.rows = [{"item": "Part Number", "guide": "Title block"}]
    if st.button("➕ Add Item"): st.session_state.rows.append({"item": "", "guide": ""}); st.rerun()
    
    inst_list = []
    for i, row in enumerate(st.session_state.rows):
        r1, r2 = st.columns([1, 2])
        it = r1.text_input(f"Item {i+1}", value=row['item'], key=f"it_{i}")
        gd = r2.text_input(f"Guide {i+1}", value=row['guide'], key=f"gd_{i}")
        if it: inst_list.append(f"- {it}: {gd}")

    st.subheader("2. Select Input Method")
    input_type = st.radio("Source", ("Local Upload", "Google Drive Folder"), horizontal=True)

    all_res, all_ev = [], []

    if input_type == "Local Upload":
        uploaded_file = st.file_uploader("Upload Drawing", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])
        if st.button("🚀 Run Local Analysis") and uploaded_file:
            with st.spinner("Analyzing..."):
                res, ev, err = process_single_file(creds, uploaded_file.getvalue(), uploaded_file.name, uploaded_file.type, "\n".join(inst_list), customer, component)
                if res: all_res.append(res); all_ev.append(ev)
                else: st.error(err)
    else:
        folder_id = st.text_input("Google Drive Folder ID (e.g. 1WDoyc...)")
        if st.button("🚀 Run Batch Analysis") and folder_id:
            files = list_files_in_folder(creds, folder_id)
            if files:
                st.info(f"Found {len(files)} files. Analyzing in parallel...")
                progress = st.progress(0)
                status_text = st.empty()
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {executor.submit(process_single_file, creds, download_file(creds, f['id']), f['name'], f['mimeType'], "\n".join(inst_list), customer, component): f for f in files}
                    for i, future in enumerate(as_completed(futures)):
                        res, ev, err = future.result()
                        if res: all_res.append(res); all_ev.append(ev)
                        progress.progress((i + 1) / len(files))
                        status_text.text(f"Processed: {i+1}/{len(files)}")
            else:
                st.warning("No files found or access denied. Ensure the folder is shared with the Service Account.")

    if all_res:
        df_res, df_ev = pd.DataFrame(all_res), pd.DataFrame(all_ev)
        st.success("Analysis Complete!")
        st.table(df_res)
        
        st.divider()
        e1, e2 = st.columns(2)
        with e1:
            out = BytesIO()
            with pd.ExcelWriter(out, engine='openpyxl') as writer:
                df_res.to_excel(writer, index=False, sheet_name='Results')
                df_ev.to_excel(writer, index=False, sheet_name='Evidence')
            st.download_button("📥 Download Excel", out.getvalue(), "Analysis_Report.xlsx")
        with e2:
            if input_type == "Google Drive Folder" and st.button("☁️ Save to Google Drive"):
                with st.spinner("Saving..."):
                    url = create_multi_sheet_spreadsheet(creds, folder_id, df_res, df_ev)
                    st.success(f"Saved! [Open Spreadsheet]({url})")
