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
from concurrent.futures import ThreadPoolExecutor

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
# 2. Google Drive / Sheets Helpers
# ==========================================
def fetch_file_from_drive(creds, file_id_or_url):
    """Google Driveからファイル名、MIMEタイプ、バイナリデータを取得"""
    try:
        drive_service = build('drive', 'v3', credentials=creds)
        file_id = file_id_or_url.split('/')[-1].split('?')[0] if "/" in file_id_or_url else file_id_or_url
        
        # メタデータの取得
        file_metadata = drive_service.files().get(fileId=file_id, fields="name, mimeType").execute()
        
        # バイナリデータのダウンロード
        request = drive_service.files().get_media(fileId=file_id)
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        return file_metadata['name'], file_metadata['mimeType'], fh.getvalue()
    except Exception as e:
        return None, None, str(e)

def save_to_google_sheets(creds, folder_id, result_df, evidence_df):
    """2シートのスプレッドシートをDriveに作成"""
    try:
        drive_service = build('drive', 'v3', credentials=creds)
        sheets_service = build('sheets', 'v4', credentials=creds)
        
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        file_metadata = {
            'name': f'Drawing_Analysis_{date_str}',
            'mimeType': 'application/vnd.google-apps.spreadsheet',
            'parents': [folder_id] if folder_id else []
        }
        
        spreadsheet = drive_service.files().create(body=file_metadata, fields='id').execute()
        ss_id = spreadsheet.get('id')
        
        # Evidenceシート追加
        body = {'requests': [{'addSheet': {'properties': {'title': 'Evidence'}}}]}
        sheets_service.spreadsheets().batchUpdate(spreadsheetId=ss_id, body=body).execute()
        
        def write_sheet(df, range_name):
            values = [df.columns.tolist()] + df.values.tolist()
            sheets_service.spreadsheets().values().update(
                spreadsheetId=ss_id, range=range_name,
                valueInputOption="RAW", body={'values': values}
            ).execute()
        
        write_sheet(result_df, "Sheet1!A1")
        write_sheet(evidence_df, "Evidence!A1")
        return f"https://docs.google.com/spreadsheets/d/{ss_id}/edit"
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# 3. AI Analysis (Gemini 2.5 Pro)
# ==========================================
def call_gemini_vertex(file_bytes, mime_type, target_instructions, customer_info, component_info):
    # Vertex AI Gemini 2.5 Pro
    model = GenerativeModel("gemini-2.5-pro")
    
    prompt = f"""
    Context: {customer_info}, {component_info}
    Task: Analyze drawing and extract JSON with evidence.
    Extraction Items:
    {target_instructions}
    
    Output Rules:
    - Describe WHERE in the drawing you found the info in Japanese for 'evidence'.
    - Return ONLY valid JSON.
    - Format: {{"results": {{"Item": "Value"}}, "evidence": {{"Item": "Location Description"}}}}
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
    st.caption("Engine: Gemini 2.5 Pro (us-central1)")

    # 1. Provide Context & Extraction Settings
    st.subheader("1. Provide Context & Extraction Settings")
    col_c1, col_c2 = st.columns(2)
    with col_c1: customer_overview = st.text_input("Customer Overview")
    with col_c2: component_details = st.text_input("Component Type")

    if 'rows' not in st.session_state:
        st.session_state.rows = [{"item": "Part Number", "guide": "Title block"}]
    
    c_btn1, c_btn2, _ = st.columns([1,1,4])
    if c_btn1.button("➕ Add Row"): st.session_state.rows.append({"item": "", "guide": ""})
    if c_btn2.button("➖ Remove Last Row"): st.session_state.rows.pop()

    inst_list = []
    for i, row in enumerate(st.session_state.rows):
        r1, r2 = st.columns([1, 2])
        it = r1.text_input(f"Item {i+1}", value=row['item'], key=f"it_{i}")
        gd = r2.text_input(f"Guide {i+1}", value=row['guide'], key=f"gd_{i}")
        if it: inst_list.append(f"- {it}: {gd}")

    # 2. Upload or Specify Drawing
    st.subheader("2. Upload Drawing")
    input_type = st.radio("Input Method", ("Local Upload", "Google Drive"), horizontal=True)
    file_bytes, mime_type, current_file_name = None, None, "unknown"
    
    if input_type == "Local Upload":
        uploaded_file = st.file_uploader("Upload Drawing", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])
        if uploaded_file:
            file_bytes, mime_type, current_file_name = uploaded_file.getvalue(), uploaded_file.type, uploaded_file.name
    else:
        drive_url = st.text_input("Google Drive File URL / ID")
        drive_folder_target = st.text_input("Target Folder ID (for Sheet save)")
        if drive_url:
            with st.spinner("Fetching file from Drive..."):
                current_file_name, mime_type, drive_data = fetch_file_from_drive(creds, drive_url)
                if isinstance(drive_data, bytes):
                    file_bytes = drive_data
                    st.success(f"File Linked: {current_file_name}")
                else:
                    st.error(f"Drive Error: {drive_data}")

    # 3. Run Analysis
    if st.button("🚀 Run AI Analysis") and file_bytes:
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            start_time = time.time()
            with ThreadPoolExecutor() as executor:
                status_text.text("AI is analyzing drawing in us-central1...")
                progress_bar.progress(40)
                future = executor.submit(call_gemini_vertex, file_bytes, mime_type, "\n".join(inst_list), customer_overview, component_details)
                result_text = future.result()

            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                full_data = json.loads(json_match.group(0))
                
                # Data Preparation
                res_dict = {"File Name": current_file_name}
                res_dict.update(full_data.get("results", {}))
                df_res = pd.DataFrame([res_dict])
                
                ev_dict = {"File Name": current_file_name}
                ev_dict.update(full_data.get("evidence", {}))
                df_ev = pd.DataFrame([ev_dict])

                st.success(f"Analysis Complete! ({int(time.time() - start_time)}s)")
                st.write("### Analysis Results")
                st.table(df_res)
                st.write("### Extraction Evidence")
                st.table(df_ev)

                # Export
                st.divider()
                e1, e2 = st.columns(2)
                with e1:
                    out = BytesIO()
                    with pd.ExcelWriter(out, engine='openpyxl') as writer:
                        df_res.to_excel(writer, index=False, sheet_name='Result')
                        df_ev.to_excel(writer, index=False, sheet_name='Evidence')
                    st.download_button("📥 Download Excel", out.getvalue(), f"Analysis_{current_file_name}.xlsx")
                with e2:
                    if st.button("☁️ Save to Google Drive"):
                        if input_type == "Google Drive":
                            url = save_to_google_sheets(creds, drive_folder_target, df_res, df_ev)
                            st.success(f"Saved: [Open Spreadsheet]({url})")
                        else:
                            st.warning("Please specify Folder ID in Step 2.")
            progress_bar.progress(100)
        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
