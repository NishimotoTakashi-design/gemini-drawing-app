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
# 2. Google Drive Helpers
# ==========================================
def get_drive_file_data(creds, file_url_or_id):
    """Google Driveからファイル名とバイナリデータを取得"""
    try:
        drive_service = build('drive', 'v3', credentials=creds)
        file_id = file_url_or_id.split('/')[-1].split('?')[0] if "/" in file_url_or_id else file_url_or_id
        
        # ファイル情報の取得
        file_metadata = drive_service.files().get(fileId=file_id, fields="name, mimeType").execute()
        # 本来はここでダウンロード処理が必要ですが、AI解析用にメタデータとIDを返します
        return file_metadata['name'], file_metadata['mimeType'], file_id
    except Exception as e:
        return None, None, str(e)

def save_to_google_sheets(creds, folder_id, data_dict):
    """結果をスプレッドシートに保存（ファイル名が一番左にくる）"""
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
        
        # カラムの順序を整理（File Nameを先頭に）
        headers = list(data_dict.keys())
        values = [list(data_dict.values())]
        body = {'values': [headers] + values}
        
        sheets_service.spreadsheets().values().update(
            spreadsheetId=ss_id, range="Sheet1!A1",
            valueInputOption="RAW", body=body
        ).execute()
        
        return f"https://docs.google.com/spreadsheets/d/{ss_id}/edit"
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# 3. AI Analysis (Gemini 2.5 Pro Logic)
# ==========================================
def call_gemini_vertex(file_bytes, mime_type, target_instructions, customer_info, component_info):
    model = GenerativeModel("gemini-2.5-pro") 
    prompt = f"""
    Context: {customer_info}, {component_info}
    Task: Extract drawing data as JSON.
    {target_instructions}
    - Return ONLY valid JSON.
    """
    doc_part = Part.from_data(data=file_bytes, mime_type=mime_type)
    response = model.generate_content([doc_part, prompt])
    return response.text

# ==========================================
# 4. Main App UI
# ==========================================
st.set_page_config(page_title="AI Drawing Analyzer Pro", layout="wide")
creds = get_credentials()

if creds and init_vertex_ai(creds):
    st.title("📄 AI Drawing Data Structurizer")
    st.caption("Engine: Gemini 2.5 Pro | Region: us-central1")

    # --- 1. Settings ---
    st.subheader("1. Extraction Settings")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        customer_overview = st.text_input("Customer Overview")
    with col_c2:
        component_details = st.text_input("Component Type")

    if 'rows' not in st.session_state:
        st.session_state.rows = [{"item": "Part Number", "guide": "Title block"}]

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    if col_btn1.button("➕ Add Item"): st.session_state.rows.append({"item": "", "guide": ""})
    if col_btn2.button("➖ Remove Last"): st.session_state.rows.pop()

    extracted_instructions = []
    for i, row in enumerate(st.session_state.rows):
        r_col1, r_col2 = st.columns([1, 2])
        it_name = r_col1.text_input(f"Item {i+1}", value=row['item'], key=f"it_{i}")
        it_guide = r_col2.text_input(f"Guide {i+1}", value=row['guide'], key=f"gd_{i}")
        if it_name: extracted_instructions.append(f"- {it_name}: {it_guide}")

    # --- 2. Input Selection ---
    st.subheader("2. Upload Drawing")
    input_type = st.radio("Select Input Method:", ("Local Upload", "Google Drive"), horizontal=True)
    
    current_file_name = "unknown_file"
    file_bytes = None
    mime_type = None

    if input_type == "Local Upload":
        uploaded_file = st.file_uploader("Upload", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])
        if uploaded_file:
            file_bytes = uploaded_file.getvalue()
            mime_type = uploaded_file.type
            current_file_name = uploaded_file.name
    else:
        drive_file_url = st.text_input("Google Drive File URL / ID")
        drive_folder_target = st.text_input("Destination Folder ID (for Spreadsheet)")
        if drive_file_url:
            with st.spinner("Fetching file info..."):
                current_file_name, mime_type, _ = get_drive_file_data(creds, drive_file_url)
                if current_file_name:
                    st.info(f"📁 File detected: {current_file_name}")

    # --- 3. Run Analysis ---
    if st.button("🚀 Run AI Analysis"):
        if not file_bytes and input_type == "Local Upload":
            st.warning("Please upload a file.")
        else:
            progress = st.progress(0)
            try:
                start_time = time.time()
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(call_gemini_vertex, file_bytes, mime_type, "\n".join(extracted_instructions), customer_overview, component_details)
                    result_text = future.result()

                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    # データの構築
                    raw_data = json.loads(json_match.group(0))
                    
                    # 【重要】一番左にファイル名を追加
                    data_dict = {"File Name": current_file_name}
                    data_dict.update(raw_data) # 解析結果を結合
                    
                    st.subheader("3. Results")
                    st.success(f"Complete! ({int(time.time() - start_time)}s)")
                    st.table([data_dict]) # 画面表示
                    
                    # --- Export ---
                    st.divider()
                    exp_col1, exp_col2 = st.columns(2)
                    with exp_col1:
                        df = pd.DataFrame([data_dict]) # File Nameが先頭のDF
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False)
                        st.download_button("📥 Download Excel (Local)", data=output.getvalue(), file_name=f"Result_{current_file_name}.xlsx")
                    
                    with exp_col2:
                        if st.button("☁️ Save to Google Drive"):
                            res = save_to_google_sheets(creds, drive_folder_target, data_dict)
                            st.success(f"Created: {res}")
                progress.progress(100)
            except Exception as e:
                st.error(f"Error: {str(e)}")
