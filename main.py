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
# 1. 統合認証とVertex AI初期化
# ==========================================
@st.cache_resource
def get_unified_credentials():
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("Secrets 'gcp_service_account' が見つかりません。")
            return None
            
        info = dict(st.secrets["gcp_service_account"])
        # 改行コードの置換（エラー防止）
        if "private_key" in info:
            info["private_key"] = info["private_key"].replace('\\n', '\n')
            
        scopes = [
            'https://www.googleapis.com/auth/cloud-platform',
            'https://www.googleapis.com/auth/drive',
            'https://www.googleapis.com/auth/spreadsheets'
        ]
        
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
        vertexai.init(project=info["project_id"], location="us-central1", credentials=creds)
        return creds
    except Exception as e:
        st.error(f"認証エラー: {e}")
        return None

# ==========================================
# 2. Google API ヘルパー（ID抽出機能付き）
# ==========================================
def extract_folder_id(input_str):
    """URLまたはパラメータ付き文字列から純粋なフォルダIDのみを抽出"""
    if not input_str:
        return ""
    # URL形式や?以降のパラメータを削除
    match = re.search(r'folders/([a-zA-Z0-9_-]+)', input_str)
    if match:
        return match.group(1)
    # 単純な文字列からパラメータ (?...) を除去
    return input_str.split('?')[0].strip()

def list_files_in_folder(creds, folder_id):
    try:
        service = build('drive', 'v3', credentials=creds)
        clean_id = extract_folder_id(folder_id)
        query = f"'{clean_id}' in parents and trashed = false and (mimeType contains 'image/' or mimeType = 'application/pdf' or mimeType contains 'tiff')"
        results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
        return results.get('files', [])
    except Exception as e:
        st.error(f"Google Driveアクセスエラー: {str(e)}\nフォルダID '{folder_id}' にサービスアカウントが招待されているか確認してください。")
        return []

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
    clean_id = extract_folder_id(folder_id)
    
    name = f"Batch_Analysis_{datetime.now().strftime('%Y%m%d_%H%M')}"
    meta = {'name': name, 'mimeType': 'application/vnd.google-apps.spreadsheet', 'parents': [clean_id]}
    
    ss = drive_service.files().create(body=meta, fields='id').execute()
    ss_id = ss.get('id')
    
    sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=ss_id, 
        body={'requests': [{'addSheet': {'properties': {'title': 'Evidence'}}}]}
    ).execute()
    
    def upload(df, range_name):
        df_clean = df.fillna("")
        vals = [df_clean.columns.tolist()] + df_clean.values.tolist()
        sheets_service.spreadsheets().values().update(
            spreadsheetId=ss_id, range=range_name, 
            valueInputOption="RAW", body={'values': vals}
        ).execute()
    
    upload(result_df, "Sheet1!A1")
    upload(evidence_df, "Evidence!A1")
    return f"https://docs.google.com/spreadsheets/d/{ss_id}/edit"

# ==========================================
# 3. AI 解析ワーカー
# ==========================================
def process_single_file(creds, file_content, file_name, mime_type, target_inst, customer, component):
    try:
        model = GenerativeModel("gemini-2.5-pro")
        prompt = f"""
        Context: {customer}, {component}
        Task: Analyze drawing and extract JSON with evidence in ENGLISH.
        Extraction Items: {target_inst}
        
        Rules for 'evidence': Describe specifically WHERE in English (e.g. "Title block").
        Format: Return ONLY valid JSON: {{"results": {{...}}, "evidence": {{...}}}}
        """
        doc = Part.from_data(data=file_content, mime_type=mime_type)
        response = model.generate_content([doc, prompt])
        
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        data = json.loads(json_match.group(0))
        
        res = {"File Name": file_name}; res.update(data.get("results", {}))
        ev = {"File Name": file_name}; ev.update(data.get("evidence", {}))
        return res, ev, None
    except Exception as e:
        return None, None, f"{file_name}: {str(e)}"

# ==========================================
# 4. メイン UI
# ==========================================
st.set_page_config(page_title="AI Batch Drawing Analyzer", layout="wide")
creds = get_unified_credentials()

if creds:
    st.title("📄 AI Batch Drawing Analyzer")
    st.caption("Engine: Gemini 2.5 Pro (Vertex AI) | Parallel Processing & Multi-Sheet Export")

    # 1. Extraction Settings
    st.subheader("1. Extraction Settings")
    c1, c2 = st.columns(2)
    with c1: customer = st.text_input("Customer Overview", value="Sumitomo Machinery")
    with c2: component = st.text_input("Component Type", value="Motor")

    if 'rows' not in st.session_state: 
        st.session_state.rows = [{"item": "Part Number", "guide": "Title block"}]
    
    if st.button("➕ Add Item"):
        st.session_state.rows.append({"item": "", "guide": ""})
        st.rerun()
    
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

    if 'all_res' not in st.session_state: st.session_state.all_res = []
    if 'all_ev' not in st.session_state: st.session_state.all_ev = []

    # --- 実行ロジック ---
    if input_type == "Local Upload":
        uploaded_file = st.file_uploader("Upload Drawing", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])
        if st.button("🚀 Run Local Analysis") and uploaded_file:
            st.session_state.all_res, st.session_state.all_ev = [], []
            with st.spinner("Analyzing..."):
                res, ev, err = process_single_file(creds, uploaded_file.getvalue(), uploaded_file.name, uploaded_file.type, target_inst, customer, component)
                if res:
                    st.session_state.all_res.append(res); st.session_state.all_ev.append(ev)
                else: st.error(err)
    else:
        # フォルダID入力（URLをそのまま貼り付けてもOKなように修正）
        raw_folder_input = st.text_input("Google Drive Folder ID / URL")
        if st.button("🚀 Run Batch Analysis") and raw_folder_input:
            st.session_state.all_res, st.session_state.all_ev = [], []
            files = list_files_in_folder(creds, raw_folder_input)
            
            if files:
                total = len(files)
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {executor.submit(process_single_file, creds, download_file(creds, f['id']), f['name'], f['mimeType'], target_inst, customer, component): f for f in files}
                    for i, future in enumerate(as_completed(futures)):
                        res, ev, err = future.result()
                        if res:
                            st.session_state.all_res.append(res); st.session_state.all_ev.append(ev)
                        progress_bar.progress((i + 1) / total)
                        status_text.text(f"進捗: {i+1}/{total} ファイル完了")
            else:
                st.warning("ファイルが見つからないか、権限がありません。フォルダをサービスアカウントに共有してください。")

    # 4. 結果表示とエクスポート
    if st.session_state.all_res:
        df_res = pd.DataFrame(st.session_state.all_res)
        df_ev = pd.DataFrame(st.session_state.all_ev)
        
        st.success("解析完了！")
        st.write("### 📊 結果プレビュー")
        st.table(df_res)
        st.write("### 🔍 エビデンス (English)")
        st.table(df_ev)
        
        st.divider()
        e1, e2 = st.columns(2)
        with e1:
            out = BytesIO()
            with pd.ExcelWriter(out, engine='openpyxl') as writer:
                df_res.to_excel(writer, index=False, sheet_name='Results')
                df_ev.to_excel(writer, index=False, sheet_name='Evidence')
            st.download_button("📥 Excelダウンロード (2シート)", out.getvalue(), "Analysis_Report.xlsx")
        with e2:
            if input_type == "Google Drive Folder" and st.button("☁️ Google Driveに保存"):
                with st.spinner("保存中..."):
                    url = create_multi_sheet_spreadsheet(creds, raw_folder_input, df_res, df_ev)
                    st.success(f"保存完了！ [スプレッドシートを開く]({url})")
