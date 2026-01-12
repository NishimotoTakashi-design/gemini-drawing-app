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
# 2. Export Helpers (Excel & Google Sheets)
# ==========================================
def save_to_google_sheets(creds, folder_id, result_df, evidence_df):
    """Google Driveに2シートのスプレッドシートを作成"""
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
        
        # Evidenceシートを作成
        body = {'requests': [{'addSheet': {'properties': {'title': 'Evidence'}}}]}
        sheets_service.spreadsheets().batchUpdate(spreadsheetId=ss_id, body=body).execute()
        
        # データ書き込み関数
        def write_sheet(df, sheet_name):
            values = [df.columns.tolist()] + df.values.tolist()
            sheets_service.spreadsheets().values().update(
                spreadsheetId=ss_id, range=f"{sheet_name}!A1",
                valueInputOption="RAW", body={'values': values}
            ).execute()
        
        write_sheet(result_df, "Sheet1") # デフォルトシート名をResultとして扱う
        write_sheet(evidence_df, "Evidence")
        
        return f"https://docs.google.com/spreadsheets/d/{ss_id}/edit"
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# 3. AI Analysis
# ==========================================
def call_gemini_vertex(file_bytes, mime_type, target_instructions, customer_info, component_info):
    model = GenerativeModel("gemini-2.5-pro")
    
    # エビデンス（根拠）も含めるようプロンプトを強化
    prompt = f"""
    Context: {customer_info}, {component_info}
    Task: Extract data as JSON with evidence.
    
    Items to Extract:
    {target_instructions}
    
    Output Format:
    {{
      "results": {{ "ItemName": "ExtractedValue", ... }},
      "evidence": {{ "ItemName": "The specific text or visual clue from the drawing used for this extraction", ... }}
    }}
    
    - Return ONLY valid JSON.
    """
    doc_part = Part.from_data(data=file_bytes, mime_type=mime_type)
    response = model.generate_content([doc_part, prompt])
    return response.text

# ==========================================
# 4. App UI
# ==========================================
st.set_page_config(page_title="AI Drawing Analyzer (2-Sheet Export)", layout="wide")
creds = get_credentials()

if creds and init_vertex_ai(creds):
    st.title("📄 AI Drawing Data Structurizer")
    
    # --- 1. Settings ---
    st.subheader("1. Extraction Settings")
    col_c1, col_c2 = st.columns(2)
    with col_c1: customer_overview = st.text_input("Customer Overview")
    with col_c2: component_details = st.text_input("Component Type")

    if 'rows' not in st.session_state:
        st.session_state.rows = [{"item": "Part Number", "guide": "Title block"}]

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    if col_btn1.button("➕ Add Row"): st.session_state.rows.append({"item": "", "guide": ""})
    if col_btn2.button("➖ Remove Last"): st.session_state.rows.pop()

    extracted_instructions = []
    for i, row in enumerate(st.session_state.rows):
        r1, r2 = st.columns([1, 2])
        it = r1.text_input(f"Item {i+1}", value=row['item'], key=f"it_{i}")
        gd = r2.text_input(f"Guide {i+1}", value=row['guide'], key=f"gd_{i}")
        if it: extracted_instructions.append(f"- {it}: {gd}")

    # --- 2. Upload ---
    st.subheader("2. Upload Drawing")
    input_type = st.radio("Input Method", ("Local Upload", "Google Drive"), horizontal=True)
    file_bytes, mime_type, current_file_name = None, None, "unknown"
    drive_folder_target = ""

    if input_type == "Local Upload":
        uploaded_file = st.file_uploader("Upload", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])
        if uploaded_file:
            file_bytes, mime_type, current_file_name = uploaded_file.getvalue(), uploaded_file.type, uploaded_file.name
    else:
        drive_url = st.text_input("Drive URL")
        drive_folder_target = st.text_input("Target Folder ID")

    # --- 3. Analysis ---
    if st.button("🚀 Run Analysis") and (file_bytes or input_type == "Google Drive"):
        with st.spinner("Analyzing..."):
            try:
                result_text = call_gemini_vertex(file_bytes, mime_type, "\n".join(extracted_instructions), customer_overview, component_details)
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                
                if json_match:
                    full_data = json.loads(json_match.group(0))
                    
                    # DataFrames for Export
                    res_dict = {"File Name": current_file_name}
                    res_dict.update(full_data.get("results", {}))
                    df_res = pd.DataFrame([res_dict])
                    
                    ev_dict = {"File Name": current_file_name}
                    ev_dict.update(full_data.get("evidence", {}))
                    df_ev = pd.DataFrame([ev_dict])

                    st.success("Analysis Complete!")
                    st.write("### Analysis Results")
                    st.table(df_res)
                    
                    st.write("### Evidence Sheet Content")
                    st.table(df_ev)

                    # Export Buttons
                    st.divider()
                    c1, c2 = st.columns(2)
                    with c1:
                        # Local Excel Download (2 Sheets)
                        out = BytesIO()
                        with pd.ExcelWriter(out, engine='openpyxl') as writer:
                            df_res.to_excel(writer, index=False, sheet_name='Analysis_Result')
                            df_ev.to_excel(writer, index=False, sheet_name='Evidence')
                        st.download_button("📥 Download 2-Sheet Excel", out.getvalue(), f"Result_{current_file_name}.xlsx")
                    
                    with c2:
                        if st.button("☁️ Save to Google Drive"):
                            url = save_to_google_sheets(creds, drive_folder_target, df_res, df_ev)
                            st.success(f"Saved: [Open Spreadsheet]({url})")
                else:
                    st.error("AI returned unexpected format.")
                    st.text(result_text)
            except Exception as e:
                st.error(str(e))
