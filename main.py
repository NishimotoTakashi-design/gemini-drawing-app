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
# 2. Google Drive / Sheets Logic
# ==========================================
def save_to_google_sheets(creds, folder_path_or_id, data_dict):
    """Saves the result to a new Spreadsheet in the specified Drive folder"""
    try:
        drive_service = build('drive', 'v3', credentials=creds)
        sheets_service = build('sheets', 'v4', credentials=creds)
        
        # Folder ID extraction (Assuming simple ID or URL)
        folder_id = folder_path_or_id.split('/')[-1] if "drive.google.com" in folder_path_or_id else folder_path_or_id
        
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        file_metadata = {
            'name': f'Drawing_Analysis_{date_str}',
            'mimeType': 'application/vnd.google-apps.spreadsheet',
            'parents': [folder_id] if folder_id else []
        }
        
        spreadsheet = drive_service.files().create(body=file_metadata, fields='id').execute()
        ss_id = spreadsheet.get('id')
        
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
# 3. AI Analysis (Gemini 2.5 Pro)
# ==========================================
def call_gemini_vertex(file_bytes, mime_type, target_instructions, customer_info, component_info):
    # vertexai uses 2.5-pro
    # If gemini-2.5-pro is explicitly enabled in your project, use that name.
    model = GenerativeModel("gemini-2.5-pro") 
    
    prompt = f"""
    [System Note: Region priority us-central1]
    Context: {customer_info}, {component_info}
    Task: Analyze the technical drawing and extract data as JSON.
    Instructions:
    {target_instructions}
    - Return ONLY valid JSON. If missing, use null.
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

    # --- 1. Provide Context & Extraction Settings ---
    st.subheader("1. Provide Context & Extraction Settings")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        customer_overview = st.text_input("Customer Overview", placeholder="e.g., Major OEM")
    with col_c2:
        component_details = st.text_input("Component Type", placeholder="e.g., Wire Harness")

    st.write("### Extraction Items")
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
        st.session_state.rows[i] = {"item": it_name, "guide": it_guide}
        if it_name: extracted_instructions.append(f"- {it_name}: {it_guide}")

    # --- 2. Upload or Specify Drawing ---
    st.subheader("2. Upload or Specify Drawing")
    input_type = st.radio("Select Input Method:", ("Local Upload", "Google Drive"), horizontal=True)
    
    file_bytes = None
    mime_type = None
    drive_folder_target = ""

    if input_type == "Local Upload":
        uploaded_file = st.file_uploader("Upload", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])
        if uploaded_file:
            file_bytes = uploaded_file.getvalue()
            mime_type = uploaded_file.type
    else:
        # In Google Drive mode, user provides the file via path/URL
        drive_file_url = st.text_input("Google Drive File URL / ID", placeholder="Paste the drawing file link here")
        drive_folder_target = st.text_input("Destination Folder ID (for Spreadsheet)", placeholder="Where to save the result")
        st.info("ℹ️ Note: If using Drive URL, the Service Account must have access to that file.")
        # Logic to fetch file from Drive would be needed here for fully automated Drive-to-AI flow

    # --- 3. Run Analysis ---
    if st.button("🚀 Run AI Analysis") and (file_bytes or input_type == "Google Drive"):
        if not file_bytes and input_type == "Local Upload":
            st.warning("Please upload a file.")
        else:
            progress = st.progress(0)
            status = st.empty()
            
            try:
                start_time = time.time()
                status.text("Step 1/2: AI Analysis in progress...")
                
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        call_gemini_vertex, 
                        file_bytes, mime_type, "\n".join(extracted_instructions), customer_overview, component_details
                    )
                    while not future.done():
                        time.sleep(1)
                        progress.progress(min(int(time.time() - start_time) * 2, 85))
                    
                    result_text = future.result()

                status.text("Step 2/2: Processing output...")
                progress.progress(95)
                
                st.subheader("3. Results")
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                
                if json_match:
                    data_dict = json.loads(json_match.group(0))
                    st.success(f"Complete! ({int(time.time() - start_time)}s)")
                    st.table([data_dict])
                    
                    # --- Export Buttons ---
                    st.divider()
                    st.write("#### 📦 Export Options")
                    exp_col1, exp_col2 = st.columns(2)
                    
                    with exp_col1:
                        # Local Excel Download
                        df = pd.DataFrame([data_dict])
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False)
                        st.download_button(
                            "📥 Download Excel (Local)", 
                            data=output.getvalue(), 
                            file_name=f"Result_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    with exp_col2:
                        # Google Drive Save
                        if st.button("☁️ Save to Google Drive"):
                            if not drive_folder_target and input_type == "Google Drive":
                                target_id = drive_file_url.split('/')[-1] # Fallback to file's location if possible
                            else:
                                target_id = drive_folder_target
                                
                            sheet_url = save_to_google_sheets(creds, target_id, data_dict)
                            if "https" in sheet_url:
                                st.success(f"Spreadsheet created! [Open]({sheet_url})")
                            else:
                                st.error(f"Error: {sheet_url}")
                else:
                    st.text_area("Raw AI Output", result_text, height=300)

                progress.progress(100)
            except Exception as e:
                st.error(f"Error: {str(e)}")
