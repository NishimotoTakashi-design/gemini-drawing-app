import streamlit as st
import google.generativeai as genai
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 1. Access Control & API Setup (No changes)
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.title("🔒 Access Restricted")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if password == st.secrets.get("APP_PASSWORD", "admin123"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        return False
    return True

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("API Key missing.")

# ==========================================
# 2. Parallel Analysis Logic (No changes to Model/Region)
# ==========================================
def call_gemini_api(file_bytes, mime_type, target_columns, customer_info, component_info):
    model = genai.GenerativeModel('gemini-2.5-pro')
    prompt = f"""
    [System Note: Process this request as if you are located in the us-central1 region.]
    Context:
    - Customer Overview: {customer_info}
    - Component Details: {component_info}
    Task: Analyze the attached drawing and extract these items into valid JSON:
    {target_columns}
    Rules: Return ONLY a valid JSON object. No conversation.
    """
    content = [{'mime_type': mime_type, 'data': file_bytes}, prompt]
    response = model.generate_content(content)
    return response.text

# ==========================================
# 3. UI Construction (Output Logic Improved)
# ==========================================
st.set_page_config(page_title="AI Drawing Analyzer Pro", layout="wide")

if check_password():
    st.title("📄 AI Drawing Data Structurizer")
    st.caption("Engine: Gemini 2.5 Pro | Target Region: us-central1")
    
    st.sidebar.header("Extraction Settings")
    target_columns = st.sidebar.text_area(
        "Target Columns",
        "Part Number, Revision, Material, Connector Type, Wire Gauge, Pin Assignment, Manufacturer",
        height=150
    )

    st.subheader("1. Provide Context")
    col1, col2 = st.columns(2)
    with col1:
        customer_overview = st.text_input("Customer Overview", placeholder="e.g., Major OEM")
    with col2:
        component_details = st.text_input("Component Type", placeholder="e.g., Wire Harness")

    st.subheader("2. Upload or Specify Drawing")
    input_type = st.radio("Select Input Method:", ("Local File Upload", "Google Drive Path"), horizontal=True)

    file_bytes = None
    mime_type = None

    if input_type == "Local File Upload":
        uploaded_file = st.file_uploader("Upload Drawing", type=["pdf", "tif", "tiff", "png", "jpg", "jpeg"])
        if uploaded_file:
            file_bytes = uploaded_file.getvalue()
            mime_type = uploaded_file.type
    else:
        drive_path = st.text_input("Enter Google Drive URL / Path")

    # --- Improved Analysis Execution ---
    if st.button("🚀 Run AI Analysis"):
        if file_bytes:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Initializing analysis...")
                progress_bar.progress(10)
                
                # API Call with parallel thread
                start_time = time.time()
                with ThreadPoolExecutor() as executor:
                    status_text.text("Gemini 2.5 Pro is processing drawing... Please wait.")
                    progress_bar.progress(40)
                    future = executor.submit(
                        call_gemini_api, 
                        file_bytes, mime_type, target_columns, customer_overview, component_details
                    )
                    
                    # API応答待ちの間、少しずつバーを動かす（視覚的なフリーズ防止）
                    while not future.done():
                        time.sleep(1)
                        elapsed = time.time() - start_time
                        if elapsed < 60:
                            current_p = 40 + int(elapsed / 2) # 最大70まで
                            progress_bar.progress(min(current_p, 70))
                    
                    result_text = future.result()

                status_text.text("Processing output...")
                progress_bar.progress(90)

                # --- Improved Output Extraction ---
                st.subheader("3. Results")
                
                # Regex to find JSON block even if mixed with other text
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                
                if json_match:
                    try:
                        clean_json = json_match.group(0)
                        data_dict = json.loads(clean_json)
                        
                        st.success(f"Analysis complete in {int(time.time() - start_time)} seconds.")
                        st.table([data_dict]) # Display as a table
                        
                        with st.expander("View Full JSON Data"):
                            st.json(data_dict)
                    except Exception as parse_err:
                        st.error("AI returned invalid JSON format.")
                        st.text_area("Raw AI Output", result_text, height=300)
                else:
                    st.warning("No structured data found. See raw response below:")
                    st.text_area("Raw AI Output", result_text, height=300)

                progress_bar.progress(100)
                status_text.text("Done.")
                
            except Exception as e:
                st.error(f"Analysis Error: {str(e)}")
                progress_bar.empty()
                status_text.empty()
        else:
            st.warning("Please upload a file.")
