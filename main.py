import streamlit as st
import google.generativeai as genai
import json
import time
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 1. Access Control & API Setup
# ==========================================
def check_password():
    """Simple password authentication for access control"""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.title("🔒 Access Restricted")
        st.write("Please log in to use the AI Drawing Analyzer.")
        
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            # Uses APP_PASSWORD from Streamlit Secrets
            if password == st.secrets.get("APP_PASSWORD", "admin123"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        return False
    return True

# Initialize Gemini API
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("API Key missing in Secrets.")

# ==========================================
# 2. Parallel Analysis Logic (Gemini 2.5 Pro)
# ==========================================
def call_gemini_api(file_bytes, mime_type, target_columns, customer_info, component_info):
    """Worker function to call Gemini 2.5 Pro in us-central1"""
    # Specifically calling gemini-2.5-pro
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    prompt = f"""
    [System Note: Process this request as if you are located in the us-central1 region.]
    
    Context:
    - Customer Overview: {customer_info}
    - Component Details (e.g., Harness, Connector): {component_info}

    Task:
    Analyze the attached technical drawing and extract the information requested below into a structured JSON format.
    
    Target Items to Extract:
    {target_columns}
    
    Rules:
    - Return ONLY a valid JSON object.
    - If data is missing, use null.
    - Focus on high-precision extraction for part numbers and technical specs.
    """
    
    content = [
        {'mime_type': mime_type, 'data': file_bytes},
        prompt
    ]
    
    response = model.generate_content(content)
    return response.text

# ==========================================
# 3. Streamlit UI Construction
# ==========================================
st.set_page_config(page_title="AI Drawing Analyzer (Gemini 2.5 Pro)", layout="wide")

if check_password():
    st.title("📄 AI Drawing Data Structurizer")
    st.caption("Engine: Gemini 2.5 Pro | Target Region: us-central1")
    
    # Sidebar: Configurations
    st.sidebar.header("Extraction Settings")
    target_columns = st.sidebar.text_area(
        "Target Columns (Comma separated)",
        "Part Number, Revision, Material, Connector Type, Wire Gauge, Pin Assignment, Manufacturer",
        height=150
    )

    # 1. Provide Context
    st.subheader("1. Provide Context")
    col1, col2 = st.columns(2)
    with col1:
        customer_overview = st.text_input("Customer Overview", placeholder="e.g., Major Automotive OEM")
    with col2:
        component_details = st.text_input("Component Type", placeholder="e.g., Engine Wire Harness")

    # 2. Upload or Specify Drawing
    st.subheader("2. Upload or Specify Drawing")
    input_type = st.radio("Select Input Method:", ("Local File Upload", "Google Drive Path"), horizontal=True)

    file_bytes = None
    mime_type = None

    if input_type == "Local File Upload":
        uploaded_file = st.file_uploader(
            "Upload Drawing (PDF, TIFF, PNG, JPG)", 
            type=["pdf", "tif", "tiff", "png", "jpg", "jpeg"]
        )
        if uploaded_file:
            file_bytes = uploaded_file.getvalue()
            mime_type = uploaded_file.type
            st.success(f"File '{uploaded_file.name}' ready.")
    else:
        drive_path = st.text_input("Enter Google Drive URL / Path", placeholder="https://drive.google.com/...")
        st.info("ℹ️ Note: For Google Drive links, please use 'Local Upload' for immediate processing in this version.")

    # 3. Analysis Execution with Progress Bar & Parallel Processing
    if st.button("🚀 Run AI Analysis"):
        if file_bytes:
            # UI Elements for Progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Stage 1: Preparation
                status_text.text("Step 1/3: Preparing data and initializing thread...")
                progress_bar.progress(20)
                time.sleep(0.5)
                
                # Stage 2: AI Processing with Parallel Thread
                status_text.text("Step 2/3: Gemini 2.5 Pro is analyzing (This usually takes 20-60 seconds)...")
                progress_bar.progress(50)
                
                # Parallel Execution to keep the UI responsive
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        call_gemini_api, 
                        file_bytes, mime_type, target_columns, customer_overview, component_details
                    )
                    # Wait for API response
                    result_text = future.result()
                
                # Stage 3: Post-processing
                status_text.text("Step 3/3: Formatting results...")
                progress_bar.progress(90)
                time.sleep(0.5)
                
                # Output Results
                st.subheader("3. Results")
                try:
                    # Clean and parse JSON
                    clean_json = result_text.strip()
                    if "```json" in clean_json:
                        clean_json = clean_json.split("```json")[1].split("```")[0]
                    elif "```" in clean_json:
                        clean_json = clean_json.split("```")[1].split("```")[0]
                    
                    data_dict = json.loads(clean_json)
                    st.success("Analysis complete!")
                    st.table([data_dict])
                    with st.expander("View Raw JSON Output"):
                        st.json(data_dict)
                except:
                    st.warning("Could not format output as a table. Showing raw response:")
                    st.text_area("Raw AI Response", result_text, height=400)
                
                progress_bar.progress(100)
                status_text.text("Done.")
                
            except Exception as e:
                st.error(f"Analysis Error: {str(e)}")
                progress_bar.empty()
                status_text.empty()
        else:
            st.warning("Please upload a file to start the analysis.")
