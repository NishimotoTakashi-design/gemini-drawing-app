import streamlit as st
import google.generativeai as genai
import json
import io

# ==========================================
# 1. Access Control & API Setup
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

# Initialize Gemini API
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("API Key missing in Secrets.")

# ==========================================
# 2. Analysis Logic (Gemini 2.5 Pro)
# ==========================================
def analyze_drawing(file_bytes, mime_type, target_columns, customer_info, component_info):
    """Extracts information using Gemini 2.5 Pro targeting us-central1 capabilities"""
    
    # Specify Gemini 2.5 Pro
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    # Prompt optimized for Gemini 2.5's reasoning
    prompt = f"""
    [System Note: Process this request with priority on us-central1 high-performance clusters.]
    
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
# 3. Streamlit UI
# ==========================================
st.set_page_config(page_title="AI Drawing Analyzer (Gemini 2.5 Pro)", layout="wide")

if check_password():
    st.title("📄 AI Drawing Data Structurizer")
    st.caption("Current Engine: Gemini 2.5 Pro | Target Region: us-central1")
    
    # Sidebar: Config
    st.sidebar.header("Extraction Settings")
    target_columns = st.sidebar.text_area(
        "Target Columns",
        "Part Number, Revision, Material, Connector Type, Wire Gauge, Pin Assignment, Manufacturer",
        height=150
    )

    # 1. Provide Context
    st.subheader("1. Provide Context")
    col1, col2 = st.columns(2)
    with col1:
        customer_overview = st.text_input("Customer Overview", placeholder="e.g., Automotive OEM")
    with col2:
        component_details = st.text_input("Component Type", placeholder="e.g., Door Wire Harness")

    # 2. Input Selection
    st.subheader("2. Upload or Specify Drawing")
    input_type = st.radio("Select Input Method:", ("Local File Upload", "Google Drive Path"), horizontal=True)

    file_bytes = None
    mime_type = None

    if input_type == "Local File Upload":
        uploaded_file = st.file_uploader(
            "Upload Drawing (PDF, TIFF, Images)", 
            type=["pdf", "tif", "tiff", "png", "jpg", "jpeg"]
        )
        if uploaded_file:
            file_bytes = uploaded_file.getvalue()
            mime_type = uploaded_file.type
    else:
        drive_path = st.text_input("Enter Google Drive URL / Path")
        st.info("ℹ️ Currently, for Google Drive links, please download and use 'Local Upload' to ensure immediate AI processing.")

    # 3. Run Analysis
    if st.button("🚀 Run AI Analysis"):
        if file_bytes:
            with st.spinner("Gemini 2.5 Pro is analyzing your drawing..."):
                try:
                    result_text = analyze_drawing(
                        file_bytes, mime_type, target_columns, customer_overview, component_details
                    )
                    
                    st.subheader("3. Results")
                    try:
                        # Clean and Parse JSON
                        clean_json = result_text.strip()
                        if "```json" in clean_json:
                            clean_json = clean_json.split("```json")[1].split("```")[0]
                        elif "```" in clean_json:
                            clean_json = clean_json.split("```")[1].split("```")[0]
                        
                        data_dict = json.loads(clean_json)
                        st.success("Analysis complete!")
                        st.table([data_dict])
                        with st.expander("View Raw JSON"):
                            st.json(data_dict)
                    except:
                        st.text_area("Raw Response", result_text, height=300)
                except Exception as e:
                    st.error(f"Analysis Error: {str(e)}")
        else:
            st.warning("Please upload a file first.")
