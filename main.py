import streamlit as st
import google.generativeai as genai
import json

# ==========================================
# 1. Security Settings
# ==========================================
def check_password():
    """Simple password authentication for access control"""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.title("🔒 Access Restricted")
        password = st.text_input("Please enter the application password", type="password")
        if st.button("Login"):
            if password == st.secrets.get("APP_PASSWORD", "admin123"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        return False
    return True

# Configure Gemini API
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("API Key not found in Secrets.")

# ==========================================
# 2. Main Logic (Drawing Analysis)
# ==========================================
def analyze_drawing(file_bytes, mime_type, target_columns, customer_info, component_info):
    """Extracts information from drawings using Gemini 1.5 Pro"""
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # Enhanced prompt including Customer and Component context
    prompt = f"""
    Context:
    - Customer Overview: {customer_info}
    - Component Details (Harness/Connectors/etc.): {component_info}

    Task:
    Analyze the attached drawing and extract the following information in JSON format.
    Required Columns: {target_columns}
    
    Instructions:
    - If a value is not found, return null.
    - Return ONLY the JSON object.
    """
    
    # Structure for Gemini API
    content = [
        {"mime_type": mime_type, "data": file_bytes},
        prompt
    ]
    
    response = model.generate_content(content)
    return response.text

# ==========================================
# 3. UI Construction
# ==========================================
st.set_page_config(page_title="AI Drawing Analyzer", layout="wide")

if check_password():
    st.title("📄 AI Drawing Data Structurizer")
    st.write("Extract structured data from technical drawings (PDF, TIFF, Images) using Google Gemini.")

    # Sidebar: Configurations
    st.sidebar.header("Configuration")
    input_method = st.sidebar.radio("Input Method", ("Local Upload", "Google Drive Path"))
    
    st.sidebar.subheader("Extraction Settings")
    target_columns = st.sidebar.text_area(
        "Target Columns (Comma separated)",
        "Part Number, Rev, Material, Manufacturer, Connector Type, Wire Gauge, Pin Count"
    )

    # Main Area: Inputs
    col1, col2 = st.columns(2)
    with col1:
        customer_overview = st.text_input("Customer Overview", placeholder="e.g., Automotive OEM, Aerospace client")
    with col2:
        component_details = st.text_input("Component Context", placeholder="e.g., Wire Harness, ECU Connector")

    file_to_process = None
    mime_type = None

    if input_method == "Local Upload":
        uploaded_file = st.file_uploader(
            "Upload Drawing", 
            type=["png", "jpg", "jpeg", "pdf", "tif", "tiff"]
        )
        if uploaded_file:
            file_to_process = uploaded_file.getvalue()
            mime_type = uploaded_file.type
            st.success(f"File '{uploaded_file.name}' ready for processing.")

    else:
        drive_path = st.text_input("Enter Google Drive Folder Path or File ID")
        st.info("Note: Google Drive integration requires Service Account credentials setup.")

    # Execution
    if st.button("Run Extraction") and file_to_process:
        with st.spinner("Analyzing drawing with Gemini..."):
            try:
                result_text = analyze_drawing(
                    file_to_process, 
                    mime_type, 
                    target_columns, 
                    customer_overview, 
                    component_details
                )
                
                st.subheader("Extraction Result")
                
                # Clean and parse JSON
                try:
                    clean_json = result_text.strip().replace("```json", "").replace("```", "")
                    data_dict = json.loads(clean_json)
                    st.table([data_dict])
                    st.json(data_dict)
                except:
                    st.text_area("Raw AI Response", result_text, height=300)
                    st.warning("Could not parse result into a table. Check the raw text above.")
                    
            except Exception as e:
                st.error(f"Error during analysis: {e}")
