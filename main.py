import streamlit as st
from google import genai
from google.genai import types
import json

# ==========================================
# 1. Security & Client Settings
# ==========================================
def check_password():
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

# Initialize Gemini Client with Region (us-central1)
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("API Key not found in Secrets.")
        st.stop()
    
    # Setting location to us-central1
    return genai.Client(
        api_key=api_key,
        location="us-central1"
    )

client = get_gemini_client()

# ==========================================
# 2. Main Logic (Drawing Analysis)
# ==========================================
def analyze_drawing(file_bytes, mime_type, target_columns, customer_info, component_info):
    """Extracts information using Gemini 2.5 Pro in us-central1"""
    
    # Instructions including context for Harness/Connectors
    prompt = f"""
    Context:
    - Customer Overview: {customer_info}
    - Component Details (e.g., Harness, Connector): {component_info}

    Task:
    Analyze the attached document and extract the following items in a JSON object.
    Target Items: {target_columns}
    
    Instructions:
    - If data is missing, use null.
    - Provide ONLY valid JSON.
    """
    
    # Call Gemini 2.5 Pro
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            prompt
        ]
    )
    return response.text

# ==========================================
# 3. UI Construction
# ==========================================
st.set_page_config(page_title="AI Drawing Analyzer (Gemini 2.5 Pro)", layout="wide")

if check_password():
    st.title("📄 AI Drawing Data Structurizer")
    st.info("Current Model: Gemini 2.5 Pro | Region: us-central1")
    
    st.sidebar.header("Configuration")
    target_columns = st.sidebar.text_area(
        "Target Columns",
        "Part Number, Revision, Material, Connector Type, Wire Gauge, Pin Assignment, Manufacturer"
    )

    col1, col2 = st.columns(2)
    with col1:
        customer_overview = st.text_input("Customer Overview", placeholder="e.g., Automotive OEM")
    with col2:
        component_details = st.text_input("Component Context", placeholder="e.g., Wire Harness for Door")

    uploaded_file = st.file_uploader(
        "Upload Drawing (PDF, TIFF, Image)", 
        type=["pdf", "tif", "tiff", "png", "jpg", "jpeg"]
    )

    if st.button("Run Extraction") and uploaded_file:
        with st.spinner("Analyzing with Gemini 2.5 Pro..."):
            try:
                # Get file info
                file_bytes = uploaded_file.getvalue()
                mime_type = uploaded_file.type
                
                # Execute analysis
                result_text = analyze_drawing(
                    file_bytes, mime_type, target_columns, customer_overview, component_details
                )
                
                # Result Display
                st.subheader("Results")
                try:
                    # JSON Cleanup (handling potential markdown formatting)
                    clean_json = result_text.strip().replace("```json", "").replace("```", "")
                    data_dict = json.loads(clean_json)
                    st.table([data_dict])
                    st.json(data_dict)
                except:
                    st.text_area("Raw Response", result_text, height=300)
                    
            except Exception as e:
                st.error(f"Error: {e}")
