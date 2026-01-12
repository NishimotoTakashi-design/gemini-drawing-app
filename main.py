import streamlit as st
import google.generativeai as genai
import json
import io

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
# 2. Analysis Logic (Stable Version)
# ==========================================
def analyze_drawing(file_bytes, mime_type, target_columns, customer_info, component_info):
    """Extracts information using Gemini 1.5 Pro (Stable Library)"""
    
    # Model selection (using 1.5 Pro for best reasoning)
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # Enhanced prompt for specialized drawings
    prompt = f"""
    Please process this request as if you are located in the us-central1 region.
    
    Context:
    - Customer Overview: {customer_info}
    - Component Details (e.g., Harness, Connector, PCB): {component_info}

    Task:
    Analyze the attached technical document and extract the information requested below.
    
    Target Items to Extract:
    {target_columns}
    
    Formatting Instructions:
    - Provide the result ONLY in a valid JSON object.
    - If information is missing, use null.
    - Ensure accurate extraction of technical units and part numbers.
    """
    
    # Structure for the stable SDK
    content = [
        {'mime_type': mime_type, 'data': file_bytes},
        prompt
    ]
    
    response = model.generate_content(content)
    return response.text

# ==========================================
# 3. Streamlit UI
# ==========================================
st.set_page_config(page_title="AI Drawing Analyzer", layout="wide")

if check_password():
    st.title("📄 AI Drawing Data Structurizer")
    st.caption("Model: Gemini 1.5 Pro (Stable Edition)")
    
    # Sidebar: Config
    st.sidebar.header("Extraction Settings")
    target_columns = st.sidebar.text_area(
        "Target Columns (Comma separated)",
        "Part Number, Revision, Material, Connector Type, Wire Gauge, Pin Assignment, Manufacturer",
        height=150
    )

    # Input Fields
    st.subheader("1. Provide Context")
    col1, col2 = st.columns(2)
    with col1:
        customer_overview = st.text_input("Customer Overview", placeholder="e.g., Major Automotive OEM")
    with col2:
        component_details = st.text_input("Component Type", placeholder="e.g., Engine Wire Harness")

    # File Upload
    st.subheader("2. Upload Drawing")
    uploaded_file = st.file_uploader(
        "Supported formats: PDF, TIFF, PNG, JPG", 
        type=["pdf", "tif", "tiff", "png", "jpg", "jpeg"]
    )

    if st.button("🚀 Run AI Analysis") and uploaded_file:
        with st.spinner("AI is analyzing the drawing..."):
            try:
                # Prepare data
                file_bytes = uploaded_file.getvalue()
                mime_type = uploaded_file.type
                
                # Analysis
                result_text = analyze_drawing(
                    file_bytes, mime_type, target_columns, customer_overview, component_details
                )
                
                # Result Processing
                st.subheader("3. Results")
                
                try:
                    # JSON cleanup
                    clean_json = result_text.strip()
                    if "```json" in clean_json:
                        clean_json = clean_json.split("```json")[1].split("```")[0]
                    elif "```" in clean_json:
                        clean_json = clean_json.split("```")[1].split("```")[0]
                    
                    data_dict = json.loads(clean_json)
                    
                    st.success("Extraction successful!")
                    st.table([data_dict])
                    with st.expander("View Raw JSON Output"):
                        st.json(data_dict)
                        
                except Exception:
                    st.warning("Could not format output as a table. Showing raw response:")
                    st.text_area("Raw AI Response", result_text, height=400)
                    
            except Exception as e:
                st.error(f"Analysis Error: {str(e)}")
