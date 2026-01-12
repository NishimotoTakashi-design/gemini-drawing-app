import streamlit as st
import json
import io

# Import the new Google GenAI SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("Error: 'google-genai' library not found. Please ensure it's in requirements.txt and reboot the app.")

# ==========================================
# 1. Access Control (Security)
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
            # Set your password in Streamlit Cloud Secrets as APP_PASSWORD
            if password == st.secrets.get("APP_PASSWORD", "admin123"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        return False
    return True

# Initialize Gemini Client with us-central1
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("API Key missing in Secrets.")
        st.stop()
    
    # Initialize client with the specific location
    return genai.Client(
        api_key=api_key,
        location="us-central1"
    )

# ==========================================
# 2. Main Logic (Drawing Analysis)
# ==========================================
def analyze_drawing(client, file_bytes, mime_type, target_columns, customer_info, component_info):
    """Extracts information using Gemini 2.5 Pro"""
    
    # Detailed prompt for Harness/Connector extraction
    prompt = f"""
    Context:
    - Customer Overview: {customer_info}
    - Component Details (e.g., Harness, Connector, PCB): {component_info}

    Task:
    Analyze the attached technical drawing and extract the specific information requested below.
    
    Target Items to Extract:
    {target_columns}
    
    Formatting Instructions:
    - Provide the result ONLY in a valid JSON object.
    - If a specific piece of information is not found in the drawing, set the value to null.
    - Ensure keys match the 'Target Items' exactly.
    """
    
    # Generate content using Gemini 2.5 Pro
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            prompt
        ]
    )
    return response.text

# ==========================================
# 3. Streamlit UI
# ==========================================
st.set_page_config(page_title="Gemini 2.5 Pro Drawing Analyzer", layout="wide")

if check_password():
    client = get_gemini_client()
    
    st.title("📄 AI Drawing Data Structurizer")
    st.caption("Powered by Gemini 2.5 Pro | Region: us-central1")
    
    # Sidebar: Config
    st.sidebar.header("Data Extraction Settings")
    target_columns = st.sidebar.text_area(
        "Target Columns (Comma separated)",
        "Part Number, Revision, Material, Connector Type, Wire Gauge, Pin Assignment, Manufacturer, Weight",
        height=150
    )

    # Input Fields for Context
    st.subheader("1. Provide Context")
    col1, col2 = st.columns(2)
    with col1:
        customer_overview = st.text_input("Customer Overview", placeholder="e.g., Major Automotive OEM")
    with col2:
        component_details = st.text_input("Component Type", placeholder="e.g., Engine Wire Harness, Board-to-Board Connector")

    # File Upload
    st.subheader("2. Upload Drawing")
    uploaded_file = st.file_uploader(
        "Supported formats: PDF, TIFF, PNG, JPG", 
        type=["pdf", "tif", "tiff", "png", "jpg", "jpeg"]
    )

    if st.button("🚀 Run AI Analysis") and uploaded_file:
        with st.spinner("Gemini is analyzing the drawing... (This may take a moment for large files)"):
            try:
                # Prepare data
                file_bytes = uploaded_file.getvalue()
                mime_type = uploaded_file.type
                
                # Analysis
                result_text = analyze_drawing(
                    client, file_bytes, mime_type, target_columns, customer_overview, component_details
                )
                
                # Output
                st.subheader("3. Results")
                
                try:
                    # Clean the AI output to ensure it's valid JSON
                    clean_json = result_text.strip()
                    if clean_json.startswith("```json"):
                        clean_json = clean_json[7:-3]
                    elif clean_json.startswith("```"):
                        clean_json = clean_json[3:-3]
                    
                    data_dict = json.loads(clean_json)
                    
                    # Display as Table and JSON
                    st.success("Extraction successful!")
                    st.table([data_dict])
                    with st.expander("View Raw JSON Output"):
                        st.json(data_dict)
                        
                except Exception as json_err:
                    st.warning("Could not format output as a table. Showing raw response:")
                    st.text_area("Raw AI Response", result_text, height=400)
                    
            except Exception as e:
                st.error(f"Analysis Error: {str(e)}")
