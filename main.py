import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 1. Access Control & Vertex AI Auth
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

@st.cache_resource
def init_vertex_ai():
    """Initialize Vertex AI with Service Account from Secrets"""
    try:
        # Construct credentials from Streamlit Secrets
        info = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(info)
        
        # Initialize Vertex AI for us-central1
        vertexai.init(
            project=info["project_id"],
            location="us-central1",
            credentials=creds
        )
        return True
    except Exception as e:
        st.error(f"Failed to initialize Vertex AI: {e}")
        return False

# ==========================================
# 2. Parallel Analysis Logic (Vertex AI)
# ==========================================
def call_gemini_vertex(file_bytes, mime_type, target_instructions, customer_info, component_info):
    # Using Gemini 2.5 Pro (Vertex AI uses 2.5 Pro)
    model = GenerativeModel("gemini-2.5-pro")
    
    prompt = f"""
    Context:
    - Customer Overview: {customer_info}
    - Component Details: {component_info}
    
    Task: Analyze the attached drawing and extract the following items into a valid JSON object.
    Specific Extraction Instructions:
    {target_instructions}
    
    Rules: 
    - Return ONLY a valid JSON object.
    - If information is missing, use null.
    """
    
    # Create the document part
    doc_part = Part.from_data(data=file_bytes, mime_type=mime_type)
    
    response = model.generate_content([doc_part, prompt])
    return response.text

# ==========================================
# 3. UI Construction
# ==========================================
st.set_page_config(page_title="AI Drawing Analyzer (Vertex AI)", layout="wide")

if check_password():
    if init_vertex_ai():
        st.title("📄 AI Drawing Data Structurizer")
        st.caption("Engine: Vertex AI (Gemini 2.5 Pro) | Region: us-central1")

        # --- 1. Provide Context & Extraction Settings ---
        st.subheader("1. Provide Context & Extraction Settings")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            customer_overview = st.text_input("Customer Overview", placeholder="e.g., Major OEM")
        with col_c2:
            component_details = st.text_input("Component Type", placeholder="e.g., Wire Harness")

        st.write("### Extraction Settings")
        if 'rows' not in st.session_state:
            st.session_state.rows = [{"item": "Part Number", "guide": "Title block"}, {"item": "Material", "guide": "Notes"}]

        col_btn1, col_btn2, _ = st.columns([1, 1, 4])
        if col_btn1.button("➕ Add Row"):
            st.session_state.rows.append({"item": "", "guide": ""})
        if col_btn2.button("➖ Remove Last Row") and len(st.session_state.rows) > 0:
            st.session_state.rows.pop()

        extracted_instructions = []
        for i, row in enumerate(st.session_state.rows):
            r_col1, r_col2 = st.columns([1, 2])
            with r_col1:
                item_name = st.text_input(f"Item Name {i+1}", value=row['item'], key=f"item_{i}")
            with r_col2:
                item_guide = st.text_input(f"Extraction Guide {i+1}", value=row['guide'], key=f"guide_{i}")
            st.session_state.rows[i] = {"item": item_name, "guide": item_guide}
            if item_name:
                extracted_instructions.append(f"- {item_name}: {item_guide}")

        target_instructions_str = "\n".join(extracted_instructions)

        # --- 2. Input Selection ---
        st.subheader("2. Upload Drawing")
        uploaded_file = st.file_uploader("Upload Drawing", type=["pdf", "tif", "tiff", "png", "jpg", "jpeg"])

        # --- 3. Run Analysis ---
        if st.button("🚀 Run AI Analysis") and uploaded_file:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                start_time = time.time()
                file_bytes = uploaded_file.getvalue()
                mime_type = uploaded_file.type

                with ThreadPoolExecutor() as executor:
                    status_text.text("Vertex AI is processing... (us-central1)")
                    progress_bar.progress(40)
                    future = executor.submit(
                        call_gemini_vertex, 
                        file_bytes, mime_type, target_instructions_str, customer_overview, component_details
                    )
                    
                    while not future.done():
                        time.sleep(1)
                        elapsed = time.time() - start_time
                        progress_bar.progress(min(40 + int(elapsed), 85))
                    
                    result_text = future.result()

                status_text.text("Formatting results...")
                progress_bar.progress(95)

                st.subheader("3. Results")
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                
                if json_match:
                    data_dict = json.loads(json_match.group(0))
                    st.success("Analysis Complete!")
                    st.table([data_dict])
                    with st.expander("JSON Output"):
                        st.json(data_dict)
                else:
                    st.text_area("Raw AI Output", result_text, height=300)

                progress_bar.progress(100)
                status_text.text("Done.")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
