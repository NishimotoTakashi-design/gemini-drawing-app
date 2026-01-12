import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 1. Vertex AI Auth (No Password Required)
# ==========================================
@st.cache_resource
def init_vertex_ai():
    try:
        # Construct credentials from Streamlit Secrets
        if "gcp_service_account" not in st.secrets:
            st.error("GCP Service Account info not found in Secrets.")
            return False
            
        info = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(info)
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
# 2. Parallel Analysis Logic
# ==========================================
def call_gemini_vertex(file_bytes, mime_type, target_instructions, customer_info, component_info):
    # Using Gemini 1.5 Pro (Optimized for technical drawings)
    model = GenerativeModel("gemini-1.5-pro-002")
    
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
    doc_part = Part.from_data(data=file_bytes, mime_type=mime_type)
    response = model.generate_content([doc_part, prompt])
    return response.text

# ==========================================
# 3. UI Construction
# ==========================================
st.set_page_config(page_title="AI Drawing Analyzer Pro", layout="wide")

# App starts directly without password check
if init_vertex_ai():
    st.title("📄 AI Drawing Data Structurizer")
    st.caption("Engine: Vertex AI (Gemini 1.5 Pro Optimized) | Region: us-central1")

    # --- 1. Provide Context & Extraction Settings ---
    st.subheader("1. Provide Context & Extraction Settings")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        customer_overview = st.text_input("Customer Overview", placeholder="e.g., Major OEM")
    with col_c2:
        component_details = st.text_input("Component Type", placeholder="e.g., Wire Harness")

    st.write("### Extraction Settings")
    
    # Session state for dynamic rows
    if 'rows' not in st.session_state:
        st.session_state.rows = [
            {"item": "Part Number", "guide": "Found in title block at bottom right"},
            {"item": "Material", "guide": "Found in general notes section"}
        ]

    # Row controls
    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    if col_btn1.button("➕ Add Row"):
        st.session_state.rows.append({"item": "", "guide": ""})
    if col_btn2.button("➖ Remove Last Row") and len(st.session_state.rows) > 0:
        st.session_state.rows.pop()

    # Display input rows
    extracted_instructions = []
    for i, row in enumerate(st.session_state.rows):
        r_col1, r_col2 = st.columns([1, 2])
        with r_col1:
            item_name = st.text_input(f"Item Name {i+1}", value=row['item'], key=f"item_{i}")
        with r_col2:
            item_guide = st.text_input(f"Extraction Guide {i+1}", value=row['guide'], key=f"guide_{i}")
        st.session_state.rows[i] = {"item": item_name, "guide": item_guide}
        if item_name:
            extracted_instructions.append(f"- {item_name}: Find it based on: {item_guide}")

    target_instructions_str = "\n".join(extracted_instructions)

    # --- 2. Upload Drawing ---
    st.subheader("2. Upload Drawing")
    uploaded_file = st.file_uploader("Upload Drawing", type=["pdf", "tif", "tiff", "png", "jpg", "jpeg"])
    
    file_bytes = None
    mime_type = None
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        mime_type = uploaded_file.type

    # --- 3. Run Analysis ---
    if st.button("🚀 Run AI Analysis"):
        if file_bytes and target_instructions_str:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                start_time = time.time()
                status_text.text("Initializing analysis...")
                progress_bar.progress(10)

                with ThreadPoolExecutor() as executor:
                    status_text.text("Vertex AI is analyzing (us-central1)... Please wait.")
                    progress_bar.progress(40)
                    future = executor.submit(
                        call_gemini_vertex, 
                        file_bytes, mime_type, target_instructions_str, customer_overview, component_details
                    )
                    
                    while not future.done():
                        time.sleep(1)
                        elapsed = time.time() - start_time
                        if elapsed < 120:
                            progress_bar.progress(min(40 + int(elapsed / 2), 90))
                    
                    result_text = future.result()

                status_text.text("Processing results...")
                progress_bar.progress(95)

                st.subheader("3. Results")
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                
                if json_match:
                    try:
                        data_dict = json.loads(json_match.group(0))
                        st.success(f"Analysis Complete in {int(time.time() - start_time)} seconds!")
                        st.table([data_dict])
                        with st.expander("Raw JSON Data"):
                            st.json(data_dict)
                    except:
                        st.error("JSON parsing error.")
                        st.text_area("Raw Response", result_text, height=300)
                else:
                    st.warning("No JSON data extracted.")
                    st.text_area("Raw Response", result_text, height=300)

                progress_bar.progress(100)
                status_text.text("Done.")
                
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
                progress_bar.empty()
                status_text.empty()
        else:
            st.warning("Please upload a file and define extraction items.")
