import streamlit as st
import json
import re
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI Bot", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    h1, h2, h3 { color: #2C3E50 !important; }
    .stButton>button { width: 100%; background-color: #2980B9; color: white; }
    .stButton>button:hover { background-color: #3498DB; color: white; }
    .stAlert { color: black !important; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# --- 2. SECURE API KEY ---
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("🚨 Key missing. Please add GROQ_API_KEY to Secrets in Dashboard.")
    st.stop()

# --- 3. LOAD BRAIN ---
@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        return FAISS.load_local("faiss_db_index_test", embeddings, allow_dangerous_deserialization=True)
    except:
        return None

vector_store = load_resources()

if not vector_store:
    st.error("⚠️ Database missing. Check 'faiss_db_index' on GitHub.")
    st.stop()

# --- 4. SIDEBAR (PRESETS) ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/engine.png", width=80)
    st.title("Mechanic AI")
    st.success("✅ System Online")
    
    st.markdown("### ⚡ Quick Queries")
    presets = [
        "Torque specifications for front suspension",
        "Brake caliper guide pin torque",
        "Wheel alignment specifications",
        "Engine oil capacity"
    ]
    
    if "query_input" not in st.session_state: 
        st.session_state.query_input = ""

    for p in presets:
        if st.button(p): 
            st.session_state.query_input = p

# --- 5. MAIN APP ---
st.title("🚗 Mechanic AI Bot")

query = st.text_input("Ask a technical question:", value=st.session_state.query_input)

if st.button("Extract Specifications", type="primary"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("⚙️ Analyzing manual..."):
            try:
                # A. Run AI
                llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=api_key)
                docs = vector_store.similarity_search(query, k=3)
                context = "\n".join([d.page_content for d in docs])
                
                # B. YOUR CUSTOM PROMPT
                prompt = ChatPromptTemplate.from_template("""
                You are a technical assistant. Extract ALL specifications for: '{question}'.

                CRITICAL INSTRUCTIONS:
                1. **Torque Tables**: Pattern is **Nm** -> **lb-ft** -> **lb-in**.
                2. **Fluids/Parts**: If no unit exists (like a part number), return null for unit.
                3. **General**: Extract EVERY row. Clean up values (numbers only for torque).

                Output JSON: {{ "specs": [ {{ "component": "...", "spec_type": "...", "value": "...", "unit": "..." }} ] }}
                If empty, return {{ "specs": [] }}

                Context:
                {context}
                """)
                
                response = (prompt | llm).invoke({"context": context, "question": query})
                
                # C. Robust Parsing
                clean_text = response.content.replace("```json", "").replace("```", "")
                try:
                    match = re.search(r"\{.*\}", clean_text, re.DOTALL)
                    json_str = match.group(0) if match else clean_text
                    data = json.loads(json_str)
                    specs = data.get("specs", [])
                    
                    if specs:
                        st.dataframe(specs, use_container_width=True)
                    else:
                        st.warning(f"No specific data found for '{query}'.")
                        
                except json.JSONDecodeError:
                    st.info("AI Response:")
                    st.write(response.content)

            except Exception as e:
                st.error(f"System Error: {e}")