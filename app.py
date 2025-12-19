import streamlit as st
import json
import re
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Mechanic AI: Fleet Command", page_icon="⚡", layout="wide")

# Custom CSS for "Pro" Look
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; font-family: 'Inter', sans-serif; }
    
    /* Header Styles */
    h1 { color: #2C3E50 !important; font-weight: 700; }
    h3 { color: #34495E !important; }
    
    /* Badge Styles */
    .badge-manual { background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; border: 1px solid #c3e6cb; }
    .badge-ai { background-color: #fff3cd; color: #856404; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; border: 1px solid #ffeeba; }
    
    /* Card/Table Styles */
    .result-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #e0e0e0; }
    
    /* Sidebar Buttons */
    .stButton>button { width: 100%; border-radius: 6px; text-align: left; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SECURE API KEY
# ==========================================
# Try to get key from Secrets (Cloud) or Environment (Local)
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("🚨 API Key Missing! Please add `GROQ_API_KEY` to Streamlit Secrets.")
    st.stop()

# ==========================================
# 3. LOAD BRAIN (Vector Store)
# ==========================================
@st.cache_resource
def load_resources():
    # Use the same model you used for ingestion
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Path to your FAISS index folder
    # IMPORTANT: Ensure you uploaded the folder 'faiss_db_index' containing .faiss and .pkl files
    index_path = "faiss_db_index" 
    
    try:
        if os.path.exists(index_path):
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return None

vector_store = load_resources()

if not vector_store:
    st.error("⚠️ Database not found! Please upload the 'faiss_db_index' folder.")
    st.stop()

# ==========================================
# 4. LOGIC HANDLERS (Same as Notebook)
# ==========================================
llm = ChatGroq(temperature=0.1, model_name="llama-3.1-8b-instant", api_key=api_key)

rag_prompt = """
You are a technical data extractor. Analyze Context for: '{question}'.
RULES:
1. Return ONLY valid JSON.
2. Structure: {{ "specs": [ {{ "component": "...", "value": "...", "unit": "...", "description": "..." }} ] }}
3. If NOT found, return {{ "specs": [] }}
Context:
{context}
"""

general_prompt = """
The user asked: '{question}'.
We searched the manuals but found NO specific match.
Answer based on general mechanical knowledge. Be concise and helpful.
"""

def clean_json(text):
    try:
        text = str(text).replace("```json", "").replace("```", "")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match: return json.loads(match.group(0))
    except: pass
    return {"specs": []}

def detect_filter(query):
    q = query.lower()
    if any(x in q for x in ["f-16", "jet", "aircraft"]): return {"vehicle_type": "jet"}
    if any(x in q for x in ["bike", "motorcycle", "ducati"]): return {"vehicle_type": "bike"}
    if any(x in q for x in ["car", "ford", "f-150"]): return {"vehicle_type": "car"}
    return None

def process_query(query):
    # 1. Chit Chat
    if query.lower().strip() in ["hi", "hello", "help"]:
        return {"type": "chat", "content": "👋 **System Ready.** I can access manuals for F-16 Jets, Ducati Bikes, and Ford F-150s."}

    # 2. Search
    active_filter = detect_filter(query)
    try:
        if active_filter: docs = vector_store.similarity_search(query, k=4, filter=active_filter)
        else: docs = vector_store.similarity_search(query, k=4)
    except: docs = []

    # 3. Extract
    if docs:
        context = "\n\n".join([d.page_content for d in docs])
        
        # Get Images (Deduplicated)
        found_images = []
        seen = set()
        for d in docs:
            path = d.metadata.get("image_path")
            if path and path not in seen:
                found_images.append(path)
                seen.add(path)

        chain = ChatPromptTemplate.from_template(rag_prompt) | llm
        response = chain.invoke({"context": context, "question": query})
        data = clean_json(response.content)

        if data.get("specs"):
            return {
                "type": "manual",
                "specs": data["specs"],
                "images": found_images,
                "source": active_filter['vehicle_type'].upper() if active_filter else "DOCS"
            }

    # 4. Fallback
    gen_chain = ChatPromptTemplate.from_template(general_prompt) | llm
    res = gen_chain.invoke({"question": query})
    return {"type": "general", "content": res.content}

# ==========================================
# 5. SIDEBAR (The "Chip" Interface)
# ==========================================
with st.sidebar:
    st.header("⚡ Fleet Control")
    st.markdown("Select a vehicle query:")
    
    # Initialize session state for query
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    # Helper to set query
    def set_query(q):
        st.session_state.query_input = q

    st.caption("🚗 **Ford F-150**")
    if st.button("Suspension Torque Specs (Car)"): set_query("Torque specifications for front suspension (Car)")
    if st.button("Fluid Capacities (Car)"): set_query("Fluid capacities (Car)")
    
    st.caption("✈️ **F-16 Jet**")
    if st.button("Engine Fire Procedure (Jet)"): set_query("Emergency procedure for engine fire on ground (F-16)")
    if st.button("Gear Speed Limits (Jet)"): set_query("Landing gear extension speed limits (F-16)")
    
    st.caption("🏍️ **Ducati Bike**")
    if st.button("Start Failure (Bike)"): set_query("Troubleshooting engine starting failure (Bike)")
    if st.button("Chain Tension (Bike)"): set_query("Chain tension adjustment (Bike)")
    
    st.divider()
    st.markdown("v3.0 • Multi-Modal RAG")

# ==========================================
# 6. MAIN UI
# ==========================================
st.title("Mechanic AI: Fleet Command")
st.markdown("Ask about technical specs, procedures, or diagnostics.")

# Search Input
query = st.text_input("Enter your question:", value=st.session_state.query_input, placeholder="e.g., 'Tire pressure F-16'")

if st.button("Search Manuals", type="primary"):
    if not query:
        st.warning("Please enter a query first.")
    else:
        with st.spinner(f"🔍 Scanning Fleet Documents for '{query}'..."):
            result = process_query(query)

        # --- DISPLAY RESULTS ---
        
        # CASE 1: MANUAL DATA (Success)
        if result['type'] == 'manual':
            st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
            st.markdown(f"<span class='badge-manual'>✅ OFFICIAL SOURCE: {result['source']}</span>", unsafe_allow_html=True)
            st.markdown("### Specifications Found")
            
            # Display Table
            st.table(result['specs'])
            
            # Display Images
            if result.get('images'):
                st.markdown("#### 📷 Visual Reference")
                cols = st.columns(len(result['images']))
                for idx, img_path in enumerate(result['images']):
                    # Handle image path (Streamlit needs correct relative path)
                    if os.path.exists(img_path):
                        st.image(img_path, caption=f"Figure {idx+1}", use_container_width=True)
                    else:
                        st.warning(f"⚠️ Image found in index but missing on disk: {img_path}")
            
            st.markdown("</div>", unsafe_allow_html=True)

        # CASE 2: GENERAL KNOWLEDGE (Fallback)
        elif result['type'] == 'general':
            st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
            st.markdown(f"<span class='badge-ai'>⚠️ AI GENERATED (Not in Manual)</span>", unsafe_allow_html=True)
            st.warning("We could not find this specific data in the uploaded manuals. Here is a general answer based on AI knowledge:")
            st.info(result['content'])
            st.markdown("</div>", unsafe_allow_html=True)

        # CASE 3: CHIT CHAT
        elif result['type'] == 'chat':
            st.markdown(f"<div class='result-card'>{result['content']}</div>", unsafe_allow_html=True)