import streamlit as st
import json
import re
import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Mechanic AI: Fleet Command", page_icon="⚡", layout="wide")

# Custom CSS for aesthetics (Badges, Headers, Cards)
st.markdown("""
<style>
    /* Force Light Mode Aesthetics for Cleanliness */
    .stApp { background-color: #f8f9fa; font-family: 'Inter', sans-serif; }
    
    /* Headers */
    h1 { color: #2C3E50 !important; font-weight: 700; margin-bottom: 0px; }
    .subtext { color: #666; font-size: 14px; margin-bottom: 20px; }
    
    /* Result Card Container */
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Badges */
    .badge-manual { 
        background-color: #d4edda; color: #155724; 
        padding: 4px 8px; border-radius: 4px; 
        font-weight: bold; font-size: 0.9em; 
        border: 1px solid #c3e6cb; 
    }
    .badge-ai { 
        background-color: #fff3cd; color: #856404; 
        padding: 4px 8px; border-radius: 4px; 
        font-weight: bold; font-size: 0.9em; 
        border: 1px solid #ffeeba; 
    }
    
    /* Sidebar Button Styling */
    .stButton>button { 
        width: 100%; 
        text-align: left; 
        border-radius: 6px; 
        height: auto; 
        padding: 10px; 
        border: 1px solid #eee;
    }
    .stButton>button:hover {
        border-color: #4CAF50;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SECURE API KEY
# ==========================================
# Try Secrets first, then Environment, then your hardcoded key (as last resort)
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") or "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"

if not api_key:
    st.error("🚨 API Key Missing! Please add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

# ==========================================
# 3. LOAD BRAIN (Vector Store)
# ==========================================
@st.cache_resource
def load_resources():
    print("🔄 Loading Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # CRITICAL: This must match the folder name in your screenshot ("faiss_db_index_test")
    index_path = "faiss_db_index_test" 
    
    try:
        if os.path.exists(index_path):
            print(f"✅ Found index at {index_path}")
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        else:
            print(f"❌ Index folder not found at: {index_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        return None

vector_store = load_resources()

if not vector_store:
    st.error("⚠️ Database Error: The folder `faiss_db_index_test` was not found. Please upload it to the app directory.")
    st.stop()

# ==========================================
# 4. LOGIC HANDLERS
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
Answer based on general mechanical knowledge. Be concise.
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
# 5. SIDEBAR (Preset Queries)
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/maintenance.png", width=60)
    st.header("Fleet Control")
    st.markdown("---")
    
    # Session State for Query Input
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    def set_query(q):
        st.session_state.query_input = q

    st.subheader("🚗 Ford F-150")
    if st.button("Suspension Torque"): set_query("Torque specifications for front suspension (Car)")
    if st.button("Fluid Capacities"): set_query("Fluid capacities (Car)")

    st.subheader("✈️ F-16 Jet")
    if st.button("Engine Fire Proc."): set_query("Emergency procedure for engine fire on ground (F-16)")
    if st.button("Landing Gear Speed"): set_query("Landing gear extension speed limits (F-16)")
    
    st.subheader("🏍️ Ducati Bike")
    if st.button("Start Failure"): set_query("Troubleshooting engine starting failure (Bike)")
    if st.button("Chain Tension"): set_query("Chain tension adjustment (Bike)")
    
    st.markdown("---")
    st.caption("v3.0 • Multi-Modal RAG System")

# ==========================================
# 6. MAIN UI
# ==========================================
st.markdown("<h1>⚡ Mechanic AI <span style='font-size:20px; color:#aaa;'>FLEET COMMAND</span></h1>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Multi-Modal Retrieval System (Text + Visuals)</div>", unsafe_allow_html=True)

# Search Input
query = st.text_input("Ask a technical question...", value=st.session_state.query_input, placeholder="e.g. 'Tire pressure F-16'")

if st.button("Search Manuals", type="primary"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("⚙️ Analyzing Fleet Documents..."):
            result = process_query(query)

        # --- RENDER RESULTS ---
        
        # CASE 1: MANUAL DATA FOUND
        if result['type'] == 'manual':
            # Badge & Header
            st.markdown(f"""
            <div class='result-card'>
                <span class='badge-manual'>✓ OFFICIAL SOURCE: {result['source']}</span>
                <h3 style='margin-top:15px; color:#333;'>Specifications Found</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # --- USE NATIVE DATAFRAME (Bulletproof Display) ---
            df = pd.DataFrame(result['specs'])
            
            # Formatting Columns if they exist
            if not df.empty:
                # Rename columns for cleaner UI
                column_map = {
                    'component': 'Component / Step',
                    'value': 'Value / Action',
                    'unit': 'Unit',
                    'description': 'Notes'
                }
                # Only rename columns that actually exist in the data
                df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
                
                # Display Interactive Table
                st.dataframe(
                    df, 
                    use_container_width=True, 
                    hide_index=True
                )
            else:
                st.warning("Data structure returned empty.")
            
            # --- RENDER IMAGES ---
            if result.get('images'):
                st.markdown("### 📷 Visual Reference")
                
                # Create a gallery layout
                cols = st.columns(min(3, len(result['images'])))
                for idx, img_path in enumerate(result['images']):
                    with cols[idx % 3]:
                        if os.path.exists(img_path):
                            st.image(img_path, caption=f"Figure {idx+1}", use_container_width=True)
                        else:
                            st.warning(f"⚠️ Image missing: {img_path}")

        # CASE 2: GENERAL KNOWLEDGE FALLBACK
        elif result['type'] == 'general':
            st.markdown(f"""
            <div class='result-card'>
                <span class='badge-ai'>⚠ GENERAL KNOWLEDGE</span>
                <div style='margin-top:15px; color:#444; line-height:1.6; font-size:16px;'>
                    {result['content']}
                </div>
                <hr style='border:0; border-top:1px solid #eee; margin:15px 0;'>
                <div style='font-size:12px; color:#888;'>
                    *Data not found in official fleet documents. Response generated by AI logic.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # CASE 3: CHIT CHAT
        elif result['type'] == 'chat':
            st.info(result['content'])