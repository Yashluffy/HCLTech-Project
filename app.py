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

# Custom CSS to match your Notebook's "Pro" look
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; font-family: 'Inter', sans-serif; }
    
    /* Header */
    h1 { color: #2C3E50 !important; font-weight: 700; margin-bottom: 0px; }
    .subtext { color: #666; font-size: 14px; margin-bottom: 20px; }
    
    /* Badges */
    .badge-manual { background-color: #d4edda; color: #155724; padding: 5px 10px; border-radius: 6px; font-weight: bold; border: 1px solid #c3e6cb; }
    .badge-ai { background-color: #fff3cd; color: #856404; padding: 5px 10px; border-radius: 6px; font-weight: bold; border: 1px solid #ffeeba; }
    
    /* Result Card */
    .result-container { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #e0e0e0; margin-top: 20px; }
    
    /* Table Styling */
    .styled-table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 15px; }
    .styled-table th { text-align: left; color: #555; background-color: #f1f3f5; padding: 10px; border-bottom: 2px solid #ddd; }
    .styled-table td { padding: 10px; border-bottom: 1px solid #eee; vertical-align: top; color: #333; }
    .val-text { font-weight: 700; color: #d63031; }
    .desc-text { color: #636e72; font-style: italic; font-size: 0.9em; }
    
    /* Sidebar Buttons */
    .stButton>button { width: 100%; text-align: left; border-radius: 6px; height: auto; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SECURE API KEY
# ==========================================
# Look for key in Secrets, Env Var, or default to the one you provided
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") or "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"

if not api_key:
    st.error("🚨 API Key Missing! Please add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

# ==========================================
# 3. LOAD BRAIN (Vector Store)
# ==========================================
@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_path = "faiss_db_index_test" 
    try:
        if os.path.exists(index_path):
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return None
    except Exception as e:
        return None

vector_store = load_resources()

if not vector_store:
    st.error("⚠️ Database not found! Please upload the 'faiss_db_index_test' folder.")
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
# 5. SIDEBAR (Presets)
# ==========================================
with st.sidebar:
    st.header("⚡ Fleet Control")
    
    # Initialize session state for query input
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    def set_query(q):
        st.session_state.query_input = q

    st.markdown("### 🚗 Car")
    if st.button("Suspension Torque Specs"): set_query("Torque specifications for front suspension (Car)")
    if st.button("Fluid Capacities"): set_query("Fluid capacities (Car)")

    st.markdown("### ✈️ Jet")
    if st.button("Engine Fire Procedure"): set_query("Emergency procedure for engine fire on ground (F-16)")
    if st.button("Landing Gear Speed"): set_query("Landing gear extension speed limits (F-16)")
    
    st.markdown("### 🏍️ Bike")
    if st.button("Start Failure"): set_query("Troubleshooting engine starting failure (Bike)")
    if st.button("Chain Tension"): set_query("Chain tension adjustment (Bike)")

# ==========================================
# 6. MAIN UI
# ==========================================
st.markdown("<h1>⚡ Mechanic AI <span style='font-size:20px; color:#aaa;'>FLEET COMMAND</span></h1>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Multi-Modal Retrieval System (v3.0)</div>", unsafe_allow_html=True)

# Search Input
query = st.text_input("Ask a question...", value=st.session_state.query_input, placeholder="e.g. 'Tire pressure F-16'")

if st.button("Search Manuals", type="primary"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("⚙️ Analyzing Fleet Documents..."):
            result = process_query(query)

        # --- RENDER RESULTS ---
        
        # CASE 1: MANUAL DATA FOUND
        if result['type'] == 'manual':
            # 1. Badge & Container Start
            st.markdown(f"""
            <div class='result-container'>
                <span class='badge-manual'>✓ OFFICIAL SOURCE: {result['source']}</span>
                <h3 style='margin-top:15px; margin-bottom:10px;'>Technical Specifications</h3>
            """, unsafe_allow_html=True)
            
            # 2. Build HTML Table dynamically
            rows = ""
            for item in result['specs']:
                unit = item.get('unit') or ""
                rows += f"""
                <tr>
                    <td><strong>{item['component']}</strong></td>
                    <td class='val-text'>{item['value']} {unit}</td>
                    <td class='desc-text'>{item.get('description', '-')}</td>
                </tr>
                """
            
            table_html = f"""
            <table class='styled-table'>
                <thead>
                    <tr><th width='35%'>Component / Step</th><th width='25%'>Value / Action</th><th>Notes</th></tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)
            
            # 3. Images Section
            if result.get('images'):
                st.markdown("<h4 style='margin-top:25px; color:#555;'>📷 Visual Reference</h4>", unsafe_allow_html=True)
                
                # Create columns for images
                cols = st.columns(min(3, len(result['images'])))
                for idx, img_path in enumerate(result['images']):
                    with cols[idx % 3]:
                        if os.path.exists(img_path):
                            st.image(img_path, caption=f"Figure {idx+1}", use_container_width=True)
                        else:
                            st.error(f"Image missing: {img_path}")
            
            st.markdown("</div>", unsafe_allow_html=True) # Close Container

        # CASE 2: GENERAL KNOWLEDGE FALLBACK
        elif result['type'] == 'general':
            st.markdown(f"""
            <div class='result-container'>
                <span class='badge-ai'>⚠ GENERAL KNOWLEDGE</span>
                <div style='margin-top:15px; color:#444; line-height:1.6;'>
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