import streamlit as st
import json
import os
import re
from PIL import Image

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="Mechanic AI Pro – HCLTech",
    page_icon="🛠️",
    layout="wide"
)

# =====================================================
# SECRETS
# =====================================================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
    api_key=GROQ_API_KEY
)

# =====================================================
# PATHS (MATCH YOUR REPO)
# =====================================================
FAISS_PATH = "faiss_db_index_test"
IMAGE_DIR = "extracted_images"
VEHICLE_JSON = "vehicle_specs.json"

PDF_FILES = [
    "HAF-F16.pdf",
    "motorcycles.pdf",
    "sample-service-manual.pdf"
]

# =====================================================
# LOAD VEHICLE SPECS
# =====================================================
@st.cache_data
def load_vehicle_specs():
    if os.path.exists(VEHICLE_JSON):
        with open(VEHICLE_JSON, "r") as f:
            return json.load(f)
    return {}

vehicle_specs = load_vehicle_specs()

# =====================================================
# LOAD FAISS VECTOR DB
# =====================================================
@st.cache_resource
def load_faiss():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

vector_db = load_faiss()

# =====================================================
# SESSION STATE
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================================
# UI HEADER
# =====================================================
st.markdown("""
<h1 style="text-align:center;">🛠️ Mechanic AI Pro</h1>
<p style="text-align:center; color:gray;">
HCLTech – Fleet, Aircraft & Vehicle Intelligence
</p>
<hr>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("📂 Knowledge Sources")
st.sidebar.write("• F-16 Aircraft Manual")
st.sidebar.write("• Motorcycle Service Guide")
st.sidebar.write("• Fleet Service Manual")
st.sidebar.write("• Vehicle Specs JSON")
st.sidebar.write("• FAISS Vector DB")

# =====================================================
# CORE AI FUNCTION
# =====================================================
def mechanic_ai(query: str):
    query_lower = query.lower()

    # -----------------------------
    # VEHICLE SPECS JSON SEARCH
    # -----------------------------
    for key, value in vehicle_specs.items():
        if key.lower() in query_lower:
            return f"### 📊 Vehicle Specs – {key}\n```json\n{json.dumps(value, indent=2)}\n```"

    # -----------------------------
    # VECTOR SEARCH (PDFs)
    # -----------------------------
    docs = vector_db.similarity_search(query, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    # -----------------------------
    # LLM PROMPT
    # -----------------------------
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert mechanical engineer and maintenance AI. "
         "Answer only from the provided manuals and documents. "
         "Be precise and technical."),
        ("human",
         "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    response = llm.invoke(
        prompt.format(context=context, question=query)
    )

    return response.content

# =====================================================
# CHAT DISPLAY
# =====================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =====================================================
# IMAGE HELPER
# =====================================================
def show_related_images(query):
    if not os.path.exists(IMAGE_DIR):
        return

    for img in os.listdir(IMAGE_DIR):
        if any(word in img.lower() for word in query.lower().split()):
            try:
                st.image(
                    Image.open(os.path.join(IMAGE_DIR, img)),
                    use_container_width=True
                )
            except:
                pass

# =====================================================
# USER INPUT
# =====================================================
user_input = st.chat_input("Ask about vehicles, aircraft, maintenance, specs...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("assistant"):
        with st.spinner("Analyzing manuals & fleet data..."):
            try:
                answer = mechanic_ai(user_input)
                st.markdown(answer)
                show_related_images(user_input)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                st.error(f"❌ Error: {e}")
