# 🚗 Mechanic AI: Intelligent Vehicle Specification Extraction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predii-intelligent-ai-bot-x8szf2yk5f.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=LangChain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama3.3-orange?style=for-the-badge&logo=meta&logoColor=white)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-00d1ce?style=for-the-badge)

## 📌 Project Overview
**Mechanic AI** is a specialized Retrieval-Augmented Generation (RAG) system developed to automate the extraction of technical specifications from automotive service manuals.

Service manuals are often hundreds of pages long with complex table structures. Standard RAG tools treat these PDFs as plain text, often failing to extract precise values like **torque settings**, **fluid capacities**, or **part dimensions** because they lose the alignment between columns and rows.

**Mechanic AI** solves this using a **Context-Aware Table Processing** engine that "reads" manuals like a human mechanic, identifying grid structures and locking headers to values before embedding.

---

## 🏗️ System Architecture

The pipeline uses a **Hybrid Chunking Strategy** to handle the mixed-media nature of PDF manuals (Narrative Text + Structured Tables).

```mermaid
graph TD
    PDF["📄 Uploaded Manual (PDF)"] --> Router{"Content Router"}
    
    Router -- "Narrative Text" --> Splitter["Recursive Text Splitter"]
    Router -- "Table Grid" --> Plumber["🔧 pdfplumber Engine"]
    
    Splitter --> ChunkA["Text Chunks"]
    
    subgraph "Smart Header Injection"
        Plumber --> Detect["Detect Headers"]
        Detect --> Inject["Inject Header into Every Row"]
        Inject --> ChunkB["Structured Data Chunks"]
    end
    
    ChunkA --> Embed["Embeddings (HuggingFace)"]
    ChunkB --> Embed
    
    Embed --> FAISS[("FAISS Vector Store")]
    
    Query["User Query"] --> FAISS
    FAISS --> Context["Top 5 Semantic Matches"]
    Context --> LLM["🤖 Groq Llama 3.3"]
    LLM --> Answer["Final Technical Answer"]
