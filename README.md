# 🚗 Mechanic AI: Multi-Modal Vehicle Specification System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predii-intelligent-ai-bot-x8szf2yk5f.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=LangChain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama3.3-orange?style=for-the-badge&logo=meta&logoColor=white)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-00d1ce?style=for-the-badge)

## 📌 Project Overview
**Mechanic AI** is a specialized Multi-Agent RAG system designed to automate the extraction of technical data from automotive service manuals.

Service manuals are complex documents containing **narrative text**, **structured specification tables**, and **critical visual diagrams** (wiring, exploded views). Standard RAG tools ignore images and mangle tables.

**Mechanic AI** solves this using a **Tri-Modal Extraction Engine**:
1.  **Text:** Recursive chunking for instructions.
2.  **Tables:** Context-aware header injection for specs.
3.  **Images:** Vision-based captioning to make diagrams searchable.

---

## 🏗️ System Architecture

The pipeline uses a **Content Router** to split the PDF into three streams, processing each media type with a specialized strategy before unifying them in the Vector Database.

```mermaid
graph TD
    PDF["📄 Uploaded Manual (PDF)"] --> Router{"Content Router"}
    
    %% Stream 1: Text
    Router -- "Narrative Text" --> Splitter["Recursive Text Splitter"]
    Splitter --> ChunkA["Text Chunks"]
    
    %% Stream 2: Tables
    Router -- "Table Grid" --> Plumber["🔧 pdfplumber Engine"]
    subgraph "Smart Header Injection"
        Plumber --> Detect["Detect Headers"]
        Detect --> Inject["Inject Header into Every Row"]
        Inject --> ChunkB["Structured Data Chunks"]
    end
    
    %% Stream 3: Images (New)
    Router -- "Visual Diagrams" --> Extractor["Image Extractor"]
    subgraph "Vision Processing"
        Extractor --> VisionModel["👁️ Vision Model (Captioning)"]
        VisionModel --> Desc["Generate Technical Description"]
        Desc --> ChunkC["Image Context Chunks"]
    end
    
    %% Unification
    ChunkA --> Embed["Embeddings (HuggingFace)"]
    ChunkB --> Embed
    ChunkC --> Embed
    
    Embed --> FAISS[("FAISS Vector Store")]
    
    Query["User Query"] --> FAISS
    FAISS --> Context["Top 5 Semantic Matches"]
    Context --> LLM["🤖 Groq Llama 3.3"]
    LLM --> Answer["Final Technical Answer"]
