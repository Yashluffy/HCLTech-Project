# 🚗 Mechanic AI: Multi-Modal Vehicle Specification System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predii-intelligent-ai-bot-x8szf2yk5f.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=LangChain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama3.3-orange?style=for-the-badge&logo=meta&logoColor=white)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-00d1ce?style=for-the-badge)

## 📌 Project Overview
**Mechanic AI** is a specialized Multi-Modal RAG system designed to automate the extraction of technical data from automotive service manuals.

Service manuals are complex documents containing **narrative text**, **structured specification tables**, and **critical visual diagrams** (wiring, exploded views). Standard RAG tools ignore images and mangle tables.

**Mechanic AI** solves this using a **Tri-Modal Extraction Engine**:
1.  **Text:** Recursive chunking for instructions.
2.  **Tables:** Context-aware header injection for specs.
3.  **Images:** Vision-based captioning to make diagrams searchable.

---
Tech Stack
**Frontend** :

1. **Streamlit:** Acts as the unified interactive web interface. It handles file uploads (PDFs), manages chat sessions, and renders retrieved technical diagrams directly in the browser using pure Python.

**Orchestration & Intelligence**

**1.LangChain:** The architectural glue of the system.

**2.langchain-core:** Manages the RAG pipeline and retrieval chains.

**2.langchain-groq:** Connects the application to the Groq inference engine.

**Groq API (The Brain)**

1. **Llama 3.3 (70B):** Performs complex reasoning on extracted text and table specifications.

2. **Llama 3.2 Vision:** Analyzes extracted technical diagrams and generates searchable text captions (e.g., "Wiring diagram of fuel pump").

**Data Extraction (ETL Engine)**

1. **pdfplumber:** The core extraction engine. It provides the precise X/Y coordinates needed to detect table grids for "Smart Header Injection" and extract raw image objects.

2. **Pillow (PIL):** Handles image pre-processing (resizing and formatting) to prepare extracted diagrams for the Vision model.

**Storage & Retrieval**

1. **FAISS (CPU):** A high-performance local Vector Database. It stores embeddings for text, table rows, and image captions in a single IndexFlatL2 index for exact similarity search.

2. **HuggingFace Embeddings:** Uses sentence-transformers/all-MiniLM-L6-v2 to convert all three data types into a unified 384-dimensional vector space.


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
