# 🏎️ Fleet Command: Multi-Modal Vehicle Intelligence System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=LangChain&logoColor=white)
![Llama 3](https://img.shields.io/badge/Llama_3.3-040404?style=for-the-badge&logo=meta&logoColor=white)

> **A Unified RAG Pipeline that ingests Unstructured (PDF), Structured (CSV), and Hierarchical (JSON) data into a single queryable Vector Database.**

---

## 📖 Project Overview

**Fleet Command** is a Retrieval-Augmented Generation (RAG) system designed to solve the "Siloed Data" problem in automotive engineering.

Standard RAG systems treat all documents as plain text, which leads to failure when processing **Technical Tables** or **Structured Data**. Fleet Command solves this by implementing a **Multi-Modal Ingestion Pipeline**: it applies specialized chunking algorithms to three distinct file formats—**PDFs (Cars), CSVs (Bikes), and JSONs (Jets)**—before unifying them into a single FAISS vector index. This allows users to ask complex questions across different vehicle domains from one simple interface.

---

## 🏗️ System Architecture

The pipeline follows a **"Chunk-Embed-Retrieve"** strategy with custom pre-processing for each data type.

```mermaid
graph TD
    User[User Query] --> Retrieve[🔍 Similarity Search (Top 10)]
    Retrieve --> Context[📄 Context Window]
    Context --> LLM[🤖 Llama 3.3 (Groq)]
    LLM --> Answer[Final Answer]

    subgraph "Data Ingestion Pipeline"
        PDF[🚗 PDF Manual] -->|Header Injection| Chunk1[Rich Text Chunks]
        CSV[🏍️ Bike CSV] -->|Row-to-Sentence| Chunk2[Rich Text Chunks]
        JSON[✈️ Jet JSON] -->|Hierarchy Flattening| Chunk3[Rich Text Chunks]
        
        Chunk1 --> Embed[Embeddings (HuggingFace)]
        Chunk2 --> Embed
        Chunk3 --> Embed
        Embed --> DB[(FAISS Vector Store)]
    end
    
    DB --> Retrieve
