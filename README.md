# 🚗 Intelligent Vehicle Specification Extraction System (Mechanic AI)

## Live Link
[![Streamlit App]([https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predii-intelligent-ai-bot-x8szf2yk5f.streamlit.app/](https://hcltech-project-m97vsgrm5ovy6ryasakqxb.streamlit.app/))
---

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=LangChain)
![Groq](https://img.shields.io/badge/Groq-Llama3-orange?style=for-the-badge)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-00d1ce?style=for-the-badge)

## 📌 Project Overview
**Mechanic AI** is a specialized Retrieval-Augmented Generation (RAG) system developed to automate the extraction of technical specifications from automotive service manuals. 

Service manuals are often hundreds of pages long with complex table structures, making it difficult for standard AI tools to extract precise values (like torque settings or fluid capacities). This tool uses **Context-Aware Table Processing** to "read" manuals like a mechanic, ensuring users receive accurate, structured JSON data for every query.

---
![Dashboard](assets/dashboard.png)


## 🏗️ System Pipeline

The system follows a high-precision RAG pipeline designed for tabular data:

1.  **Ingestion & Pre-processing:** * The PDF manual is scanned page-by-page.
    * **Smart Header Injection** detects tables, identifies headers (e.g., "Nm | lb-ft"), and injects them into every data row to prevent context loss.
2.  **Embedding:** * Processed chunks are converted into 384-dimensional vectors using **HuggingFace** (`sentence-transformers/all-MiniLM-L6-v2`).
3.  **Storage (The Brain):** * Vectors are stored in a local **FAISS** index for millisecond-latency retrieval.
4.  **Retrieval:** * User queries are embedded and compared against the database to find the top **10 relevant chunks** (High `k` value ensures coverage of long tables).
5.  **Generation:** * The retrieved context is sent to **Groq Llama 3.3 (70B)**, which extracts the specific values into a strict JSON schema.

---

## 🛠️ Tools & Tech Stack

| Component | Tool | Purpose |
| :--- | :--- | :--- |
| **Frontend** | **Streamlit** | Interactive web interface for querying the system. |
| **LLM Engine** | **Groq API** | Ultra-fast inference using Llama 3.3 (70B) for parsing data. |
| **Orchestration** | **LangChain** | Managing the retrieval chains and prompt logic. |
| **PDF Processing** | **pdfplumber** | extracting complex table structures and grid data. |
| **Vector DB** | **FAISS** | Similarity search and storage. |
| **Embeddings** | **HuggingFace** | `all-MiniLM-L6-v2` model for semantic vectorization. |

---

## 🧠 Technical Deep Dive

### 🧩 1. Hybrid Chunking Strategy
Mechanic AI employs a **Hybrid Splitting Architecture** to handle the diverse layouts found in automotive service manuals.

* **A. Context-Aware Table Processing (The "Smart" Splitter)**
    * **Target:** Complex specification tables (Torque specs, Fluid capacities).
    * **Technique:** Dynamic Header Injection via `pdfplumber`.
    * **Logic:** The system heuristically scans the first 5 rows of a table to identify true headers (e.g., "Nm", "lb-ft"). It then **injects this header into every single data row**.
    * *Result:* A raw row `| Bolt | 17 |` transforms into `Component: Bolt, Torque (Nm): 17`.

* **B. Recursive Text Splitting (The "Standard" Splitter)**
    * **Target:** Narrative text, instructions, and troubleshooting guides.
    * **Technique:** `RecursiveCharacterTextSplitter`.
    * **Config:** Chunk Size: 2000 chars | Overlap: 300 chars.
    * *Why:* Prioritizes keeping paragraphs and procedures together in one context window.

### 🗄️ 2. Vector Database Strategy (FAISS)
The system utilizes **Facebook AI Similarity Search (FAISS)** for dense vector retrieval.

* **Indexing Method:** **`IndexFlatL2`** (Euclidean Distance).
    * **Why FlatL2?** For a dataset of this size (single manual, <5,000 chunks), an exact search (`Flat`) offers **100% retrieval accuracy** (Ground Truth).
    * **Why not IVF?** Inverted File Index (IVF) is faster for massive datasets (millions of vectors) but introduces a small margin of error by approximating clusters. Since engineering specifications require zero error, we prioritized accuracy over the negligible speed gain of IVF.

---

## 🚀 Future Roadmap & Improvements

To enhance the robustness and accuracy of Mechanic AI, the following architectural improvements are planned:

| Feature | Impact | Status |
| :--- | :--- | :--- |
| **JSON-Based Table Chunking** | ⭐⭐⭐⭐⭐ (High) | ![Planned](https://img.shields.io/badge/Status-Planned-blue?style=flat-square) |
| **Hybrid Search (BM25)** | ⭐⭐⭐⭐ (Medium) | ![Research](https://img.shields.io/badge/Status-Researching-yellow?style=flat-square) |
| **Vision (Multi-Modal)** | ⭐⭐⭐ (Medium) | ![Backlog](https://img.shields.io/badge/Status-Backlog-lightgrey?style=flat-square) |

### 📄 1. JSON-Based Table Chunking
* **Proposed Solution:** Instead of embedding text strings, convert every table row into a **JSON Object** before embedding.
* **Benefit:** guarantees that the LLM receives a structured key-value pair for every single data point, eliminating parsing errors for complex multi-column grids.

### 🔍 2. Hybrid Search Architecture
* **Proposed Solution:** Combine **BM25 (Keyword Search)** with **FAISS (Semantic Search)** using Reciprocal Rank Fusion (RRF).
* **Benefit:** Semantic search is great for concepts ("how to fix brakes"), but BM25 is superior for exact identifiers ("Part #99-X-200"). A hybrid approach gives the best of both worlds.

### 👁️ 3. Multi-Modal Vision Capabilities
* **Proposed Solution:** Integrate **Llama 3.2 Vision**.
* **User Story:** A mechanic uploads a photo of a rusted component. The AI identifies the part visually and automatically retrieves its torque spec without the user typing a word.

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the Repository
git clone [https://github.com/deepakkishore4578/Intelligent-Vehicle-Specification-Extraction-System.git](https://github.com/deepakkishore4578/Intelligent-Vehicle-Specification-Extraction-System.git)
cd Intelligent-Vehicle-Specification-Extraction-System

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Run the App
streamlit run app.py
