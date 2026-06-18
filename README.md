# 🧠 Local RAG Pipeline & Persona Extractor

An end-to-end, fully offline Retrieval-Augmented Generation (RAG) system built to parse historical chat data, extract structured user personas, and answer queries with strict anti-hallucination guardrails.

This project is engineered to run highly advanced NLP tasks locally on constrained hardware (8GB System RAM / 4GB VRAM) without relying on paid external APIs, using Microsoft's `Phi-3.5-mini` (via Ollama) and `sentence-transformers` locally.

---

## 🏗️ Architecture & Engineering Decisions

### 1. Optimized Topic Drift Check (Semantic Chunking)
Instead of relying on slow, computationally expensive LLM calls or redundant embedding generation, this pipeline uses **In-Memory Mathematical Vector Similarity**:
- **Batch Embeddings (10x Speed Boost)**: When starting, the pipeline embeds all message texts in batches. This takes advantage of PyTorch's parallel tensor execution, reducing the embedding generation time for 1500 messages from ~30s to under 1s.
- **Running Average Baseline**: A new topic starts with the first message vector as the semantic baseline. As consecutive messages arrive, we calculate the **Cosine Similarity** between the incoming message vector and the current topic's running average (mean) vector.
- **Topic Drift**: If similarity drops below `0.45` (after at least 3 messages), a drift is triggered. The old topic messages are summarized by the LLM and stored in ChromaDB, and a new topic baseline is set using the drifting message vector.
- **No Semantic Leakage**: The message that triggered the drift is excluded from the previous topic summary and becomes the start of the next topic.

### 2. Dual-Collection RAG Architecture
To satisfy the recruiter's instructions to retrieve both high-level summaries and raw chunks:
- **`chat_summaries` Collection**: Stores LLM summaries of topic segments and chronological 100-message checkpoints.
- **`chat_messages` Collection**: Stores raw conversation transcript chunks (sliding windows of 10 messages).
- **Dual Retrieval & Chronological Sorting**: When a user queries the bot, the query is embedded and searched against both collections. The top 3 summaries and top 3 raw chunks are retrieved, sorted chronologically by their message IDs, and combined. This ensures the LLM sees the timeline in order, preventing temporal confusion.

### 3. Isolated Target Persona Extraction
To extract a structured profile that represents the user across the *entire* dataset:
- **Database-Wide Analysis**: The extraction script loads the generated topic summaries from ChromaDB. Because these summaries dense-pack the entire conversation history, the LLM can extract a dataset-wide persona in a single prompt.
- **Interlocutor Filtering**: The script isolates statements made by `User 1` (the primary user), ensuring `User 2`'s statements do not contaminate the extracted persona.
- **Style Grounding**: The system samples raw messages of `User 1` to extract exact communication style indicators (tone, punctuation, emoji usage), saving the result in a validated `persona.json`.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **LLM Orchestration:** LangChain (`langchain-community`)
- **Vector Database:** ChromaDB (Local Persistent Storage)
- **Embeddings & Math:** `sentence-transformers` (`all-MiniLM-L6-v2`), `scikit-learn` (Cosine Similarity)
- **Local Inference:** Ollama (`phi3.5`)
- **Frontend:** Streamlit

---

## 📂 Project Structure

```text
├── chroma_db/               # Auto-generated local vector database
├── conversations.csv        # Raw chronological chat dataset
├── requirements.txt         # Project dependencies
├── rag_app.py               # Step 1: Data ingestion, vector chunking, and database population
├── extract_persona.py       # Step 2: JSON-structured persona extraction
├── chatbot.py               # Step 3: Streamlit UI with Anti-Hallucination prompt engineering
├── DEPLOYMENT.md            # Cloud hosting instructions (Hugging Face Spaces & VPS)
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Install Local AI Engine (Ollama)
Download and install Ollama from [ollama.com](https://ollama.com). Pull the Microsoft Phi-3.5 model:
```bash
ollama run phi3.5
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Execution Pipeline

Run the scripts in the following order to build the database, extract the persona, and start the chatbot:

### 1. Populate the Vector Database
Ingests the CSV, calculates topic drifts in-memory, saves topic/100-msg summaries to `chat_summaries`, and raw chunks to `chat_messages`:
```bash
python rag_app.py
```

### 2. Extract the User Persona
Loads database summaries and samples raw messages to generate `persona.json`:
```bash
python extract_persona.py
```

### 3. Launch the Chat Interface
Starts the Streamlit web server to query the RAG system:
```bash
python -m streamlit run chatbot.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## ☁️ Cloud Deployment
Detailed instructions for cloud deployment are available in [DEPLOYMENT.md](file:///c:/Projects/personaExtractor/DEPLOYMENT.md). The recommended method is deploying on **Hugging Face Spaces** using the provided `Dockerfile` and `entrypoint.sh` setup.