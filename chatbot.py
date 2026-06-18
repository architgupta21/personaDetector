import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import streamlit as st
import chromadb
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
import json

# Set premium page config
st.set_page_config(
    page_title="Persona & RAG Chatbot Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (CSS)
st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Background adjustments */
    .stApp {
        background: linear-gradient(180deg, #0e1118 0%, #121824 100%);
        color: #f0f3f8;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0e14 !important;
        border-right: 1px solid #1f2937;
    }
    
    /* Premium Profile Card */
    .profile-card {
        background: linear-gradient(135deg, rgba(31, 41, 55, 0.5) 0%, rgba(17, 24, 39, 0.7) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .profile-header {
        font-size: 22px;
        font-weight: 700;
        background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 15px;
        text-align: center;
    }
    
    .section-title {
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        color: #9ca3af;
        margin-top: 15px;
        margin-bottom: 8px;
        letter-spacing: 0.05em;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 4px;
    }
    
    /* Persona details list */
    .persona-item {
        font-size: 14px;
        color: #e5e7eb;
        margin-bottom: 6px;
        display: flex;
        align-items: flex-start;
    }
    
    .persona-item::before {
        content: "✦";
        color: #a855f7;
        margin-right: 8px;
        font-size: 12px;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 12px;
        margin: 3px;
        font-weight: 500;
    }
    
    .badge-blue {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Main Header styling */
    .main-title {
        font-size: 38px;
        font-weight: 800;
        background: linear-gradient(135deg, #c084fc 0%, #6366f1 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .main-subtitle {
        color: #9ca3af;
        font-size: 16px;
        margin-bottom: 30px;
    }
    
    /* Stat boxes */
    .stat-container {
        display: flex;
        gap: 15px;
        margin-bottom: 25px;
    }
    
    .stat-box {
        flex: 1;
        background: rgba(17, 24, 39, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }
    
    .stat-val {
        font-size: 20px;
        font-weight: 700;
        color: #f3f4f6;
    }
    
    .stat-lbl {
        font-size: 11px;
        color: #6b7280;
        text-transform: uppercase;
    }
    
    /* Chat bubbles styling */
    .stChatMessage {
        border-radius: 16px !important;
        margin-bottom: 15px !important;
        padding: 15px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Adjust width of the page container */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_systems():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "chroma_db")
    persona_path = os.path.join(current_dir, "persona.json")
    
    # Initialize DB client and load both collections
    db_client = chromadb.PersistentClient(path=db_path)
    summary_col = db_client.get_collection(name="chat_summaries")
    message_col = db_client.get_collection(name="chat_messages")
    
    # Initialize models
    llm = Ollama(model="phi3.5")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        with open(persona_path, "r") as f:
            persona = json.load(f)
    except FileNotFoundError:
        persona = {
            "habits": ["No habits loaded yet."],
            "personal_facts": ["No facts loaded yet."],
            "personality_traits": ["Unknown"],
            "communication_style": ["Conversational"]
        }
        
    return summary_col, message_col, llm, embeddings, persona

summary_col, message_col, llm, embeddings, persona = load_systems()

# Render Sidebar with User Persona & Stats
with st.sidebar:
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    st.markdown('<div class="profile-header">👤 USER PROFILE CARD</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">🎭 Personality Traits</div>', unsafe_allow_html=True)
    for trait in persona.get("personality_traits", []):
        st.markdown(f'<span class="badge">{trait}</span>', unsafe_allow_html=True)
        
    st.markdown('<div class="section-title">🏷️ Habits & Routines</div>', unsafe_allow_html=True)
    for habit in persona.get("habits", []):
        st.markdown(f'<div class="persona-item">{habit}</div>', unsafe_allow_html=True)
        
    st.markdown('<div class="section-title">📌 Personal Facts</div>', unsafe_allow_html=True)
    for fact in persona.get("personal_facts", []):
        st.markdown(f'<div class="persona-item">{fact}</div>', unsafe_allow_html=True)
        
    st.markdown('<div class="section-title">💬 Communication Style</div>', unsafe_allow_html=True)
    for style in persona.get("communication_style", []):
        st.markdown(f'<div class="persona-item">{style}</div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # System Stats
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    st.markdown('<div class="profile-header" style="font-size:16px;">⚙️ SYSTEM METRICS</div>', unsafe_allow_html=True)
    
    # Get database counts
    try:
        topics_count = summary_col.count()
        msgs_count = message_col.count()
    except Exception:
        topics_count, msgs_count = 0, 0
        
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="stat-box"><div class="stat-val">{topics_count}</div><div class="stat-lbl">Topics</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-box"><div class="stat-val">{msgs_count}</div><div class="stat-lbl">Raw Chunks</div></div>', unsafe_allow_html=True)
        
    st.markdown('<div class="section-title">Local Models</div>', unsafe_allow_html=True)
    st.markdown('<span class="badge badge-blue">LLM: Phi-3.5 (Ollama)</span>', unsafe_allow_html=True)
    st.markdown('<span class="badge badge-blue">Embeddings: MiniLM-L6 (Local)</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main Area
st.markdown('<div class="main-title">🧠 Conversation Persona Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">A local dual-collection RAG pipeline analyzing user dialogues offline.</div>', unsafe_allow_html=True)

# Chat Session Management
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ready to answer questions about the user based on the conversation history and extracted persona. Ask me anything, or try: \n- *'What kind of person is this user?'*\n- *'What are their habits?'*\n- *'How do they talk?'*"}]

# Display messages
for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask about the user's background, traits, or chat history..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Embedding query & searching collections..."):
            # 1. Embed query vector using the same local model
            query_vector = embeddings.embed_query(prompt)
            
            # 2. Query both collections (High-level summaries + Fine-grained raw message chunks)
            summary_results = summary_col.query(query_embeddings=[query_vector], n_results=3)
            message_results = message_col.query(query_embeddings=[query_vector], n_results=3)
            
            retrieved_summaries = []
            retrieved_raw_chunks = []
            
            # Extract and chronologically sort summaries
            if summary_results["documents"] and summary_results["documents"][0]:
                docs = summary_results["documents"][0]
                metas = summary_results["metadatas"][0]
                items = []
                for doc, meta in zip(docs, metas):
                    items.append({
                        "doc": doc,
                        "start_id": int(meta.get("start_id", 0)),
                        "end_id": int(meta.get("end_id", 0)),
                        "type": meta.get("type", "topic_chunk")
                    })
                items.sort(key=lambda x: x["start_id"])
                for item in items:
                    label = "Topic segment" if item["type"] == "topic_chunk" else "100-Msg Checkpoint"
                    retrieved_summaries.append(f"{label} (Messages {item['start_id']}-{item['end_id']}): {item['doc']}")
                    
            # Extract and chronologically sort raw message chunks
            if message_results["documents"] and message_results["documents"][0]:
                docs = message_results["documents"][0]
                metas = message_results["metadatas"][0]
                items = []
                for doc, meta in zip(docs, metas):
                    items.append({
                        "doc": doc,
                        "start_id": int(meta.get("start_id", 0)),
                        "end_id": int(meta.get("end_id", 0))
                    })
                items.sort(key=lambda x: x["start_id"])
                for item in items:
                    retrieved_raw_chunks.append(f"Messages {item['start_id']}-{item['end_id']}:\n{item['doc']}")
            
            summary_context = "\n\n".join(retrieved_summaries)
            raw_chunk_context = "\n\n".join(retrieved_raw_chunks)
            
            # Formulate robust, anti-hallucination prompt
            final_prompt = f"""You are an intelligent assistant answering questions about a user based ONLY on their conversation history and persona.
            
            User Persona:
            {json.dumps(persona, indent=2)}
            
            Relevant Topic Summaries:
            {summary_context}
            
            Relevant Raw Conversation Chunks:
            {raw_chunk_context}
            
            Question: {prompt}
            
            CRITICAL INSTRUCTIONS:
            1. Based strictly on the persona, summaries, and raw chunks provided above, answer the question accurately and concisely.
            2. If the answer cannot be explicitly found in the provided context, you MUST reply with exactly: "I do not have enough information to answer that based on the conversation history."
            3. Do NOT guess, infer, or make up any information.
            """
            
            response = llm.invoke(final_prompt).strip()
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})