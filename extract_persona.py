import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import json
import chromadb
from langchain_community.llms import Ollama
import re

def extract_profile():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "chroma_db")
    
    summaries_text = ""
    # 1. Attempt to load summaries from ChromaDB
    try:
        if os.path.exists(db_path):
            db_client = chromadb.PersistentClient(path=db_path)
            collection = db_client.get_collection(name="chat_summaries")
            results = collection.get(where={"type": "topic_chunk"})
            if results and results["documents"]:
                print(f"Loaded {len(results['documents'])} topic summaries from ChromaDB.")
                summaries_text = "\n".join([f"- {doc}" for doc in results["documents"]])
    except Exception as e:
        print(f"Notice: Could not load summaries from database ({e}). Will fall back to CSV sampling.")
        
    # 2. Sample raw messages for User 1 from CSV to capture communication style
    print("Loading raw conversations to analyze communication style...")
    df = pd.read_csv('conversations.csv', header=None)
    
    user1_messages = []
    # Sample from the first 50 rows to get diverse contexts
    for idx, row in df.iloc[:50].iterrows():
        lines = str(row[0]).split('\n')
        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)
                sender = parts[0].strip()
                text = parts[1].strip()
                if sender == "User 1":
                    user1_messages.append(text)
                    if len(user1_messages) >= 100:
                        break
        if len(user1_messages) >= 100:
            break
            
    raw_sample = "\n".join([f"User 1: {msg}" for msg in user1_messages[:100]])
    
    # If we couldn't get summaries from ChromaDB, let's extract some dialogue lines from CSV
    if not summaries_text:
        print("Using CSV sampling for full context...")
        all_lines = []
        for idx, row in df.iloc[:15].iterrows():
            lines = str(row[0]).split('\n')
            all_lines.extend([line.strip() for line in lines if ':' in line])
        summaries_text = "Dialogue Sample:\n" + "\n".join(all_lines[:200])

    print("Asking Phi-3.5 to analyze personality and habits for User 1...")
    llm = Ollama(model="phi3.5")
    
    prompt = f"""You are a specialized behavioral analysis AI.
Your task is to analyze the following information about "User 1" (who is the target user of this analysis) and build a highly accurate, structured JSON persona profile.

Source Data 1: High-level summaries of User 1's conversations across different days:
{summaries_text}

Source Data 2: Raw message samples written by User 1 (use this to capture their tone, style, message length, and emoji usage):
{raw_sample}

CRITICAL INSTRUCTIONS:
1. Extract persona attributes ONLY for "User 1". Do NOT include any information, habits, or facts belonging to User 2.
2. Under "habits", list concrete, repeating behaviors (e.g. food preferences, sleeping patterns, hobbies).
3. Under "personal_facts", list static details (e.g. family, pets, relationships, career, location, plans).
4. Under "personality_traits", list psychological qualities (e.g. helpful, emotional, enthusiastic, funny, serious).
5. Under "communication_style", list styling/syntax habits (e.g. uses emojis frequently, writes short sentences, polite, informal, uses punctuation).
6. Ground every attribute in the actual conversation signals. Do not guess or hallucinate.
7. Output STRICTLY as a valid JSON object. Do not wrap it in markdown code blocks or add any introductory/concluding text.

Required JSON structure:
{{
    "habits": ["habit 1", "habit 2", ...],
    "personal_facts": ["fact 1", "fact 2", ...],
    "personality_traits": ["trait 1", "trait 2", ...],
    "communication_style": ["style 1", "style 2", ...]
}}
"""
    
    response = llm.invoke(prompt).strip()
    
    # Robustly parse JSON (in case LLM wraps it in markdown ```json ... ```)
    try:
        # Try to find a JSON block in the response
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            json_str = response
            
        persona_data = json.loads(json_str)
        
        # Save to file
        with open("persona.json", "w") as f:
            json.dump(persona_data, f, indent=4)
        print("\n[SUCCESS] Persona extracted and saved to 'persona.json'.")
        print(json.dumps(persona_data, indent=2))
        
    except Exception as e:
        print("\n[WARNING] Failed to parse JSON. Raw output from AI was:")
        print(response)
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_profile()