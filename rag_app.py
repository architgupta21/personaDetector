import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb


def load_and_flatten_data(csv_filepath):
    print(f"\n[1] Loading data from {csv_filepath}...")
    df = pd.read_csv(csv_filepath, header=None)
    messages = []
    global_index = 1
    
    for index, row in df.iterrows():
        full_conversation = str(row[0])
        lines = full_conversation.split('\n')
        for line in lines:
            line = line.strip()
            if not line: continue
            if ":" in line:
                parts = line.split(":", 1)
                messages.append({
                    "id": global_index,
                    "sender": parts[0].strip(),
                    "text": parts[1].strip()
                })
                global_index += 1
    return messages

class RAGPipeline:
    def __init__(self):
        print("\n[2] Waking up AI Models & Database...")
        self.llm = Ollama(model="phi3.5")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, "chroma_db")
        self.db_client = chromadb.PersistentClient(path=db_path)
        
        # Reset collections if you are rerunning to ensure clean metadata
        for col_name in ["chat_summaries", "chat_messages"]:
            try:
                self.db_client.delete_collection(col_name)
            except Exception:
                pass
        
        self.summary_collection = self.db_client.get_or_create_collection(name="chat_summaries")
        self.message_collection = self.db_client.get_or_create_collection(name="chat_messages")

    def embed_messages_in_batches(self, messages, batch_size=1000):
        print(f"Generating embeddings for {len(messages)} messages in batches of {batch_size}...")
        all_vectors = []
        texts = [m['text'] for m in messages]
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_vectors = self.embeddings.embed_documents(batch_texts)
            all_vectors.extend(batch_vectors)
        for msg, vec in zip(messages, all_vectors):
            msg['vector'] = vec

    def summarize_and_store(self, message_chunk, chunk_type, start_id, end_id, topic_id=None):
        script = "\n".join([f"{m['sender']}: {m['text']}" for m in message_chunk])
        prompt = f"Summarize this conversation briefly:\n\n{script}\n\nSummary:"
        
        summary_text = self.llm.invoke(prompt).strip()
        
        # Format the output exactly as requested by HR
        if chunk_type == "topic_chunk":
            print(f"Topic {topic_id} -> messages {start_id}-{end_id} -> {summary_text}")
        else:
            print(f"100-Msg Checkpoint -> messages {start_id}-{end_id} -> {summary_text}")
        
        # Explicitly generate summary embedding for ChromaDB
        summary_emb = self.embeddings.embed_query(summary_text)
        
        # Save BOTH summary and raw_messages to satisfy Part 3 constraints
        self.summary_collection.add(
            documents=[summary_text],
            embeddings=[summary_emb],
            metadatas=[{
                "type": chunk_type, 
                "start_id": int(start_id), 
                "end_id": int(end_id),
                "raw_messages": script # Storing the raw chunks!
            }],
            ids=[f"{chunk_type}_{start_id}_{end_id}"]
        )

    def store_message_chunk(self, message_chunk):
        script = "\n".join([f"{m['sender']}: {m['text']}" for m in message_chunk])
        start_id = message_chunk[0]['id']
        end_id = message_chunk[-1]['id']
        
        chunk_emb = self.embeddings.embed_query(script)
        
        self.message_collection.add(
            documents=[script],
            embeddings=[chunk_emb],
            metadatas=[{
                "start_id": int(start_id),
                "end_id": int(end_id)
            }],
            ids=[f"msg_chunk_{start_id}_{end_id}"]
        )

    def process_messages(self, messages):
        # 1. Precompute embeddings in batches for 10x speedup
        self.embed_messages_in_batches(messages)
        
        print(f"\n[3] Processing {len(messages)} messages for topics and summaries...\n")
        
        current_topic_messages = []
        topic_counter = 1
        baseline_vector = None
        
        for i, msg in enumerate(messages):
            # A. 100-Message Checkpoint
            if i > 0 and i % 100 == 0:
                chunk = messages[i-100:i]
                self.summarize_and_store(chunk, "100_msg_chunk", chunk[0]['id'], chunk[-1]['id'])

            # B. 10-Message Chunk (Store raw messages in ChromaDB)
            if i > 0 and i % 10 == 0:
                chunk = messages[i-10:i]
                self.store_message_chunk(chunk)

            # C. Topic Drift Check
            new_msg_vector = msg['vector']
            
            if not current_topic_messages:
                current_topic_messages.append(msg)
                baseline_vector = np.array(new_msg_vector)
            else:
                # Calculate Cosine Similarity in-memory using cached embeddings
                similarity = cosine_similarity([baseline_vector], [new_msg_vector])[0][0]
                
                # Check topic drift (only after we have at least 3 messages in the current topic)
                if len(current_topic_messages) >= 3 and similarity < 0.45:
                    # Drift detected: Summarize current topic (excluding the new message)
                    self.summarize_and_store(
                        current_topic_messages, 
                        "topic_chunk", 
                        current_topic_messages[0]['id'], 
                        current_topic_messages[-1]['id'], 
                        topic_counter
                    )
                    topic_counter += 1
                    
                    # Reset topic with the current message
                    current_topic_messages = [msg]
                    baseline_vector = np.array(new_msg_vector)
                else:
                    current_topic_messages.append(msg)
                    # Update baseline vector as the running average of vectors in the topic
                    topic_vectors = [m['vector'] for m in current_topic_messages]
                    baseline_vector = np.mean(topic_vectors, axis=0)

        # Process the final leftover topic chunk
        if current_topic_messages:
            self.summarize_and_store(
                current_topic_messages, 
                "topic_chunk", 
                current_topic_messages[0]['id'], 
                current_topic_messages[-1]['id'], 
                topic_counter
            )

        # Process the final leftover 100-message checkpoint
        leftover_100_start = ((len(messages) - 1) // 100) * 100
        if leftover_100_start < len(messages):
            chunk = messages[leftover_100_start:]
            self.summarize_and_store(chunk, "100_msg_chunk", chunk[0]['id'], chunk[-1]['id'])

        # Process the final leftover 10-message chunk
        leftover_10_start = ((len(messages) - 1) // 10) * 10
        if leftover_10_start < len(messages):
            chunk = messages[leftover_10_start:]
            self.store_message_chunk(chunk)

if __name__ == "__main__":
    all_messages = load_and_flatten_data('conversations.csv')
    pipeline = RAGPipeline()
    test_slice = all_messages[:1500] 
    pipeline.process_messages(test_slice)
    print("\n[SUCCESS] Processing Complete!")