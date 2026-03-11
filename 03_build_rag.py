import os
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings


KNOWLEDGE_BASE_DIR = "knowledge_base"
CHROMA_DB_DIR      = "chroma_db"
COLLECTION_NAME    = "bfsi_knowledge"
EMBED_MODEL        = "all-MiniLM-L6-v2"   
CHUNK_SIZE         = 500
TOP_K              = 3  

def load_documents(folder: str) -> list[dict]:
    docs = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for i, para in enumerate(paragraphs):
                
                if len(para) > CHUNK_SIZE:
                    words = para.split()
                    chunk = ""
                    chunk_id = 0
                    for word in words:
                        if len(chunk) + len(word) < CHUNK_SIZE:
                            chunk += word + " "
                        else:
                            docs.append({
                                "id": f"{filename}_{i}_{chunk_id}",
                                "text": chunk.strip(),
                                "source": filename
                            })
                            chunk = word + " "
                            chunk_id += 1
                    if chunk.strip():
                        docs.append({
                            "id": f"{filename}_{i}_{chunk_id}",
                            "text": chunk.strip(),
                            "source": filename
                        })
                else:
                    docs.append({
                        "id": f"{filename}_{i}",
                        "text": para,
                        "source": filename
                    })
    return docs


def build_index():
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Loading documents...")
    docs = load_documents(KNOWLEDGE_BASE_DIR)
    print(f"Loaded {len(docs)} chunks from knowledge base")

    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

   
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    print("Indexing documents...")
    texts     = [d["text"] for d in docs]
    ids       = [d["id"] for d in docs]
    metadatas = [{"source": d["source"]} for d in docs]
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        collection.add(
            documents=texts[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            ids=ids[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    print(f"✅ Indexed {len(docs)} chunks into ChromaDB")
    return collection, embedder


def query_rag(question: str, collection, embedder) -> str:
    query_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K
    )

    if not results["documents"][0]:
        return "No relevant information found in knowledge base."

    
    context = "\n\n".join(results["documents"][0])
    sources = list(set([m["source"] for m in results["metadatas"][0]]))

    return f"CONTEXT:\n{context}\n\nSOURCES: {', '.join(sources)}"


def load_index():
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"✅ Loaded existing index with {collection.count()} chunks")
    return collection, embedder


if __name__ == "__main__":
    
    if not os.path.exists(CHROMA_DB_DIR):
        collection, embedder = build_index()
    else:
        try:
            collection, embedder = load_index()
        except:
            print("Index not found, rebuilding...")
            collection, embedder = build_index()

    
    test_questions = [
        "What is the maximum LTV ratio for a home loan?",
        "What documents are needed for a personal loan?",
        "What happens if I miss an EMI payment?",
        "What is the TDS rate on fixed deposit interest?",
        "How do I report unauthorized transactions?"
    ]

    print("\n=== RAG TEST QUERIES ===\n")
    for q in test_questions:
        print(f"Q: {q}")
        result = query_rag(q, collection, embedder)
        
        print(f"A (context snippet): {result[:300]}...")
        print("-" * 60)
