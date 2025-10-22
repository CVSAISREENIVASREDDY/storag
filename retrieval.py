# retrieval.py
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer 
import pickle
import faiss 
import os
import glob
import numpy as np 
from typing import List, Dict 


# -----------------------------
# 1️⃣ Load and chunk PDFs
# -----------------------------
def load_and_chunk_pdfs(folder_path: str, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    docs = []
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    for file in pdf_files:
        reader = PdfReader(file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue
            # Chunk text
            for j in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[j:j + chunk_size]
                docs.append({
                    "source": os.path.basename(file),
                    "page": i + 1,
                    "chunk_id": len(docs),
                    "text": chunk
                })
    return docs


# -----------------------------
# 2️⃣ Build or load FAISS index
# -----------------------------
def build_faiss_index(docs: List[Dict], model_name: str, faiss_path: str, metadata_path: str):
    if os.path.exists(faiss_path) and os.path.exists(metadata_path):
        print("🔹 Loading existing FAISS index...")
        index = faiss.read_index(faiss_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata

    print("🔹 Building new FAISS index...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode([d["text"] for d in docs], show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))

    faiss.write_index(index, faiss_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(docs, f)

    return index, docs


# -----------------------------
# 3️⃣ Retrieve top-K similar documents
# -----------------------------
def retrieve_similar_documents(query: str, embedding_model_name: str, faiss_index_path: str, metadata_path: str, top_k: int) -> List[Dict]:
    # Load embedding model
    model = SentenceTransformer(embedding_model_name)

    # Convert query to embedding
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Load FAISS index
    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}. Build it first.")

    index = faiss.read_index(faiss_index_path)

    # Search for top-K nearest chunks
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)

    # Load metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Build it first.")

    with open(metadata_path, "rb") as f:
        documents = pickle.load(f)

    # Return corresponding chunks
    threshold = 0.5  # example threshold for L2 distance

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist <= threshold:
            results.append(documents[idx])
            
    return results
