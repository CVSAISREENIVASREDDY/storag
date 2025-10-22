from retrieval import load_and_chunk_pdfs, build_faiss_index, retrieve_similar_documents
from augmented import build_prompt
from generation import generate_answer
import config
import os 

HF_API_TOKEN = os.getenv("HF_API_TOKEN") 

def main():
    # 1️⃣ Load and chunk PDFs
    docs = load_and_chunk_pdfs(config.PDF_FOLDER, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    # 2️⃣ Build or load FAISS index
    index, metadata = build_faiss_index(docs, config.EMBEDDING_MODEL, config.FAISS_INDEX_PATH, config.METADATA_PATH)

    # 3️⃣ Get user query
    query = input("\nEnter your question about the stories: ")

    # 4️⃣ Retrieve context
    retrieved = retrieve_similar_documents(query, config.EMBEDDING_MODEL, config.FAISS_INDEX_PATH, config.METADATA_PATH, config.TOP_K)

    # 5️⃣ Build augmented prompt
    prompt = build_prompt(query, retrieved)

    # 6️⃣ Generate answer
    answer = generate_answer(prompt, config.MISTRAL_MODEL, config.HF_API_TOKEN, config.MAX_NEW_TOKENS)

    print("\n--- Final Answer ---\n")
    print(answer) 


if __name__ == "__main__":
    main() 
