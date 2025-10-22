import os
from dotenv import load_dotenv
load_dotenv()

PDF_FOLDER = "./pdfs"
FAISS_INDEX_PATH = "./faiss_index.index"
METADATA_PATH = "./faiss_metadata.pkl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

HF_API_TOKEN = os.getenv("HF_API_TOKEN") 
print(f"HF_API_TOKEN loaded: {HF_API_TOKEN}")

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
TOP_K = 3
MAX_NEW_TOKENS = 400
