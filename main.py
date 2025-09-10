import os
import hashlib
from fastapi import FastAPI, Request
import uvicorn
from supabase import create_client, Client
from openai import OpenAI
import fitz  # PyMuPDF for PDF parsing
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Service role key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not OPENAI_KEY:
    raise RuntimeError("Missing SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, or OPENAI_API_KEY in .env")

# --- Create clients ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai = OpenAI(api_key=OPENAI_KEY)

# --- FastAPI app ---
app = FastAPI()

# --- Helpers ---
def compute_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using PyMuPDF"""
    text = ""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    """Split text into smaller chunks"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# --- Routes ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(request: Request):
    data = await request.json()
    print("Received event:", data)

    doc_id = data["doc_id"]
    org_id = data["org_id"]
    file_path = data["file_path"]

    # 1. Download file from Supabase Storage
    try:
        file_resp = supabase.storage.from_("originals").download(file_path)
        file_bytes = file_resp
    except Exception as e:
        return {"error": f"Failed to download file: {str(e)}"}

    # 2. Compute hash and check for duplicates
    file_hash = compute_hash(file_bytes)
    existing = supabase.table("documents").select("id").eq("org_id", org_id).eq("content_hash", file_hash).execute()

    if existing.data:
        print("Duplicate found, skipping ingestion")
        return {"status": "duplicate_skipped"}

    # 3. Extract text (currently PDF only)
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    else:
        return {"error": "Unsupported file type (yet)"}

    if not text.strip():
        return {"error": "No text extracted from file"}

    # 4. Chunk text
    chunks = chunk_text(text, chunk_size=1000)

    # 5. Embed and insert into doc_chunks
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        embedding = openai.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding

        supabase.table("doc_chunks").insert({
            "doc_id": doc_id,
            "org_id": org_id,
            "chunk_index": idx,
            "content": chunk,
            "embedding": embedding
        }).execute()

    # 6. Update documents row with hash
    supabase.table("documents").update({"content_hash": file_hash}).eq("id", doc_id).execute()

    return {"status": "ingestion_complete", "chunks": len(chunks)}

# --- Run locally ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
