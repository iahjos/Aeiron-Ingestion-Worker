import os
import time
import json
import random
import logging
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from openai import OpenAI

# -------------------------------------------------
# Logging setup
# -------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------------------------
# Environment variables
# -------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("Missing Supabase credentials (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY).")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key (OPENAI_API_KEY).")

# -------------------------------------------------
# Initialize clients
# -------------------------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
logging.info("✅ Connected to Supabase and OpenAI successfully.")

# -------------------------------------------------
# Ingestion helper functions
# -------------------------------------------------
def fetch_pending_documents():
    """Fetch pending documents from Supabase (status = 'pending')"""
    try:
        response = supabase.table("documents").select("*").eq("status", "pending").execute()
        docs = response.data or []
        logging.info(f"📄 Found {len(docs)} pending documents.")
        return docs
    except Exception as e:
        logging.error(f"❌ Error fetching documents: {e}")
        return []

def update_document_status(doc_id, status):
    """Update document status in Supabase"""
    try:
        supabase.table("documents").update({"status": status}).eq("id", doc_id).execute()
        logging.info(f"✅ Updated document {doc_id} → status = {status}")
    except Exception as e:
        logging.error(f"❌ Failed to update status for {doc_id}: {e}")

def embed_text(text):
    """Generate embedding using OpenAI"""
    try:
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
        return embedding
    except Exception as e:
        logging.error(f"❌ Embedding error: {e}")
        return None

def process_document(doc):
    """Main ingestion logic"""
    doc_id = doc["id"]
    org_id = doc["org_id"]
    path = doc["storage_path"]
    logging.info(f"🚀 Processing document {path} for org {org_id}")

    try:
        # Example Supabase file URL
        file_url = f"{SUPABASE_URL}/storage/v1/object/public/{path}"
        logging.info(f"🔗 Fetching file from {file_url}")

        # Simulated text extraction (replace later with PyMuPDF/docx/pandas)
        fake_text = f"This is a simulated ingestion for document {path}."

        # Generate embedding
        embedding = embed_text(fake_text)
        if not embedding:
            raise Exception("Embedding generation failed.")

        # Store into doc_chunks
        supabase.table("doc_chunks").insert({
            "doc_id": doc_id,
            "org_id": org_id,
            "content": fake_text,
            "embedding": json.dumps(embedding)
        }).execute()

        # Mark document as processed
        update_document_status(doc_id, "processed")
        time.sleep(1)  # brief cooldown
    except Exception as e:
        logging.error(f"❌ Error processing document {doc_id}: {e}")
        update_document_status(doc_id, "failed")

# -------------------------------------------------
# Background ingestion loop (runs in its own thread)
# -------------------------------------------------
def ingestion_loop():
    logging.info("🟢 Ingestion worker started and listening for jobs...")
    while True:
        try:
            docs = fetch_pending_documents()
            for doc in docs:
                process_document(doc)
            time.sleep(10 + random.randint(0, 5))
        except Exception as e:
            logging.error(f"Worker loop error: {e}")
            time.sleep(15)

# -------------------------------------------------
# FastAPI setup for /ask endpoint
# -------------------------------------------------
app = FastAPI(title="Aeiron Unified Service", description="Ingestion + Query API")

class AskRequest(BaseModel):
    question: str
    org_id: str
    top_k: int = 5

@app.get("/")
def home():
    return {"status": "Aeiron unified worker is running"}

@app.post("/ask")
def ask_question(req: AskRequest):
    """Answer user questions using embedded document chunks"""
    try:
        # 1️⃣ Embed the question
        q_embed = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=req.question
        ).data[0].embedding

        # 2️⃣ Query similar chunks using your SQL function match_documents()
        result = supabase.rpc("match_documents", {
            "query_embedding": q_embed,
            "match_count": req.top_k,
            "org_id_param": req.org_id
        }).execute()

        if not result.data:
            return {"answer": "No relevant content found."}

        context = "\n\n".join([r["content"] for r in result.data])

        # 3️⃣ Ask GPT
        prompt = f"Answer the question using this context:\n{context}\n\nQuestion: {req.question}"
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = completion.choices[0].message.content

        return {"answer": answer, "matches": len(result.data)}

    except Exception as e:
        logging.error(f"❌ /ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------
# Entry point: run both ingestion + FastAPI
# -------------------------------------------------
if __name__ == "__main__":
    # Start ingestion in background thread
    threading.Thread(target=ingestion_loop, daemon=True).start()

    # Start FastAPI web server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
