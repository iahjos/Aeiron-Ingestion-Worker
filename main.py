# --- Background Worker --- #

import os
import time
import json
import logging
import random
import fitz  # PyMuPDF for PDF parsing
from supabase import create_client, Client
from openai import OpenAI

# ----------------------------------------
# Logging setup
# ----------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------------------------
# Load environment variables
# ----------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key. Please set OPENAI_API_KEY.")

# Initialize clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

logging.info("✅ Connected to Supabase project and OpenAI API successfully.")

# ----------------------------------------
# Helper functions
# ----------------------------------------
def fetch_pending_documents():
    """Fetch documents from Supabase where status = 'pending'"""
    try:
        response = supabase.table("documents").select("*").eq("status", "pending").execute()
        docs = response.data or []
        logging.info(f"📄 Found {len(docs)} pending documents.")
        return docs
    except Exception as e:
        logging.error(f"❌ Error fetching pending documents: {e}")
        return []


def update_document_status(doc_id: str, status: str):
    """Update the processing status of a document"""
    try:
        supabase.table("documents").update({"status": status}).eq("id", doc_id).execute()
        logging.info(f"✅ Updated document {doc_id} → {status}")
    except Exception as e:
        logging.error(f"❌ Failed to update document {doc_id} status: {e}")


def embed_text(text: str):
    """Generate text embeddings via OpenAI API"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"❌ Embedding generation failed: {e}")
        return None


def extract_text_from_pdf(pdf_bytes: bytes):
    """Extract readable text from PDF bytes"""
    try:
        text = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
        return text.strip()
    except Exception as e:
        logging.error(f"❌ PDF extraction error: {e}")
        return ""


def process_document(doc):
    """Main ingestion pipeline for a single document"""
    doc_id = doc.get("id")
    org_id = doc.get("org_id")
    path = doc.get("storage_path")

    logging.info(f"🚀 Starting ingestion for {path} (org {org_id})")

    try:
        # Step 1: Download file bytes directly from Supabase Storage
        bucket_name, file_name = path.split("/", 1)
        data = supabase.storage.from_(bucket_name).download(file_name)
        pdf_bytes = data

        # Step 2: Extract text
        text_content = extract_text_from_pdf(pdf_bytes)
        if not text_content:
            raise Exception("No text extracted from PDF.")

        # Step 3: Generate embedding
        embedding = embed_text(text_content)
        if not embedding:
            raise Exception("Embedding creation failed.")

        # Step 4: Insert document chunk
        supabase.table("doc_chunks").insert({
            "doc_id": doc_id,
            "org_id": org_id,
            "content": text_content[:15000],  # limit for small docs
            "embedding": json.dumps(embedding)
        }).execute()

        # Step 5: Mark as processed
        update_document_status(doc_id, "processed")
        logging.info(f"✅ Successfully ingested {path}")

        # cooldown for API safety
        time.sleep(1)

    except Exception as e:
        logging.error(f"❌ Error processing document {path}: {e}")
        update_document_status(doc_id, "failed")


# ----------------------------------------
# Main worker loop
# ----------------------------------------
def main():
    logging.info("🟢 Background worker is now listening for new documents...")
    while True:
        try:
            docs = fetch_pending_documents()
            if not docs:
                logging.info("⏳ No pending documents. Checking again soon...")
            for doc in docs:
                process_document(doc)

            # Sleep between polling intervals (10–15s)
            sleep_time = 10 + random.randint(0, 5)
            time.sleep(sleep_time)

        except Exception as e:
            logging.error(f"🔥 Worker loop encountered an error: {e}")
            time.sleep(15)


if __name__ == "__main__":
    main()
