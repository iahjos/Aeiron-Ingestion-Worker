import os
import time
import json
import logging
import random
import requests
from supabase import create_client, Client
from openai import OpenAI
import fitz  # PyMuPDF for real PDF text extraction

# ----------------------------------------
# Logging setup
# ----------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------------------------
# Environment setup
# ----------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("❌ Missing Supabase credentials (URL or Service Role key).")

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OpenAI API key.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

logging.info("✅ Connected to Supabase and OpenAI successfully.")

# ----------------------------------------
# Helpers
# ----------------------------------------
def fetch_pending_documents():
    try:
        response = supabase.table("documents").select("*").eq("status", "pending").execute()
        docs = response.data or []
        logging.info(f"📄 Found {len(docs)} pending documents.")
        return docs
    except Exception as e:
        logging.error(f"❌ Error fetching documents: {e}")
        return []

def update_document_status(doc_id, status):
    try:
        supabase.table("documents").update({"status": status}).eq("id", doc_id).execute()
        logging.info(f"✅ Updated document {doc_id} → {status}")
    except Exception as e:
        logging.error(f"❌ Failed to update status for {doc_id}: {e}")

def embed_text(text):
    try:
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
        return embedding
    except Exception as e:
        logging.error(f"❌ Embedding error: {e}")
        return None

def generate_signed_url(path, expires_in=60):
    """Generate a short-lived signed URL for a private Supabase file."""
    try:
        result = supabase.storage.from_("company_docs").create_signed_url(path.replace("company_docs/", ""), expires_in)
        signed_url = result.get("signedURL") or result.get("signed_url")
        if not signed_url:
            raise Exception("No signed URL returned from Supabase.")
        full_url = f"{SUPABASE_URL}{signed_url}"
        return full_url
    except Exception as e:
        logging.error(f"❌ Error generating signed URL: {e}")
        return None

def extract_text_from_pdf(url):
    """Download and extract text from a PDF using PyMuPDF."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open("/tmp/temp.pdf", "wb") as f:
            f.write(response.content)
        doc = fitz.open("/tmp/temp.pdf")
        text = ""
        for page in doc:
            text += page.get_text("text")
        doc.close()
        if not text.strip():
            raise Exception("No text extracted from PDF.")
        return text.strip()
    except Exception as e:
        logging.error(f"❌ PDF extraction failed for {url}: {e}")
        return None

def process_document(doc):
    doc_id = doc["id"]
    org_id = doc["org_id"]
    path = doc["storage_path"]
    logging.info(f"🚀 Processing document {path} for org {org_id}")

    try:
        signed_url = generate_signed_url(path)
        if not signed_url:
            raise Exception("Could not generate signed URL.")

        text = extract_text_from_pdf(signed_url)
        if not text:
            raise Exception("No text extracted.")

        embedding = embed_text(text)
        if not embedding:
            raise Exception("Embedding generation failed.")

        supabase.table("doc_chunks").insert({
            "doc_id": doc_id,
            "org_id": org_id,
            "content": text[:1000],  # Limit to 1k chars per chunk (simplified)
            "embedding": json.dumps(embedding)
        }).execute()

        update_document_status(doc_id, "processed")
        time.sleep(1)
    except Exception as e:
        logging.error(f"❌ Error processing document {doc_id}: {e}")
        update_document_status(doc_id, "failed")

# ----------------------------------------
# Main worker loop
# ----------------------------------------
def main():
    logging.info("🟢 Ingestion worker started...")
    while True:
        try:
            docs = fetch_pending_documents()
            for doc in docs:
                process_document(doc)
            time.sleep(10 + random.randint(0, 5))
        except Exception as e:
            logging.error(f"Worker loop error: {e}")
            time.sleep(15)

if __name__ == "__main__":
    main()
