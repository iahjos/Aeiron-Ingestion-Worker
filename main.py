import os
import time
import logging
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
    """Fetch pending documents from Supabase (status = 'pending')"""
    response = supabase.table("documents").select("*").eq("status", "pending").execute()
    docs = response.data or []
    logging.info(f"📄 Found {len(docs)} pending documents.")
    return docs

def update_document_status(doc_id, status):
    """Update document status in Supabase"""
    supabase.table("documents").update({"status": status}).eq("id", doc_id).execute()
    logging.info(f"✅ Updated document {doc_id} → status = {status}")

def embed_text(text):
    """Generate embedding using OpenAI"""
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding
    return embedding

def process_document(doc):
    """Main ingestion logic"""
    doc_id = doc["id"]
    org_id = doc["org_id"]
    path = doc["storage_path"]
    logging.info(f"🚀 Processing document {path} for org {org_id}")

    try:
        # Example fetch from Supabase storage (assuming bucket 'company_docs')
        file_url = f"{SUPABASE_URL}/storage/v1/object/public/{path}"
        logging.info(f"🔗 Fetching file from {file_url}")

        # For simplicity, we’ll simulate reading the document text
        # Replace this with your actual file reading & chunking logic
        fake_text = f"This is a simulated ingestion for document {path}."

        # Create embedding
        embedding = embed_text(fake_text)

        # Store chunks/embeddings into your doc_chunks table
        supabase.table("doc_chunks").insert({
            "doc_id": doc_id,
            "org_id": org_id,
            "content": fake_text,
            "embedding": embedding
        }).execute()

        # Mark document as processed
        update_document_status(doc_id, "processed")

    except Exception as e:
        logging.error(f"❌ Error processing document {doc_id}: {e}")
        update_document_status(doc_id, "failed")

# ----------------------------------------
# Main ingestion loop
# ----------------------------------------
def main():
    logging.info("🟢 Ingestion worker started and listening for jobs...")
    while True:
        try:
            docs = fetch_pending_documents()
            for doc in docs:
                process_document(doc)

            time.sleep(10)  # Poll every 10s
        except Exception as e:
            logging.error(f"Worker error: {e}")
            time.sleep(15)

if __name__ == "__main__":
    main()
