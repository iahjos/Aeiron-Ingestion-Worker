import os
import io
import json
import asyncio
import tempfile
from typing import List, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

import asyncpg
import pandas as pd
import fitz              # PyMuPDF for PDFs
import docx              # python-docx
from supabase import create_client, Client

# ---- OpenAI (new SDK) ----
try:
    from openai import OpenAI
    _openai_client = OpenAI()
    def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        # batch once to keep it simple; could batch in chunks of 100+ if needed
        resp = _openai_client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]
except Exception as e:
    # If the new SDK isn't available, fall back to the legacy import for dev
    import openai as _openai_legacy
    def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        resp = _openai_legacy.Embedding.create(model=model, input=texts)
        return [d["embedding"] for d in resp["data"]]

# ------------ Load env early ------------
load_dotenv()  # IMPORTANT: make sure .env is read when running in Docker

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DATABASE_URL  = os.getenv("DATABASE_URL_POOLER") or os.getenv("DATABASE_URL_DIRECT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Fail fast if any are missing
missing = [k for k,v in [
    ("SUPABASE_URL", SUPABASE_URL),
    ("SUPABASE_KEY", SUPABASE_KEY),
    ("DATABASE_URL_POOLER or DATABASE_URL_DIRECT", DATABASE_URL),
    ("OPENAI_API_KEY", OPENAI_API_KEY),
] if not v]
if missing:
    raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

# legacy openai var needed if using legacy path above
os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- FastAPI ----------
app = FastAPI(title="Aeiron Ingestion Worker")

# --------- Models for /health & /ask (optional) ----------
class AskRequest(BaseModel):
    org_id: str
    question: str
    match_threshold: float = 0.3
    match_count: int = 8
    debug: bool = False

@app.get("/health")
def health():
    return {"status": "ok"}

# --------- Helpers ----------
def chunk_text(txt: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    txt = (txt or "").strip()
    if not txt:
        return []
    chunks = []
    start = 0
    n = len(txt)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(txt[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    pages = [page.get_text("text") for page in doc]
    return "\n".join(pages).strip()

def extract_text_from_docx(file_path: str) -> str:
    d = docx.Document(file_path)
    return "\n".join([p.text for p in d.paragraphs]).strip()

def extract_text_from_csv(file_path: str) -> str:
    # Robust CSV → text: join header + rows with CSV-like formatting
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        # try common encodings
        for enc in ("utf-8-sig","latin-1"):
            try:
                df = pd.read_csv(file_path, encoding=enc)
                break
            except Exception:
                df = None
        if df is None:
            raise
    if df is None or df.empty:
        return ""
    # Convert to a readable text table: header then rows
    lines = []
    lines.append(", ".join(map(str, df.columns.tolist())))
    for row in df.itertuples(index=False, name=None):
        lines.append(", ".join(map(lambda x: "" if x is None else str(x), row)))
    return "\n".join(lines)

def guess_extractor(mime: str, ext: str):
    ext = (ext or "").lower()
    mime = (mime or "").lower()
    if ext.endswith(".pdf") or "pdf" in mime:
        return "pdf"
    if ext.endswith(".docx") or "word" in mime:
        return "docx"
    if ext.endswith(".csv") or "csv" in mime:
        return "csv"
    return "unknown"

async def download_to_temp(bucket: str, path: str) -> str:
    # Download using supabase storage and write to a temp file.
    # NOTE: assumes the service key has access.
    print(f"📥 Downloading: bucket={bucket} path={path}")
    res = supabase.storage.from_(bucket).download(path)
    # Some supabase clients return bytes directly; some return HTTPResponse-like objs.
    file_bytes = res if isinstance(res, (bytes, bytearray)) else getattr(res, "content", None)
    if not file_bytes:
        raise RuntimeError("Download returned empty bytes")
    print(f"   → bytes: {len(file_bytes)}")
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(file_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name

async def insert_chunks(conn, org_id: str, doc_id: str, chunks: List[str]) -> int:
    if not chunks:
        return 0
    embeddings = embed_texts(chunks)  # list of vectors
    # Insert with asyncpg executemany
    sql = """
        insert into doc_chunks (org_id, doc_id, chunk_index, content, embedding)
        values ($1, $2, $3, $4, $5)
    """
    rows = [(org_id, doc_id, i, chunks[i], embeddings[i]) for i in range(len(chunks))]
    await conn.executemany(sql, rows)
    return len(rows)

async def process_one_job(conn, job: dict):
    job_id     = job["id"]
    org_id     = job["org_id"]
    doc_id     = job["doc_id"]
    bucket     = job["bucket"]
    path       = job["path"]
    mime_type  = job.get("mime_type") or ""
    extension  = job.get("extension") or ""

    print(f"\n▶️  Processing job {job_id} (org={org_id}, doc={doc_id})")

    await conn.execute("update ingestion_queue set status='processing', error=null where id=$1", job_id)

    # Download file to temp
    local_path = await download_to_temp(bucket, path)

    # Extract text
    which = guess_extractor(mime_type, extension)
    try:
        if which == "pdf":
            text = extract_text_from_pdf(local_path)
        elif which == "docx":
            text = extract_text_from_docx(local_path)
        elif which == "csv":
            text = extract_text_from_csv(local_path)
        else:
            raise RuntimeError(f"Unsupported file type: mime={mime_type} ext={extension}")
    except Exception as e:
        await conn.execute(
            "update ingestion_queue set status='failed', error=$2 where id=$1",
            job_id, f"extract_error: {e}"
        )
        print(f"❌ Extract error: {e}")
        return

    print(f"🧾 Extracted text length: {len(text)}")
    if len(text) == 0:
        await conn.execute(
            "update ingestion_queue set status='failed', error=$2 where id=$1",
            job_id, "empty_text_after_extraction"
        )
        print("❌ Empty text; aborting.")
        return

    # Chunk + embed + insert
    chunks = chunk_text(text)
    print(f"🪓 Chunk count: {len(chunks)}")
    if not chunks:
        await conn.execute(
            "update ingestion_queue set status='failed', error=$2 where id=$1",
            job_id, "no_chunks_generated"
        )
        print("❌ No chunks generated; aborting.")
        return

    try:
        inserted = await insert_chunks(conn, org_id, doc_id, chunks)
        print(f"✅ Inserted chunks: {inserted}")
    except Exception as e:
        await conn.execute(
            "update ingestion_queue set status='failed', error=$2 where id=$1",
            job_id, f"insert_error: {e}"
        )
        print(f"❌ Insert error: {e}")
        return

    # Success
    await conn.execute("update ingestion_queue set status='done', error=null where id=$1", job_id)
    print("🎉 Job completed.")

async def process_queue_forever():
    print("📡 Queue processor loop starting...")
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        while True:
            rows = await conn.fetch("""
                select * from ingestion_queue
                where status in ('queued','retry')
                order by created_at asc
                limit 1
            """)
            if not rows:
                await asyncio.sleep(3)
                continue

            job = dict(rows[0])
            try:
                async with conn.transaction():
                    await process_one_job(conn, job)
            except Exception as e:
                print(f"❌ Unexpected job error: {e}")
                try:
                    await conn.execute(
                        "update ingestion_queue set status='failed', error=$2 where id=$1",
                        job["id"], f"unexpected_error: {e}"
                    )
                except Exception:
                    pass
            # small pause to avoid hot loop
            await asyncio.sleep(0.5)
    finally:
        await conn.close()

@app.on_event("startup")
async def on_startup():
    print("🚀 Worker starting…")
    asyncio.create_task(process_queue_forever())
    print("✅ Queue processor started")
