import os
import json
import asyncio
import psycopg
import requests
from fastapi import FastAPI, UploadFile, Form
from openai import OpenAI
from datetime import datetime

# PDF/text extraction
import fitz  # PyMuPDF
from textwrap import wrap

# Load env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DB_URL_DIRECT = os.getenv("DATABASE_URL_DIRECT")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Clients
client = OpenAI(api_key=OPENAI_KEY)
app = FastAPI()

# -------------------------
# Database helper
# -------------------------
def get_conn():
    return psycopg.connect(DB_URL_DIRECT, autocommit=True)

# -------------------------
# Ingestion logic (reusable)
# -------------------------
async def process_document(org_id, doc_id, uploader_id, name, storage_path, mime_type):
    """
    Pull file text, split into chunks, embed, and save to doc_chunks.
    """
    print(f"🚀 Processing document {doc_id} for org {org_id}")

    # For demo, assume PDF only (can extend later)
    text = ""
    if mime_type == "application/pdf":
        # fetch from Supabase storage if needed later
        # right now assume it's in local/dev path
        pdf_path = storage_path
        if os.path.exists(pdf_path):
            pdf = fitz.open(pdf_path)
            for page in pdf:
                text += page.get_text("text") + "\n"
        else:
            print(f"⚠️ File not found at {pdf_path}, skipping text extraction")
            return

    chunks = wrap(text, 1000)

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    insert into doc_chunks (org_id, doc_id, chunk_index, content, embedding)
                    values (%s, %s, %s, %s, %s)
                """, (org_id, doc_id, i, chunk, emb))

    # Update document status
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("update documents set status='ready' where id=%s", (doc_id,))

    print(f"✅ Finished processing {doc_id}, chunks: {len(chunks)}")

# -------------------------
# Upload (manual, testing)
# -------------------------
@app.post("/upload")
async def upload_file(file: UploadFile, org_id: str = Form(...), user_id: str = Form(...)):
    """
    Manual upload endpoint. Stores metadata + processes immediately.
    """
    contents = await file.read()
    filename = file.filename

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                insert into documents (org_id, uploader_id, name, storage_path, status, mime_type)
                values (%s, %s, %s, %s, 'uploaded', %s)
                returning id
            """, (org_id, user_id, filename, f"local/{filename}", file.content_type))
            doc_id = cur.fetchone()[0]

    # save file locally for testing
    os.makedirs("local", exist_ok=True)
    with open(f"local/{filename}", "wb") as f:
        f.write(contents)

    await process_document(org_id, doc_id, user_id, filename, f"local/{filename}", file.content_type)

    return {"doc_id": doc_id, "status": "ready"}

# -------------------------
# Chat Endpoint
# -------------------------
@app.post("/chat")
async def chat(org_id: str = Form(...), user_id: str = Form(...), question: str = Form(...)):
    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                select id, content, 1 - (embedding <=> %s::vector) as score
                from doc_chunks
                where org_id = %s
                order by embedding <=> %s::vector
                limit 8
            """, (query_emb, org_id, query_emb))
            rows = cur.fetchall()

    context = "\n\n".join([f"[{r[0]}] {r[1]}" for r in rows])

    system_prompt = f"You are {org_id}'s internal AI assistant. Use only the provided context."
    user_prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer with citations like [chunk_id]."

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    answer = completion.choices[0].message.content

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                insert into queries (org_id, user_id, question, created_at)
                values (%s, %s, %s, now())
                returning id
            """, (org_id, user_id, question))
            qid = cur.fetchone()[0]

            cur.execute("""
                insert into answers (query_id, content, model)
                values (%s, %s, %s)
            """, (qid, answer, "gpt-4o-mini"))

    return {"answer": answer, "citations": [r[0] for r in rows]}

# -------------------------
# Healthcheck
# -------------------------
@app.get("/")
def root():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

# -------------------------
# Listen for DB Notifications
# -------------------------
async def listen_for_notifications():
    async with await psycopg.AsyncConnection.connect(DB_URL_DIRECT) as conn:
        await conn.execute("LISTEN ingest_channel;")
        print("🔔 Listening on ingest_channel")

        async for notify in conn.notifies():
            print("📨 Got notification:", notify.payload)
            try:
                payload = json.loads(notify.payload)
                await process_document(
                    org_id=payload["org_id"],
                    doc_id=payload["doc_id"],
                    uploader_id=payload["uploader_id"],
                    name=payload["name"],
                    storage_path=payload["storage_path"],
                    mime_type=payload["mime_type"]
                )
            except Exception as e:
                print("❌ Error processing notification:", e)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(listen_for_notifications())
