import os
import json
import asyncio
import psycopg
import requests
from fastapi import FastAPI, UploadFile, Form
from openai import OpenAI
from datetime import datetime
import fitz  # PyMuPDF
from textwrap import wrap

# -------------------------
# Load env
# -------------------------
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
# Processor: fetch + chunk + embed
# -------------------------
async def process_document(doc_id, org_id, contents, filename):
    """Extracts, chunks, embeds, stores a document."""
    text = ""
    if filename.lower().endswith(".pdf"):
        pdf = fitz.open(stream=contents, filetype="pdf")
        for page in pdf:
            text += page.get_text("text") + "\n"
    else:
        try:
            text = contents.decode("utf-8")
        except Exception:
            text = ""

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

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("update documents set status='ready' where id=%s", (doc_id,))
    print(f"✅ Finished processing doc {doc_id} ({len(chunks)} chunks)")


# -------------------------
# Listener: reacts to NOTIFY
# -------------------------
async def listen_for_notifications():
    async with await psycopg.AsyncConnection.connect(DB_URL_DIRECT) as conn:
        async with conn.cursor() as cur:
            await cur.execute("LISTEN ingest_channel;")
            print("👂 Listening on ingest_channel")

            async for notify in conn.notifies():
                print("📢 Got NOTIFY:", notify.payload)
                try:
                    payload = json.loads(notify.payload)
                    doc_id = payload.get("doc_id")
                    org_id = payload.get("org_id")
                    filename = payload.get("name")
                    storage_path = payload.get("storage_path")

                    print(f"⚙️ Processing doc {doc_id} for org {org_id}")

                    # Fetch file from Supabase storage
                    url = f"{SUPABASE_URL}/storage/v1/object/{storage_path}"
                    headers = {"Authorization": f"Bearer {SUPABASE_KEY}"}
                    resp = requests.get(url, headers=headers)
                    resp.raise_for_status()

                    await process_document(doc_id, org_id, resp.content, filename)

                except Exception as e:
                    print("❌ Error in listener:", e)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(listen_for_notifications())


# -------------------------
# Upload (manual, for testing)
# -------------------------
@app.post("/upload")
async def upload_file(file: UploadFile, org_id: str = Form(...), user_id: str = Form(...)):
    """
    Manual upload endpoint. Stores file metadata in `documents`, 
    and also processes it immediately.
    """
    contents = await file.read()
    filename = file.filename

    # Insert metadata
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                insert into documents (org_id, uploader_id, name, storage_path, status)
                values (%s, %s, %s, %s, 'uploaded')
                returning id
            """, (org_id, user_id, filename, f"documents/{filename}"))
            doc_id = cur.fetchone()[0]

    # Process now
    await process_document(doc_id, org_id, contents, filename)
    return {"doc_id": doc_id}


# -------------------------
# Chat
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

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are {org_id}'s internal AI assistant."},
            {"role": "user", "content": f"Q: {question}\n\nContext:\n{context}"}
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

async def listen_for_notifications():
    print("🔔 Listening on ingest_channel")
    async with await psycopg.AsyncConnection.connect(DB_URL_DIRECT) as conn:
        async with conn.cursor() as cur:
            await cur.execute("LISTEN ingest_channel;")
            while True:
                msg = await conn.notifies.get()
                print(f"📩 Notification: {msg.payload}")
                # TODO: decode JSON payload and call ingestion logic
                data = json.loads(msg.payload)
                org_id = data["org_id"]
                doc_id = data["doc_id"]
                uploader_id = data["uploader_id"]
                # -> you can reuse your ingestion code here

# Start listener on startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(listen_for_notifications())