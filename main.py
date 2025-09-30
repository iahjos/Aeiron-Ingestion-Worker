import os
import json
import asyncio
import psycopg
import requests
import socket
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
# Debug: DNS resolution check
# -------------------------
def debug_ipv4_resolution():
    try:
        host = DB_URL_DIRECT.split("@")[1].split(":")[0]
        print("🌐 Testing DNS resolution for host:", host)
        for res in socket.getaddrinfo(host, 5432):
            print(" ->", res[0].name, res[4][0])
    except Exception as e:
        print("❌ DNS resolution check failed:", e)


# -------------------------
# Ingestion logic
# -------------------------
async def process_document(org_id, doc_id, uploader_id, name, storage_path, mime_type):
    print(f"🚀 Processing document {doc_id} for org {org_id}")

    text = ""
    if mime_type == "application/pdf":
        if os.path.exists(storage_path):
            pdf = fitz.open(storage_path)
            for page in pdf:
                text += page.get_text("text") + "\n"
        else:
            print(f"⚠️ File {storage_path} not found locally")
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

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("update documents set status='ready' where id=%s", (doc_id,))

    print(f"✅ Finished processing {doc_id}, chunks: {len(chunks)}")


# -------------------------
# Listener for notifications
# -------------------------
async def listen_for_notifications():
    conn = await psycopg.AsyncConnection.connect(DB_URL_DIRECT)
    async with conn.cursor() as cur:
        await cur.execute("LISTEN ingest_channel;")
        print("🔔 Listening on ingest_channel")

    try:
        async for notify in conn.notifies():
            print("📨 Got notification:", notify.payload)
            try:
                data = json.loads(notify.payload)
                await process_document(
                    data["org_id"],
                    data["doc_id"],
                    data["uploader_id"],
                    data["name"],
                    data["storage_path"],
                    data["mime_type"],
                )
            except Exception as e:
                print("❌ Error handling notification:", e)
    except Exception as e:
        print("🔥 Listener crashed:", e)
    finally:
        await conn.close()

@app.on_event("startup")
async def startup_event():
    debug_ipv4_resolution()
    asyncio.create_task(listen_for_notifications())


# -------------------------
# Upload endpoint
# -------------------------
@app.post("/upload")
async def upload_file(file: UploadFile, org_id: str = Form(...), user_id: str = Form(...)):
    contents = await file.read()
    filename = file.filename

    os.makedirs("local", exist_ok=True)
    local_path = f"local/{filename}"
    with open(local_path, "wb") as f:
        f.write(contents)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                insert into documents (org_id, uploader_id, name, storage_path, status, mime_type)
                values (%s, %s, %s, %s, 'uploaded', %s)
                returning id
            """, (org_id, user_id, filename, local_path, file.content_type))
            doc_id = cur.fetchone()[0]

    await process_document(org_id, doc_id, user_id, filename, local_path, file.content_type)

    return {"doc_id": doc_id, "status": "ready"}


# -------------------------
# Chat endpoint
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
