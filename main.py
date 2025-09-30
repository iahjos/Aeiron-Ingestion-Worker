import os
import json
import asyncio
import psycopg
from fastapi import FastAPI, UploadFile, Form
from openai import OpenAI
from datetime import datetime

# Load env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DB_URL_DIRECT = os.getenv("DATABASE_URL_DIRECT")
DB_URL_POOLER = os.getenv("DATABASE_URL_POOLER")
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
# Ingestion Endpoint
# -------------------------
@app.post("/upload")
async def upload_file(file: UploadFile, org_id: str = Form(...), user_id: str = Form(...)):
    """
    Uploads a document (PDF/DOCX/TXT for now).
    Stores file metadata in `documents`, processes into chunks, stores in `doc_chunks`.
    """
    import fitz  # PyMuPDF
    from textwrap import wrap

    contents = await file.read()
    filename = file.filename

    # Save metadata
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                insert into documents (org_id, uploader_id, name, storage_path, status)
                values (%s, %s, %s, %s, 'uploaded')
                returning id
            """, (org_id, user_id, filename, f"local/{filename}"))
            doc_id = cur.fetchone()[0]

    # ---- Extract text (only PDF shown; extend later) ----
    text = ""
    if filename.lower().endswith(".pdf"):
        pdf = fitz.open(stream=contents, filetype="pdf")
        for page in pdf:
            text += page.get_text("text") + "\n"

    # ---- Chunk text ----
    # simple splitter, refine later
    chunks = wrap(text, 1000)

    # ---- Embed + insert ----
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

    # Update status
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("update documents set status='ready' where id=%s", (doc_id,))

    return {"doc_id": doc_id, "chunks": len(chunks)}

# -------------------------
# Chat Endpoint
# -------------------------
@app.post("/chat")
async def chat(org_id: str = Form(...), user_id: str = Form(...), question: str = Form(...)):
    """
    Chat with company docs. Retrieves org's chunks, builds context, calls GPT.
    """
    # Embed query
    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    # Retrieve top chunks
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

    # Build prompt
    system_prompt = f"You are {org_id}'s internal AI assistant. Use only the provided context."
    user_prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer with citations like [chunk_id]."

    # Call GPT
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    answer = completion.choices[0].message.content

    # Log query
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
