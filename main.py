import os
import json
import asyncio
import requests
import socket
from fastapi import FastAPI, UploadFile, Form
from openai import OpenAI
from datetime import datetime
import fitz  # PyMuPDF
from textwrap import wrap
from supabase import create_client, Client

# -------------------------
# Load env
# -------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Clients
client = OpenAI(api_key=OPENAI_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()


# -------------------------
# Ingestion logic
# -------------------------
async def process_document(org_id, doc_id, uploader_id, name, storage_path, mime_type):
    print(f"🚀 Processing document {doc_id} for org {org_id}")

    local_path = storage_path
    text = ""

    if mime_type == "application/pdf":
        if os.path.exists(local_path):
            pdf = fitz.open(local_path)
            for page in pdf:
                text += page.get_text("text") + "\n"
        else:
            # Try downloading from Supabase storage if not local
            print(f"📥 Downloading {storage_path} from Supabase...")
            url = f"{SUPABASE_URL}/storage/v1/object/public/{storage_path}"
            headers = {"Authorization": f"Bearer {SUPABASE_KEY}"}
            resp = requests.get(url, headers=headers)
            if resp.status_code != 200:
                print(f"❌ Failed to download {storage_path}, status {resp.status_code}")
                return
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(resp.content)
            pdf = fitz.open(local_path)
            for page in pdf:
                text += page.get_text("text") + "\n"

    chunks = wrap(text, 1000)

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding

        supabase.table("doc_chunks").insert({
            "org_id": org_id,
            "doc_id": doc_id,
            "chunk_index": i,
            "content": chunk,
            "embedding": emb
        }).execute()

    supabase.table("documents").update({"status": "ready"}).eq("id", doc_id).execute()

    print(f"✅ Finished processing {doc_id}, chunks: {len(chunks)}")


# -------------------------
# Realtime listener
# -------------------------
async def on_insert(payload):
    print("📨 New document inserted:", payload)
    data = payload["new"]

    await process_document(
        data["org_id"],
        data["id"],
        data["uploader_id"],
        data["name"],
        data["storage_path"],
        data["mime_type"]
    )


async def start_realtime_listener():
    realtime = supabase.realtime
    await realtime.connect()

    channel = realtime.channel("documents-insert")
    channel.on_postgres_changes(
        event="INSERT",
        schema="public",
        table="documents",
        callback=on_insert
    )
    await channel.subscribe()

    print("🔔 Subscribed to Realtime inserts on documents")
    await realtime.listen()


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(start_realtime_listener())


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

    # Insert doc metadata into DB
    response = supabase.table("documents").insert({
        "org_id": org_id,
        "uploader_id": user_id,
        "name": filename,
        "storage_path": local_path,
        "status": "uploaded",
        "mime_type": file.content_type
    }).execute()

    doc_id = response.data[0]["id"]

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

    rows = supabase.rpc("match_doc_chunks", {
        "query_embedding": query_emb,
        "match_count": 8,
        "org_id": org_id
    }).execute()

    context = "\n\n".join([f"[{r['id']}] {r['content']}" for r in rows.data])

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are {org_id}'s internal AI assistant."},
            {"role": "user", "content": f"Q: {question}\n\nContext:\n{context}"}
        ]
    )
    answer = completion.choices[0].message.content

    supabase.table("queries").insert({
        "org_id": org_id,
        "user_id": user_id,
        "question": question
    }).execute()

    return {"answer": answer, "citations": [r["id"] for r in rows.data]}


# -------------------------
# Healthcheck
# -------------------------
@app.get("/")
def root():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}
