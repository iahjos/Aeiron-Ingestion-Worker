from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from openai import OpenAI
import os
import uuid
from typing import Optional, List, Dict, Any

# --- ENV VARIABLES ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # service role (server-only)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing env vars: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Aeiron Chat API", version="2.0")

# ----------------------------
# Models
# ----------------------------
class AskRequest(BaseModel):
    org_id: str
    user_id: str                 # auth.users.id (pass from frontend)
    question: str
    session_id: Optional[str] = None
    top_k: int = 5
    history_limit: int = 20      # last N messages for memory


class AskResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    sources: List[Dict[str, Any]]


# ----------------------------
# Helpers
# ----------------------------
def create_session(org_id: str, created_by: str, title: str = "New chat") -> str:
    session_id = str(uuid.uuid4())
    supabase.table("chat_sessions").insert({
        "id": session_id,
        "org_id": org_id,
        "created_by": created_by,
        "title": title
    }).execute()
    return session_id


def get_session(org_id: str, session_id: str):
    res = supabase.table("chat_sessions").select("*").eq("id", session_id).eq("org_id", org_id).limit(1).execute()
    return (res.data or [None])[0]


def fetch_history(session_id: str, limit: int = 20):
    # oldest -> newest for OpenAI messages
    res = (
        supabase.table("chat_messages")
        .select("role,content")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return res.data or []


def save_message(org_id: str, session_id: str, user_id: Optional[str], role: str, content: str):
    supabase.table("chat_messages").insert({
        "org_id": org_id,
        "session_id": session_id,
        "user_id": user_id,
        "role": role,
        "content": content
    }).execute()


def vector_search(org_id: str, query_embedding: List[float], top_k: int):
    # expects your existing SQL RPC function: match_documents(match_count int, org_id_param uuid, query_embedding vector)
    res = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "org_id_param": org_id
        }
    ).execute()
    return res.data or []


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"status": "Aeiron Chat API is live (v2).", "endpoints": ["/ask"]}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    org_id = payload.org_id
    user_id = payload.user_id
    question = payload.question.strip()

    if not org_id or not user_id or not question:
        raise HTTPException(status_code=400, detail="org_id, user_id, and question are required.")

    # 1) Ensure session exists (or create one)
    session_id = payload.session_id
    if session_id:
        session = get_session(org_id, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="session_id not found for this org.")
    else:
        # set a nicer default title from the first message
        title = question[:60] if len(question) > 0 else "New chat"
        session_id = create_session(org_id, user_id, title=title)

    # 2) Load chat memory
    history = fetch_history(session_id, limit=max(0, min(payload.history_limit, 50)))

    # 3) Embed question
    embed = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    query_embedding = embed.data[0].embedding

    # 4) Vector search (org-scoped)
    matches = vector_search(org_id, query_embedding, top_k=max(1, min(payload.top_k, 12)))

    # Turn matches into a “sources” payload (safe to show in UI)
    sources = []
    for m in matches:
        sources.append({
            "doc_id": m.get("doc_id"),
            "chunk_id": m.get("id"),
            "similarity": m.get("similarity"),
            "content": (m.get("content") or "")[:600]  # snippet for UI
        })

    # 5) Build context
    context = "\n\n---\n\n".join([m.get("content", "") for m in matches if m.get("content")])

    # 6) Save user message
    save_message(org_id, session_id, user_id, "user", question)

    # 7) Generate answer with memory + retrieved context
    messages = [
        {
            "role": "system",
            "content": (
                "You are Aeiron, an org-scoped assistant. "
                "Answer using the provided document context when possible. "
                "If the context does not contain the answer, say you don't know and ask a clarifying question."
            )
        }
    ]

    # add history
    for h in history[-20:]:
        if h["role"] in ("user", "assistant", "system"):
            messages.append({"role": h["role"], "content": h["content"]})

    # add retrieval + question
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    })

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    answer = completion.choices[0].message.content or ""

    # 8) Save assistant message
    save_message(org_id, session_id, None, "assistant", answer)

    return {
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "sources": sources
    }
