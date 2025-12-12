# --- Web Werver --- #

import os
import time
import uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from supabase import create_client, Client
from openai import OpenAI


# =========================
# ENV / CLIENT SETUP
# =========================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # server-side only
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Aeiron Chat API", version="1.1")


# =========================
# CORS (so your dashboard can call this)
# =========================
# In production, set ALLOWED_ORIGINS="https://aeiron.com,https://www.aeiron.com"
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = (
    ["*"] if allowed_origins_env.strip() == "*" else [o.strip() for o in allowed_origins_env.split(",")]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# (Optional) In-memory fallback "memory"
# =========================
# If you haven’t created chat tables yet, this allows short-lived memory by chat_id.
# NOTE: resets when the service restarts.
_IN_MEMORY_CHATS: Dict[str, List[Dict[str, str]]] = {}


# =========================
# REQUEST / RESPONSE MODELS
# =========================
class QueryRequest(BaseModel):
    org_id: str
    question: str
    top_k: int = 5


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    org_id: str
    chat_id: Optional[str] = None  # if not provided, we create a new chat
    message: str
    top_k: int = 5
    history_limit: int = 12  # how many prior messages to include
    return_sources: bool = True


class SourceMatch(BaseModel):
    chunk_id: str
    similarity: Optional[float] = None
    content_preview: str


class ChatResponse(BaseModel):
    chat_id: str
    answer: str
    sources: List[SourceMatch] = []
    used_chunks: int = 0


# =========================
# HELPERS
# =========================
def _embed(text: str) -> List[float]:
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding


def _vector_search(org_id: str, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
    # expects your SQL function: match_documents(match_count int, org_id_param uuid, query_embedding vector(1536))
    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "org_id_param": org_id,
        }
    ).execute()

    return result.data or []


def _safe_preview(text: str, n: int = 280) -> str:
    t = (text or "").strip().replace("\n", " ")
    return t[:n] + ("…" if len(t) > n else "")


def _db_enabled() -> bool:
    # Turn on DB persistence when you’re ready
    # ENABLE_CHAT_DB=true
    return os.getenv("ENABLE_CHAT_DB", "false").lower() == "true"


def _get_chat_history_db(chat_id: str, limit: int) -> List[Dict[str, str]]:
    """
    Requires tables:
      public.chat_messages(chat_id uuid, role text, content text, created_at timestamptz)
    """
    try:
        resp = (
            supabase.table("chat_messages")
            .select("role,content,created_at")
            .eq("chat_id", chat_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        rows = resp.data or []
        return [{"role": r["role"], "content": r["content"]} for r in rows]
    except Exception:
        # table not present / RLS / etc.
        return []


def _append_chat_message_db(chat_id: str, org_id: str, role: str, content: str) -> None:
    """
    Requires tables:
      public.chat_sessions(id uuid, org_id uuid, created_at timestamptz)
      public.chat_messages(id uuid, chat_id uuid, org_id uuid, role text, content text, created_at timestamptz)
    """
    try:
        # Ensure session exists (idempotent)
        supabase.table("chat_sessions").upsert(
            {"id": chat_id, "org_id": org_id},
            on_conflict="id"
        ).execute()

        supabase.table("chat_messages").insert(
            {"chat_id": chat_id, "org_id": org_id, "role": role, "content": content}
        ).execute()
    except Exception:
        # if tables don’t exist yet, ignore
        pass


def _get_chat_history(chat_id: str, limit: int) -> List[Dict[str, str]]:
    if _db_enabled():
        hist = _get_chat_history_db(chat_id, limit)
        if hist:
            return hist

    # fallback memory
    return _IN_MEMORY_CHATS.get(chat_id, [])[-limit:]


def _append_chat_message(chat_id: str, org_id: str, role: str, content: str) -> None:
    if _db_enabled():
        _append_chat_message_db(chat_id, org_id, role, content)

    # always keep in-memory too (helpful even if DB enabled)
    _IN_MEMORY_CHATS.setdefault(chat_id, [])
    _IN_MEMORY_CHATS[chat_id].append({"role": role, "content": content})


# =========================
# ROUTES
# =========================
@app.get("/")
def root():
    return {"status": "Aeiron Chat API is live."}


@app.post("/ask")
def ask(request: QueryRequest):
    # 1) embed
    query_embedding = _embed(request.question)

    # 2) retrieve
    matches = _vector_search(request.org_id, query_embedding, request.top_k)

    # 3) context
    context = "\n\n".join([m.get("content", "") for m in matches])

    # 4) answer
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Aeiron, an expert assistant answering based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"}
        ],
    )
    answer = completion.choices[0].message.content

    return {"question": request.question, "answer": answer, "matches": matches}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Session-memory chat:
      - Use a NEW chat_id for a "new chat" (new memory)
      - Reuse the same chat_id to continue the same conversation
    """
    # Create chat_id if missing
    chat_id = req.chat_id or str(uuid.uuid4())

    # Save user message
    _append_chat_message(chat_id, req.org_id, "user", req.message)

    # Embed the *latest* user message (simple + effective)
    query_embedding = _embed(req.message)

    # Retrieve org-scoped chunks
    matches = _vector_search(req.org_id, query_embedding, req.top_k)

    # Build context + sources
    context_chunks = [m.get("content", "") for m in matches if m.get("content")]
    context = "\n\n".join(context_chunks)

    sources: List[SourceMatch] = []
    if req.return_sources:
        for m in matches:
            chunk_id = str(m.get("id", ""))
            sources.append(
                SourceMatch(
                    chunk_id=chunk_id,
                    similarity=m.get("similarity"),
                    content_preview=_safe_preview(m.get("content", "")),
                )
            )

    # Pull memory for this chat_id
    history = _get_chat_history(chat_id, req.history_limit)

    # Compose messages
    system_prompt = (
        "You are Aeiron, an expert organization-scoped assistant. "
        "Answer using the provided context. If the context is insufficient, say so."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Include chat history (memory)
    # We will replay a short window of prior turns
    for msg in history:
        if msg["role"] in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current retrieval context as the final user message wrapper
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nUser message: {req.message}"
    })

    # Generate answer
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    # Save assistant message
    _append_chat_message(chat_id, req.org_id, "assistant", answer)

    return ChatResponse(
        chat_id=chat_id,
        answer=answer,
        sources=sources,
        used_chunks=len(context_chunks),
    )
