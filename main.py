from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from supabase import create_client
import openai
import os
import fitz  # PyMuPDF for PDFs
import docx
import pandas as pd
import re
import requests

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Initialize Supabase client
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# ==========================
# MODELS
# ==========================

class AskRequest(BaseModel):
    org_id: str
    question: str
    # Retrieval controls
    match_threshold: float = 0.3
    match_count: int = 8
    # Debug + Mode
    debug: bool = False
    # "strict" => use ONLY org docs; "blend" => allow model priors for predictive questions
    mode: str = "blend"  # "strict" | "blend"

# ==========================
# HELPERS
# ==========================

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_string()

def is_predictive(q: str) -> bool:
    ql = q.lower()
    patterns = [
        r"\bnext year\b", r"\bupcoming\b", r"\bforecast\b", r"\bprojection\b",
        r"\bproject(ed|ion|ions)?\b", r"\bestimate(d|s)?\b", r"\boutlook\b",
        r"\btrend(s)?\b", r"\btrajectory\b", r"\bwhat will\b", r"\bexpected\b",
        r"\bshould we expect\b", r"\bfuture\b"
    ]
    return any(re.search(p, ql) for p in patterns)

def build_system_prompt(allow_general_knowledge: bool) -> str:
    if not allow_general_knowledge:
        return (
            "You are a company assistant. Answer clearly and concisely using ONLY the provided context. "
            "If the context is insufficient to answer, say you don't know and suggest which documents to consult. "
            "Never invent numbers, dates, or facts that aren't in the context."
        )
    return (
        "You are a company assistant. First, rely on the provided context. "
        "If the question is predictive or the context is insufficient, you may use general domain knowledge "
        "to produce a careful ESTIMATE. When you do, you must:\n"
        "• Prefer ranges over point estimates.\n"
        "• State key assumptions briefly.\n"
        "• Indicate uncertainty level (Low/Medium/High).\n"
        "• Never contradict the provided context; if context conflicts with general knowledge, follow the context.\n"
        "• If no reasonable estimate can be made, say so.\n"
        "Do not fabricate specific internal company numbers that aren't present in context."
    )

def summarize_chunks_for_context(rows):
    if not rows:
        return ""
    return "\n\n".join(r.get("content", "") for r in rows if r.get("content"))

# ==========================
# ROUTES
# ==========================

@app.get("/")
def root():
    return {"message": "Ingestion worker + RAG API is running."}

@app.post("/upload")
async def upload_file(file: UploadFile, org_id: str = Form(...)):
    """
    Handle file upload, parse, embed, and store in Supabase.
    """
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_location)
    elif file.filename.endswith(".docx"):
        text = extract_text_from_docx(file_location)
    elif file.filename.endswith(".csv"):
        text = extract_text_from_csv(file_location)
    else:
        text = ""

    return {"filename": file.filename, "length": len(text)}

@app.post("/ingest")
async def ingest_file(payload: dict):
    """
    Triggered by Supabase when a new document is inserted.
    Downloads, extracts, embeds, and stores chunks in Supabase.
    """
    doc_id = payload.get("doc_id")
    org_id = payload.get("org_id")
    file_path = payload.get("file_path")

    if not (doc_id and org_id and file_path):
        return {"error": "Missing doc_id, org_id, or file_path"}

    # Build Supabase public file URL
    file_url = f"{os.getenv('SUPABASE_URL')}/storage/v1/object/public/{file_path}"

    tmp_path = f"/tmp/{os.path.basename(file_path)}"
    resp = requests.get(file_url)
    with open(tmp_path, "wb") as f:
        f.write(resp.content)

    # Extract text
    if tmp_path.endswith(".pdf"):
        text = extract_text_from_pdf(tmp_path)
    elif tmp_path.endswith(".docx"):
        text = extract_text_from_docx(tmp_path)
    elif tmp_path.endswith(".csv"):
        text = extract_text_from_csv(tmp_path)
    else:
        text = ""

    if not text.strip():
        return {"error": "No text extracted"}

    # Split into chunks
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Embed + insert
    for idx, chunk in enumerate(chunks):
        emb = openai.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embedding = emb.data[0].embedding

        supabase.table("doc_chunks").insert({
            "doc_id": doc_id,
            "org_id": org_id,
            "chunk_index": idx,
            "content": chunk,
            "embedding": embedding
        }).execute()

    return {"message": f"Ingested {len(chunks)} chunks for {file_path}"}

@app.post("/ask")
async def ask(request: AskRequest):
    """
    Accept org_id + question, retrieve context, and return GPT answer.
    Blend mode: for predictive questions, carefully allow general knowledge with labels.
    """
    predictive = is_predictive(request.question)
    allow_general = (request.mode == "blend") and predictive

    emb_resp = openai.embeddings.create(
        model="text-embedding-3-small",
        input=request.question
    )
    embedding = emb_resp.data[0].embedding

    rpc_payload = {
        "query_embedding": embedding,
        "match_threshold": request.match_threshold,
        "match_count": request.match_count,
        "org_id": request.org_id
    }
    results = supabase.rpc("match_documents", rpc_payload).execute()
    rows = results.data or []
    context_text = summarize_chunks_for_context(rows)

    system_prompt = build_system_prompt(allow_general_knowledge=allow_general)
    temperature = 0.2 if not allow_general else 0.35

    user_msg = (
        f"Question: {request.question}\n\n"
        f"Context (company docs):\n{context_text if context_text else '(no relevant context retrieved)'}\n\n"
        "Instructions:\n"
        "- If using only the context, answer directly and concisely.\n"
        "- If providing an estimate (allowed in this turn), clearly label it as 'Estimate', add brief 'Assumptions', "
        "and an 'Uncertainty' level."
    )

    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
    )

    answer = completion.choices[0].message.content

    resp = {
        "answer": answer,
        "mode_used": "blend" if allow_general else "strict",
    }
    if request.debug:
        resp["retrieval"] = [
            {
                "doc_id": r.get("doc_id"),
                "chunk_index": r.get("chunk_index"),
                "similarity": r.get("similarity"),
                "snippet": (r.get("content") or "")[:400]
            } for r in rows
        ]
        resp["retrieval_params"] = {
            "match_threshold": request.match_threshold,
            "match_count": request.match_count
        }

    return resp
