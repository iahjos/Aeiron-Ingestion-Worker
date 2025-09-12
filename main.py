from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from supabase import create_client
import openai
import os
import fitz  # PyMuPDF for PDFs
import docx
import pandas as pd

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
    match_threshold: float = 0.75   # default if not provided
    match_count: int = 5            # default if not provided


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

    # Extract text
    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_location)
    elif file.filename.endswith(".docx"):
        text = extract_text_from_docx(file_location)
    elif file.filename.endswith(".csv"):
        text = extract_text_from_csv(file_location)
    else:
        text = ""

    # TODO: chunk + embed + store (already working in your ingestion worker)
    return {"filename": file.filename, "length": len(text)}


@app.post("/ask")
async def ask(request: AskRequest):
    """
    Accept org_id + question, retrieve context, and return GPT answer.
    """
    # 1. Embed the question
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=request.question
    )
    embedding = response.data[0].embedding

    # 2. Retrieve top chunks from Supabase (using your match_documents function)
    results = supabase.rpc("match_documents", {
        "query_embedding": embedding,
        "match_threshold": request.match_threshold,
        "match_count": request.match_count,
        "org_id": request.org_id
    }).execute()

    chunks = " ".join([r["content"] for r in results.data]) if results.data else ""

    # 3. Call GPT with the retrieved context
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful company assistant. Only answer using the provided context."},
            {"role": "user", "content": f"Question: {request.question}\n\nContext: {chunks}"}
        ]
    )

    return {"answer": completion.choices[0].message.content}
