from fastapi import FastAPI, Request
from pydantic import BaseModel
from supabase import create_client, Client
from openai import OpenAI
import os

# --- ENV VARIABLES ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Aeiron Chat API", version="1.0")

# --- Request Model ---
class QueryRequest(BaseModel):
    org_id: str
    question: str
    top_k: int = 5

# --- Root Endpoint ---
@app.get("/")
def root():
    return {"status": "Aeiron Chat API is live."}

# --- Ask Endpoint ---
@app.post("/ask")
def ask(request: QueryRequest):
    # 1️⃣ Embed the user question
    embed_response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=request.question
    )
    query_embedding = embed_response.data[0].embedding

    # 2️⃣ Vector similarity search
    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": request.top_k,
            "org_id_param": request.org_id
        }
    ).execute()

    matches = result.data or []

    # 3️⃣ Concatenate the most relevant chunks
    context = "\n\n".join([match["content"] for match in matches])

    # 4️⃣ Generate the AI answer
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Aeiron, an expert assistant answering based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"}
        ],
    )

    answer = completion.choices[0].message.content
    return {
        "question": request.question,
        "answer": answer,
        "matches": matches
    }
