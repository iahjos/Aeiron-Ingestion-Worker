from fastapi import FastAPI
from pydantic import BaseModel
import openai
from supabase import create_client
import os

app = FastAPI()

class AskRequest(BaseModel):
    org_id: str
    question: str

@app.post("/ask")
async def ask(request: AskRequest):
    # 1. Embed question
    embedding = openai.embeddings.create(
        model="text-embedding-3-small",
        input=request.question
    )["data"][0]["embedding"]

    # 2. Retrieve top chunks via your SQL function
    query = f"""
    select content
    from match_documents('{request.org_id}', {embedding})
    limit 5;
    """
    results = supabase.rpc("match_documents", {
        "query_embedding": embedding,
        "match_count": 5,
        "org_id": request.org_id
    }).execute()

    chunks = " ".join([r["content"] for r in results.data])

    # 3. Call GPT with context
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful company assistant. Answer only using the provided context."},
            {"role": "user", "content": f"Question: {request.question}\n\nContext: {chunks}"}
        ]
    )

    return {"answer": completion.choices[0].message.content}
