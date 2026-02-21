from fastapi import FastAPI
from pydantic import BaseModel
from query import embeddings_query, generate_answer
from orchestrator import app as orchestrator_app


app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/query", response_model=QueryResponse)
def ask(request: QueryRequest):
    result = orchestrator_app.invoke({"question": request.question})
    return QueryResponse(answer=result['answer'], sources=result.get('sources', []))

