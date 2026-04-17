"""
main.py — FastAPI entry point for the Student Q&A system.

Exposes a single POST /query endpoint that accepts a student question,
delegates all processing to the LangGraph orchestrator, and returns
a structured response with an answer, source list, and source type.
"""

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
    source_type: str    # "papers" or "web"

@app.post("/query", response_model=QueryResponse)
def ask(request: QueryRequest):
    """
    Accept a student question and return an AI-generated answer.

    Routes through the LangGraph orchestrator which classifies the question,
    retrieves context (from ChromaDB or web), and generates a grounded answer.
    The response includes the answer text, source filenames or 'web search',
    and whether the source was academic papers or the web.
    """
    result = orchestrator_app.invoke({"question": request.question})
    return QueryResponse(
        answer=result['answer'],
        sources=result.get('sources', []),
        source_type=result.get('source_type', 'unknown')
    )

