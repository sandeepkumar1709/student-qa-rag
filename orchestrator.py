"""
orchestrator.py — LangGraph state machine that orchestrates the Q&A pipeline.

Defines a directed graph with two paths:
  - Academic path:  classify → retrieve → generate_answer_from_internal_source → END
  - Off-topic path: classify → web_search → generate_answer_from_web → END

The compiled graph is exported as `app` and invoked directly by main.py.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from query import embeddings_query, generate_answer, generate_web_answer
from openai import OpenAI
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import os
import sys

load_dotenv()

client = OpenAI(
    api_key=os.getenv('LLM_API_KEY'),
    base_url=os.getenv('LLM_BASE_URL')
)

# State — the backpack passed between all nodes
class State(TypedDict):
    question: str
    category: str
    context: str
    sources: list
    source_type: str   # "papers" or "web"
    answer: str



def classify_question(state: State) -> State:
    """
    LLM-based router: classify the question as 'academic' or 'off_topic'.

    Uses the full cloud LLM but caps output at max_tokens=10 — the model only
    needs to output a single word. This gives high classification quality at
    minimal cost. Result is stored in state['category'] for the route() function.
    """
    prompt = f"Question: {state['question']}\n\nIs this question related to academic research? Answer 'academic' or 'off_topic'."
    response = client.chat.completions.create(
        model=os.getenv('CLOUD_LLM_MODEL', 'Qwen/Qwen3-235B-A22B-Instruct-2507'),
            messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies questions."},
            {"role": "user", "content": prompt}       ],
        max_tokens=10
    )
    classification = response.choices[0].message.content.strip().lower()
    return {"category": classification}


def retrieve_context(state: State) -> State:
    """
    Retrieve relevant paper chunks from ChromaDB using vector similarity search.

    Embeds the question and queries ChromaDB for the top 5 most similar chunks.
    Populates state with the joined context text, deduplicated source filenames,
    and sets source_type to 'papers'.
    """
    results = embeddings_query(state['question'])
    context = "\n\n".join(results['documents'][0])
    sources = list(set([m['source'] for m in results['metadatas'][0]]))
    return {"context": context, "sources": sources, "source_type": "papers"}

def generate_answer_node(state: State) -> State:
    """Generate a grounded answer from retrieved paper context using the LLM."""
    answer = generate_answer(state['question'], state['context'])
    return {"answer": answer}

def generate_web_node(state: State) -> State:
    """Generate an answer synthesized from web search results using the LLM."""
    answer = generate_web_answer(state['question'], state['context'])
    return {"answer": answer}

def web_search_node(state: State) -> State:
    """
    Call the MCP server's web_search tool for off-topic (non-academic) questions.

    Spawns mcp_server.py as a subprocess using stdio transport, initialises an
    MCP client session, and calls the 'web_search' tool with the user's question.
    Returns the formatted web results as context for the generate_web node.
    Uses asyncio.run() to bridge the async MCP client into the synchronous graph.
    """
    async def _search():
        python_path = os.path.join(os.path.dirname(sys.executable), 'python.exe')
        server_params = StdioServerParameters(
            command=python_path,
            args=["mcp_server.py"]
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("web_search", {"query": state["question"]})
                return result.content[0].text
    
    context = asyncio.run(_search())
    print(f"[DEBUG] Web search context:\n{len(context)} characters")
    print(f"[DEBUG] Web search context :\n{context[:500]}")
    return {"context": context, "sources": ["web search"], "source_type": "web"}

def route(state: State) -> str:
    """
    Conditional edge function: determine which pipeline to run based on classification.

    Returns 'retrieve' for academic questions (ChromaDB path) or
    'web_search' for everything else (MCP web search path).
    """
    if "academic" in state['category']:
        return "retrieve"
    return "web_search"

# Build the graph
graph = StateGraph(State)

# Add nodes
graph.add_node("classify", classify_question)
graph.add_node("retrieve", retrieve_context)
graph.add_node("generate_answer_from_internal_source", generate_answer_node)
graph.add_node("web_search", web_search_node)
graph.add_node("generate_answer_from_web", generate_web_node)

# Add edges
graph.set_entry_point("classify")
graph.add_conditional_edges("classify", route, {
    "retrieve": "retrieve",
    "web_search": "web_search"
})
graph.add_edge("retrieve", "generate_answer_from_internal_source")
graph.add_edge("generate_answer_from_internal_source", END)
graph.add_edge("web_search", "generate_answer_from_web")
graph.add_edge("generate_answer_from_web", END)

# Compile
app = graph.compile()

# Test it
if __name__ == "__main__":
    result = app.invoke({"question": "What does the paper discuss about retrieval augmented generation evaluation?"})

    print(f"Answer: {result['answer']}")
    print(f"Sources: {result.get('sources', [])}")
    print(f"Source Type: {result.get('source_type', 'unknown')}")