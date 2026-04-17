"""
query.py — Vector search and LLM generation for the Student Q&A system.

Handles two concerns:
  1. Embedding-based retrieval: embeds a question and queries ChromaDB for
     the most semantically similar paper chunks.
  2. Answer generation: calls the configured LLM (cloud or local) with
     retrieved context to produce a grounded answer.

LLM provider is controlled by USE_LOCAL env var:
  - false (default): any OpenAI-compatible cloud API (Nebius, Groq, OpenAI, etc.)
  - true: local Ollama instance (air-gap / offline mode)

Embeddings always use the cloud provider for consistency — changing the
embedding model requires full re-ingestion of all documents.
"""

import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

USE_LOCAL = os.getenv('USE_LOCAL', 'false').lower() == 'true'

# Embeddings always use the same provider — consistent, no re-ingestion needed when switching generation LLM
embed_client = OpenAI(
    api_key=os.getenv('LLM_API_KEY'),
    base_url=os.getenv('LLM_BASE_URL')
)
EMBED_MODEL = os.getenv('EMBED_MODEL', 'Qwen/Qwen3-Embedding-8B')

# LLM switches based on USE_LOCAL — this is where cost/privacy tradeoff matters
if USE_LOCAL:
    # Ollama runs locally — air gap mode, no internet required for generation
    llm_client = OpenAI(
        api_key='ollama',
        base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
    )
    LLM_MODEL = os.getenv('LOCAL_LLM_MODEL', 'gemma3:4b')
else:
    # Cloud LLM — any OpenAI-compatible provider (Nebius, Groq, OpenAI, etc.)
    llm_client = OpenAI(
        api_key=os.getenv('LLM_API_KEY'),
        base_url=os.getenv('LLM_BASE_URL')
    )
    LLM_MODEL = os.getenv('CLOUD_LLM_MODEL', 'Qwen/Qwen3-235B-A22B-Instruct-2507')

print(f"[INFO] Generation mode: {'LOCAL (Ollama)' if USE_LOCAL else 'CLOUD (Nebius)'}")

chroma_client = chromadb.PersistentClient(path='./chroma_db')
collection = chroma_client.get_or_create_collection(name='academic_papers')

def embeddings_query(query):
    """
    Embed the query and retrieve the top 5 most similar chunks from ChromaDB.

    Returns the raw ChromaDB result dict containing 'documents' and 'metadatas'
    lists — each a list-of-lists (one inner list per query, here always length 1).
    """
    query_response = embed_client.embeddings.create(
        model=EMBED_MODEL,
        input=[query]
    )
    query_embedding = query_response.data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=['documents', 'metadatas']
    )
    return results

def generate_answer(query, context):
    """
    Generate a grounded answer from academic paper context.

    The system prompt instructs the LLM to answer only from the provided context
    and explicitly say 'I don't know' if the answer isn't present — preventing
    hallucination outside the retrieved papers.
    """
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful academic assistant. Answer questions based only on the provided context. If the answer is not in the context, say you don't know."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

def generate_web_answer(query, context):
    """
    Generate a concise answer synthesised from DuckDuckGo web search results.

    Context is a formatted string of title + snippet + URL entries returned
    by the MCP web_search tool. The LLM is prompted to be concise and informative.
    """
    prompt = f"Context from web search:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the question using the provided web search results. Be concise and informative."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()



def main():
    query = "What are the main components of a retrieval augmented generation system?"
    results = embeddings_query(query)
    context = "\n\n".join(results['documents'][0])
    sources = list(set([m['source'] for m in results['metadatas'][0]]))
    answer = generate_answer(query, context)
    print(f"Question: {query}\n")
    print(f"Answer: {answer}\n")
    print(f"Sources: {sources}")

if __name__ == '__main__':
    main()
