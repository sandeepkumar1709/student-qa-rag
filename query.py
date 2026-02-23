import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

USE_LOCAL = os.getenv('USE_LOCAL', 'false').lower() == 'true'

# Embeddings always use Nebius — consistent, cheap, no re-ingestion needed
embed_client = OpenAI(
    api_key=os.getenv('NEBIUS_API_KEY'),
    base_url=os.getenv('NEBIUS_BASE_URL')
)
EMBED_MODEL = 'BAAI/bge-en-icl'

# LLM switches based on USE_LOCAL — this is where cost/privacy tradeoff matters
if USE_LOCAL:
    # Ollama runs locally — air gap mode, no internet required for generation
    llm_client = OpenAI(
        api_key='ollama',
        base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
    )
    LLM_MODEL = 'gemma3:4b'
else:
    # Nebius cloud LLM
    llm_client = OpenAI(
        api_key=os.getenv('NEBIUS_API_KEY'),
        base_url=os.getenv('NEBIUS_BASE_URL')
    )
    LLM_MODEL = 'Qwen/Qwen3-235B-A22B-Instruct-2507'

print(f"[INFO] Generation mode: {'LOCAL (Ollama)' if USE_LOCAL else 'CLOUD (Nebius)'}")

chroma_client = chromadb.PersistentClient(path='./chroma_db')
collection = chroma_client.get_or_create_collection(name='academic_papers')

def embeddings_query(query):
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
