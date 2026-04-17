"""
ingest.py — One-time PDF ingestion pipeline for the Student Q&A system.

Reads all PDF files from the 'Sample PDFs/' directory, extracts text,
splits it into overlapping chunks, generates vector embeddings via the
configured LLM provider, and stores everything in ChromaDB for later retrieval.

Run this script once before starting the API, and re-run whenever new PDFs
are added. Re-ingestion is safe — ChromaDB will overwrite existing IDs.
"""

import pypdf
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv('LLM_API_KEY'),
    base_url=os.getenv('LLM_BASE_URL')
)

chroma_client = chromadb.PersistentClient(path='./chroma_db')
collection = chroma_client.get_or_create_collection(name='academic_papers')

def embed_and_store(chunks, filename):
    """
    Generate embeddings for all chunks and store them in ChromaDB.

    Sends the full list of chunks to the embedding API in one batch call,
    then adds each chunk with its embedding and source filename as metadata.
    IDs are deterministic (filename + index) so re-ingestion overwrites safely.
    """
    response = client.embeddings.create(
            model=os.getenv('EMBED_MODEL', 'Qwen/Qwen3-Embedding-8B'),
            input=chunks
        )
    for i, each_response in enumerate(response.data):
        embedding = each_response.embedding
        collection.add(
            ids=[f"{filename}_{i}"],
            embeddings=[embedding],
            documents=[chunks[i]],
            metadatas=[{"source": filename}]
            )
    print(f"Stored {len(chunks)} chunks from {filename} in ChromaDB")


def chunk_text(text, filename):
    """
    Split extracted PDF text into overlapping chunks for embedding.

    Uses LangChain's RecursiveCharacterTextSplitter with 1000-char chunks
    and 200-char overlap so context isn't lost at chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    return chunks



def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file by concatenating text from every page."""
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text



def main():
    pdf_path = 'Sample PDFs'
    for filename in os.listdir(pdf_path):
        if filename.endswith('.pdf'):
            full_path = os.path.join(pdf_path, filename)
            text = extract_text_from_pdf(full_path)
            chunks = chunk_text(text, filename)
            embed_and_store(chunks, filename)
            print(f'{filename} → {len(chunks)} chunks')


if __name__ == '__main__':
    main()
