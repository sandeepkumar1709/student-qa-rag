import pypdf
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv('NEBIUS_API_KEY'),
    base_url=os.getenv('NEBIUS_BASE_URL')
)

chroma_client = chromadb.PersistentClient(path='./chroma_db')
collection = chroma_client.get_or_create_collection(name='academic_papers')

def embed_and_store(chunks, filename):
    response = client.embeddings.create(
            model='BAAI/bge-en-icl',
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    return chunks



def extract_text_from_pdf(pdf_path):
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
