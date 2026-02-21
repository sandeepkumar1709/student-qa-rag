from typing import TypedDict
from langgraph.graph import StateGraph, END
from query import embeddings_query, generate_answer
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv('NEBIUS_API_KEY'),
    base_url=os.getenv('NEBIUS_BASE_URL')
)

# State — the backpack passed between all nodes
class State(TypedDict):
    question: str      # input from student
    category: str      # set by classify node: "academic" or "off_topic"
    context: str       # set by retrieve node
    sources: list      # set by retrieve node
    answer: str        # set by generate node



def classify_question(state: State) -> State:
    prompt = f"Question: {state['question']}\n\nIs this question related to academic research? Answer 'academic' or 'off_topic'."
    response = client.chat.completions.create(
        model='Qwen/Qwen3-235B-A22B-Instruct-2507',
            messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies questions."},
            {"role": "user", "content": prompt}       ],
        max_tokens=10
    )
    classification = response.choices[0].message.content.strip().lower()
    return {"category": classification}


def retrieve_context(state: State) -> State:
    results = embeddings_query(state['question'])
    context = "\n\n".join(results['documents'][0])
    sources = list(set([m['source'] for m in results['metadatas'][0]]))
    return {"context": context, "sources": sources}

def generate_answer_node(state: State) -> State:
    answer = generate_answer(state['question'], state['context'])
    return {"answer": answer}

def reject_off_topic(state: State) -> State:
    return {"answer": "Sorry, I can only answer questions related to academic research.", "sources": []}




def route(state: State) -> str:
    if "academic" in state['category']:
        return "retrieve"
    return "reject"

# Build the graph
graph = StateGraph(State)

# Add nodes
graph.add_node("classify", classify_question)
graph.add_node("retrieve", retrieve_context)
graph.add_node("generate", generate_answer_node)
graph.add_node("reject", reject_off_topic)

# Add edges
graph.set_entry_point("classify")
graph.add_conditional_edges("classify", route, {
    "retrieve": "retrieve",
    "reject": "reject"
})
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
graph.add_edge("reject", END)

# Compile
app = graph.compile()

# Test it
if __name__ == "__main__":
    result = app.invoke({"question": "What is the best pizza place in College Park?"})

    print(f"Answer: {result['answer']}")
    print(f"Sources: {result.get('sources', [])}")
