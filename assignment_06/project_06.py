# I dont have the OpenAI key at the moment so deper analysis of AI input won't be available yet

from pathlib import Path
import os

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


#--- Step 1: Setup ---

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

assert api_key, "OPENAI_API_KEY was not found in .env"
print("API key loaded successfully.")

docs_dir = Path("assignments_06/resources/groundwork_docs")

assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
print(f"Document directory found: {docs_dir}")


#--- Step 2: Load the Documents ---

documents = SimpleDirectoryReader(input_dir=str(docs_dir)).load_data()

print(f"\nLoaded {len(documents)} documents:")

for document in documents:
    file_name = document.metadata.get("file_name", "Unknown file")
    print(f"- {file_name}")


#--- Step 3: Build the Index and Query Engine ---

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=3)

print("\nIndex built successfully. Ready to answer questions.")


#--- Step 4: Query the Assistant ---

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    response = query_engine.query(question)

    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print(f"Answer: {response}")

    if response.source_nodes:
        source = response.source_nodes[0]

        file_name = source.node.metadata.get("file_name", "Unknown file")
        score = source.score
        text = source.node.get_content()

        print("\nTop Retrieved Source:")
        print(f"Document: {file_name}")
        print(f"Similarity Score: {score}")
        print(f"Chunk: {text[:200]}...")


#--- Step 5: Find a Failure ---

failure_question = ("Are there any frequently asked questions about their menu and what is the most popular menu item?")

failure_response = query_engine.query(failure_question)

print(f"\nQuestion: {failure_question}")
print(f"\nFull Response:\n{failure_response}")

print("\nAll Retrieved Source Nodes:")

for i, source in enumerate(failure_response.source_nodes, start=1):
    file_name = source.node.metadata.get("file_name", "Unknown file")
    score = source.score
    text = source.node.get_content()

    print(f"\nSource {i}:")
    print(f"Document: {file_name}")
    print(f"Similarity Score: {score}")
    print(f"Chunk: {text[:200]}...")