from pathlib import Path
import os

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


# --- Step 1: Setup ---

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "OPENAI_API_KEY was not found in .env"

docs_dir = (
    Path(__file__).resolve().parent
    / "resources"
    / "groundwork_docs"
)

assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
print(f"Document directory found: {docs_dir}")


# --- Step 2: Load the Documents ---

documents = SimpleDirectoryReader(input_dir=str(docs_dir)).load_data()

print(f"\nLoaded {len(documents)} documents:")

for document in documents:
    file_name = document.metadata.get("file_name", "Unknown file")
    print(f"- {file_name}")


# --- Step 3: Build the Index and Query Engine ---

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

print("\nIndex built successfully. Ready to answer questions.")


def print_source(source, number=None, chunk_length=200):
    """Print a retrieved source node in a consistent format."""
    file_name = source.node.metadata.get("file_name", "Unknown file")
    text = source.node.get_content().replace("\n", " ")

    if number is not None:
        print(f"\nSource {number}:")

    print(f"Document: {file_name}")
    print(f"Similarity Score: {source.score}")
    print(f"Chunk: {text[:chunk_length]}...")


# --- Step 4: Query the Assistant ---

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
        print("\nTop Retrieved Source:")
        print_source(response.source_nodes[0])

# The assistant's answers should be confident and accurate when the requested
# details are directly stated in the Groundwork documents. The retrieval
# results should also show that different questions lead to the matching hours,
# menu, loyalty, company history, or catering document.

# I would still compare each answer with its retrieved chunk because a
# confident tone does not prove that an answer is accurate.


# --- Step 5: Find a Failure ---

failure_question = (
    "What is Groundwork's most popular menu item, and why do customers "
    "prefer it?"
)

failure_response = query_engine.query(failure_question)

print("\n" + "=" * 60)
print(f"Question: {failure_question}")
print(f"Full Response: {failure_response}")
print("\nAll Three Retrieved Source Nodes:")

for number, source in enumerate(
    failure_response.source_nodes[:3],
    start=1,
):
    print_source(source, number=number)

# I asked about Groundwork's most popular menu item because the menu may list
# available products without providing sales or customer preference data. This
# makes the requested conclusion difficult to support from the documents.

# The retriever may find the menu because it contains the words "menu item,"
# but that does not mean the chunk identifies which item is most popular. If
# the model names an item anyway, it is guessing beyond the retrieved
# information.

# The model may still sound confident even when the source does not contain the
# answer. This shows why AI-generated responses should be checked against their
# retrieved evidence instead of being trusted based on tone alone.

# I would improve the system by giving the model instructions to clearly say
# when the documents do not contain enough information. I could also use a
# minimum similarity threshold, retrieve more candidate chunks, and rerank them
# before generating the final response.


# --- Step 6: Reflection ---

# The main LlamaIndex implementation only required a few lines to load the
# documents, create the vector index, and build the query engine. This shows
# that a framework can handle much of the chunking, embedding, storage, and
# retrieval logic that would otherwise need to be written manually.

# A useful business application would be a commercial real estate assistant
# that searches offering memorandums, leases, inspection reports, and operating
# statements. An analyst could ask questions about occupancy, expenses, lease
# terms, or property risks and receive answers connected to the relevant source
# documents.
  