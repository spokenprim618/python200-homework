# I dont have the OpenAI key at the moment so deper analysis of AI input won't be available yet
from dotenv import load_dotenv
import os
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)

# --- RAG Concepts ---

# Concepts 1

# Scenario A: The signficant detail is they want their internal library of pdf's accessible. This makes me believe that RAG.

# Scenario B: Due to the style not being much online but they have many examples in house. It seems fine tuning would be needed and these examples could be used to train the model for their special needs.

# Scenario C: Due to the data analyst only needing this reponse in the short term. I assume prompt engineering would be needed to ask the AI.

# Concepts 2

# It is way worse because it could stear a person the wrong way who trusts only the AI's response. The way the model says something now makes us not trust as much due to how it wants to convince instead of inform.


# Concepts 3

steps = [
    "Receive the user's query",
    "Embed the user's query",
    "Extract text from source documents",
    "Split text into chunks",
    "Retrieve the most relevant chunks",
    "Convert text chunks into embeddings",
    "Inject retrieved chunks into the prompt",
    "Generate a response from the LLM",
]

import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


# --- Keyword RAG ---

# Keyword 1

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

simple_keyword_retrieval(query,documents)

# Don't know till key recieved.

# Keyword 2

query = "Do you have anything without caffeine?"

simple_keyword_retrieval(query,documents)

# Don't know till key recieved. But I'd assume menu.txt would be given and no caffine is the keyword

# Keyword 3

query = "How do I sign up for rewards?"

simple_keyword_retrieval(query,documents)

# I'd assume loyalty.txt would be chosen but I won't know.

# --- Semantic RAG Concepts ---

# Semantic Q1

# A vector embedding is a numerical representation of for example a word in multiple dimensions. It is used to find other similiar embeddings in the same space or close.
# 0.85 is the most relevant because it is the closest in relationship.
# It is due to methods such as cosine similiarity.

# Semantic Q2

# | Feature                    | Keyword RAG                       | Semantic RAG |
# |----------------------------|-----------------------------------|--------------|
# | What is compared?          | Exact word overlap                | Vector closeness            |
# | What is retrieved?         | Full document                     | Vectors            |
# | Can it handle synonyms?    | No                                | Yes      |
# | Storage format             | Plain text dictionary             | Vector storage            |
# | Relevance score            | Number of overlapping keywords    | cosine similarity            |

# --- LlamaIndex ---

# LlamaInde Q1


documents = SimpleDirectoryReader("../../asingments_06/resources/brightleaf_pdf")

print(f"Loaded {len(documents)} documents.")



index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=3)


# LlamaIndex Q1


questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]


for question in questions:

    print(f"\nQuestion: {question}")

    response = query_engine.query(question)

    print(f"\nAnswer:\n{response}")

    print("\nRetrieved Source Nodes:")

    for i, node in enumerate(response.source_nodes, start=1):
        print(f"\nSource Node {i}")
        print(f"Similarity Score: {node.score}")
        print(f"Chunk: {node.node.get_content()[:150]}")




# LlamaIndex Q2

question = "What employee benefits does BrightLeaf offer?"


# similarity_top_k = 1
query_engine_k1 = index.as_query_engine(
    similarity_top_k=1
)

response_k1 = query_engine_k1.query(question)



print(f"\nQuestion: {question}")
print(f"\nResponse:\n{response_k1}")

print("\nSource Nodes:")

for i, node in enumerate(response_k1.source_nodes, start=1):
    print(f"\nSource Node {i}")
    print(f"Similarity Score: {node.score}")
    print(f"Chunk: {node.node.get_content()[:150]}")


# similarity_top_k = 5
query_engine_k5 = index.as_query_engine(
    similarity_top_k=5
)

response_k5 = query_engine_k5.query(question)

print(f"\nQuestion: {question}")
print(f"\nResponse:\n{response_k5}")

print("\nSource Nodes:")

for i, node in enumerate(response_k5.source_nodes, start=1):
    print(f"\nSource Node {i}")
    print(f"Similarity Score: {node.score}")
    print(f"Chunk: {node.node.get_content()[:150]}")



# LlamaIndex Question 3

q3 = "What will BrightLeaf's earnings report look like in the next couple of years due to their product and security?"

query_engine_q3 = index.as_query_engine(similarity_top_k=3)

response_q3 = query_engine_q3.query(q3)

print(f"\nQuestion: {q3}")

print(f"\nResponse:\n{response_q3}")

print("\nRetrieved Source Nodes:")

for i, node in enumerate(response_q3.source_nodes, start=1):
    print(f"\nSource Node {i}")
    print(f"Similarity Score: {node.score}")
    print(f"Chunk: {node.node.get_content()}")





# LlamaIndex Q4

q4 = "What employee benefits does BrightLeaf offer?"

# Use gpt-4o-mini as the judge LLM
judge_llm = OpenAI(model="gpt-4o-mini")

faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)

relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)


# Get response for the good query
response_q4 = query_engine.query(q4)


faithfulness_result = faithfulness_evaluator.evaluate_response(response=response_q4)

relevancy_result = relevancy_evaluator.evaluate_response(query=q4,response=response_q4,)

print(f"\nQuestion: {q4}")
print(f"\nResponse:\n{response_q4}")

print("\nEvaluation:")
print(f"Faithfulness Score: {faithfulness_result.score}")
print(f"Relevancy Score: {relevancy_result.score}")

bad_query = "What is BrightLeaf's theme music?"

bad_response = query_engine.query(bad_query)

bad_faithfulness_result = faithfulness_evaluator.evaluate_response(response=bad_response)

bad_relevancy_result = relevancy_evaluator.evaluate_response(query=bad_query,response=bad_response,)

print(f"\nQuestion: {bad_query}")
print(f"\nResponse:\n{bad_response}")

print("\nEvaluation:")
print(f"Faithfulness Score: {bad_faithfulness_result.score}")

print(f"Relevancy Score: {bad_relevancy_result.score}")