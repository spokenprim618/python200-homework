from dotenv import load_dotenv
import string

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI as LlamaOpenAI


if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# --- RAG Concepts ---

# Concepts Question 1

# Scenario A: RAG would be the best approach because the legal team needs
# responses grounded in hundreds of internal PDFs. RAG also allows the
# documents to be updated without retraining the model every quarter.

# Scenario B: Fine-tuning would be the best approach because the company wants
# the model to consistently reproduce a specialized writing style. The 3,000
# examples give the model material from which it can learn that style.

# Scenario C: Prompt engineering, specifically context injection, would be the
# simplest approach. The analyst can place the short report directly in the
# prompt because the system only needs to work with this one document.


# Concepts Question 2

# A confidently wrong answer is more harmful because its certain tone can make
# a person trust and act on false information. For example, an AI could
# confidently provide an incorrect medication dosage, causing someone to
# follow dangerous medical guidance instead of checking with a professional.
# An answer that says "I am not sure" signals uncertainty and encourages the
# user to verify the information before relying on it.


# Concepts Question 3

# 1. Extract text from source documents. This reads usable text from PDFs or
# other source files.
# 2. Split text into chunks. This divides long documents into smaller sections
# that can be searched.
# 3. Convert text chunks into embeddings. This represents each chunk as a
# numerical vector that captures its meaning.
# 4. Receive the user's query. This accepts the question the user wants the
# system to answer.
# 5. Embed the user's query. This converts the question into a vector using the
# same embedding model used for the document chunks.
# 6. Retrieve the most relevant chunks. This compares the query embedding with
# the stored chunk embeddings.
# 7. Inject retrieved chunks into the prompt. This gives the most relevant
# source text to the language model as context.
# 8. Generate a response from the LLM. This uses the retrieved context to
# produce a grounded answer.

steps = [
    "Extract text from source documents",
    "Split text into chunks",
    "Convert text chunks into embeddings",
    "Receive the user's query",
    "Embed the user's query",
    "Retrieve the most relevant chunks",
    "Inject retrieved chunks into the prompt",
    "Generate a response from the LLM",
]


# --- Keyword RAG ---

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        word.translate(translator)
        for word in query.lower().split()
        if word not in stopwords
    }

    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []

    for name, content in documents.items():
        content_words = {
            word.translate(translator)
            for word in content.lower().split()
            if word not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))

        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(
        (
            (name, content)
            for score, name, content in scores
            if score > 0
        ),
        None,
    )

    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]

    if verbose:
        print("\nNo overlapping keywords found.")

    return [("None found", "No relevant content.")]


documents = {
    "menu.txt": (
        "We serve espresso, lattes, cappuccinos, and cold brew. "
        "Pastries include croissants and muffins baked fresh daily. "
        "Oat milk and almond milk are available."
    ),
    "hours.txt": (
        "We are open Monday through Friday from 7am to 7pm. "
        "On weekends we open at 8am and close at 5pm. "
        "We are closed on Thanksgiving and Christmas Day."
    ),
    "hiring.txt": (
        "We are currently hiring baristas and shift supervisors. "
        "Send your resume to jobs@groundworkcoffee.com."
    ),
    "loyalty.txt": (
        "Join our loyalty program to earn one point per dollar spent. "
        "Redeem 100 points for a free drink of your choice."
    ),
}


# Keyword Question 1

query = "What are your hours on weekends?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"Selected document: {result[0][0]}")

# The function selected hours.txt because the words "hours" and "weekends"
# appear in both the query and that document. This is a successful example of
# keyword retrieval because the query and document use the same wording.


# Keyword Question 2

query = "Do you have anything without caffeine?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"Selected document: {result[0][0]}")

# The function returned "None found" because none of the important words in
# the query appear exactly in menu.txt. Although the menu lists drinks, it does
# not use words such as "without caffeine" or "decaffeinated." Semantic
# retrieval would perform better because it can compare meaning instead of
# depending only on exact keyword overlap.


# Keyword Question 3

# I predict that loyalty.txt should be selected because "rewards" has a similar
# meaning to a loyalty program. However, keyword retrieval may fail because
# "sign up" and "rewards" do not appear exactly in the document.

query = "How do I sign up for rewards?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"Selected document: {result[0][0]}")

# My prediction was not correct because the function returned "None found."
# Loyalty.txt is conceptually relevant, but keyword retrieval does not
# understand that "rewards" and "loyalty program" are related ideas.


# --- Semantic RAG Concepts ---

# Semantic Question 1

# A vector embedding is a numerical representation of the meaning of text.
# Text with similar meaning should have embeddings located near each other in
# the vector space.

# The chunk with a cosine similarity score of 0.85 is more relevant. Its
# embedding has a stronger relationship with the query embedding than the
# chunk with a score of 0.30.

# Semantic search compares meanings represented by embeddings rather than
# exact words. It can recognize that phrases such as "rewards program" and
# "loyalty points" are related even though they use different words.


# Semantic Question 2

# Keyword RAG compares exact word overlap, while semantic RAG compares the
# meaning represented by vector embeddings.

# Keyword RAG retrieves a full document in this example, while semantic RAG
# can retrieve the most relevant chunks from within documents.

# Keyword RAG normally cannot handle synonyms unless the matching logic is
# expanded. Semantic RAG can handle synonyms because similar meanings should
# produce similar embeddings.

# Keyword RAG can store its documents in a plain-text dictionary. Semantic RAG
# stores embeddings in a vector index or vector database.

# Keyword RAG uses the number of overlapping keywords as its relevance score.
# Semantic RAG uses a vector similarity measurement such as cosine similarity.


# --- LlamaIndex ---

brightleaf_dir = "lessons/05_AI_intro/resources/brightleaf_dir"

brightleaf_documents = SimpleDirectoryReader(
    input_dir=brightleaf_dir
).load_data()

print(f"\nLoaded {len(brightleaf_documents)} BrightLeaf documents.")

index = VectorStoreIndex.from_documents(brightleaf_documents)
query_engine = index.as_query_engine(similarity_top_k=3)


def print_source_nodes(response, limit=None, chunk_length=150):
    """Print retrieved source-node information."""
    source_nodes = response.source_nodes

    if limit is not None:
        source_nodes = source_nodes[:limit]

    for number, source in enumerate(source_nodes, start=1):
        text = source.node.get_content().replace("\n", " ")
        print(f"\nSource Node {number}")
        print(f"Similarity Score: {source.score}")
        print(f"Chunk: {text[:chunk_length]}")


# LlamaIndex Question 1

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for question in questions:
    response = query_engine.query(question)

    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print(f"Answer: {response}")
    print("\nTop 3 Retrieved Source Nodes:")
    print_source_nodes(response, limit=3, chunk_length=150)

# The chunks retrieved for the benefits question should come from information
# about employee benefits, making them relevant to the question. The answer
# should sound specific when those details are clearly stated in the source.

# The security query should retrieve chunks describing BrightLeaf's security
# rules or procedures. If unrelated company information appears, it may mean
# the question was too broad or some chunks contained overlapping company terms
# without actually answering the question.

# These observations should be compared with the printed output because
# retrieval results can vary slightly each time the index is built.


# LlamaIndex Question 2

question = "What employee benefits does BrightLeaf offer?"

for top_k in [1, 5]:
    comparison_engine = index.as_query_engine(similarity_top_k=top_k)
    response = comparison_engine.query(question)

    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print(f"similarity_top_k: {top_k}")
    print(f"Answer: {response}")
    print("\nRetrieved Source Nodes:")
    print_source_nodes(response, limit=top_k, chunk_length=150)

# Using top_k=1 gives the model less context and makes the answer depend on one
# chunk. Using top_k=5 may provide more details, but it may also retrieve weaker
# or unrelated chunks. More context is not always better because irrelevant
# information can distract the model from the strongest source.


# LlamaIndex Question 3

q3 = (
    "What will BrightLeaf's earnings report look like over the next "
    "two years because of its products and security policies?"
)

response_q3 = query_engine.query(q3)

print("\n" + "=" * 60)
print(f"Question: {q3}")
print(f"Answer: {response_q3}")
print("\nAll Retrieved Source Nodes:")
print_source_nodes(response_q3, chunk_length=150)

# I expected this question to be difficult because it asks for a future
# financial prediction that may not be contained in the documents. The
# retriever can find information about products or security, but those chunks
# may not support a forecast of future earnings.

# If the model still produces a financial prediction, it is going beyond the
# retrieved evidence. I would improve the system by adding instructions to say
# when the documents do not contain enough information and by using a minimum
# similarity threshold before generating an answer.


# LlamaIndex Question 4

judge_llm = LlamaOpenAI(model="gpt-4o-mini")
faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)


def evaluate_query(question):
    """Query the index and evaluate its response."""
    response = query_engine.query(question)

    faithfulness = faithfulness_evaluator.evaluate_response(
        response=response
    )
    relevancy = relevancy_evaluator.evaluate_response(
        query=question,
        response=response,
    )

    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print(f"Answer: {response}")
    print(f"Faithfulness Score: {faithfulness.score}")
    print(f"Relevancy Score: {relevancy.score}")

    return response, faithfulness, relevancy


main_query = "What employee benefits does BrightLeaf offer?"
low_quality_query = "What is BrightLeaf's theme music?"

evaluate_query(main_query)
evaluate_query(low_quality_query)

# A faithfulness score of 1.0 means the response's claims are supported by the
# retrieved context. A score of 0.0 means the response contains claims that are
# not supported by that context.

# Relevancy measures whether the response and retrieved context address the
# user's actual question. It differs from faithfulness because an answer can be
# supported by a document while still being unrelated to what was asked.

# The unsupported query may receive different scores because the documents do
# not contain information about theme music. However, if the model correctly
# says the information is unavailable, it may still receive a good
# faithfulness score because it did not invent an answer.

# LLM-as-a-judge means using another language model to evaluate qualities such
# as support and relevance. It is used because RAG answers are open-ended and
# may have several correct wordings, so an exact-match accuracy calculation
# would not evaluate them fairly.
