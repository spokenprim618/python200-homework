from dotenv import load_dotenv
from openai import OpenAI
import json


load_dotenv()
client = OpenAI()


# --- Chat Completions API ---

# API Q1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?",
        }
    ],
)

print("API Q1 Response:", response.choices[0].message.content)
print("API Q1 Model name:", response.model)
print("API Q1 Total tokens used:", response.usage.total_tokens)


# API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temperature in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    print(f"\nAPI Q2 - Temperature {temperature}:")
    print(response.choices[0].message.content)

# Temperature 0 produced the most direct and predictable response. Higher
# temperatures produced more variety and not on task ideas. I would use temperature
# 0 when I need the most consistent and reproducible output.


# API Q3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library).",
        }
    ],
    n=3,
    temperature=1.0,
)

for number, choice in enumerate(response.choices, start=1):
    print(f"\nAPI Q3 - Completion {number}:")
    print(choice.message.content)


# API Q4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15,
)

print("\nAPI Q4 Response:")
print(response.choices[0].message.content)

# The response stopped before completing the explanation because it reached the
# 15-token limit. In a real application, max_tokens can control cost, latency,
# and the maximum length of a response.


# --- System Messages and Personas ---

# System Q1 - Personality 1: Patient tutor
tutor_messages = [
    {
        "role": "system",
        "content": (
            "You are a patient, encouraging Python tutor. Explain ideas simply, "
            "use a small example, and end with a word of encouragement."
        ),
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is.",
    },
]

tutor_response = client.chat.completions.create(
    model="gpt-4o-mini", messages=tutor_messages
)
print("\nSystem Q1 - Patient Tutor Personality:")
print(tutor_response.choices[0].message.content)

# System Q1 - Personality 2: Impatient pirate captain
pirate_messages = [
    {
        "role": "system",
        "content": (
            "You are an impatient pirate captain teaching Python to your crew. "
            "Use pirate vocabulary, short commands, and a dramatic nautical "
            "analogy. Do not use the gentle tone of a tutor."
        ),
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is.",
    },
]

pirate_response = client.chat.completions.create(
    model="gpt-4o-mini", messages=pirate_messages
)
print("\nSystem Q1 - Pirate Captain Personality:")
print(pirate_response.choices[0].message.content)

# The patient tutor should use a calm explanation and encouragement. The pirate
# captain should use less proper vocabulary, brief commands, and a forceful tone.
# Both explain the same concept, but their personalities are completely different.


# System Q2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {
        "role": "assistant",
        "content": (
            "Nice to meet you, Jordan! Python is a great choice. "
            "What would you like to work on?"
        ),
    },
    {"role": "user", "content": "Can you remind me what my name is?"},
]

response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print("\nSystem Q2 Response:")
print(response.choices[0].message.content)

# The model knows Jordan's name because the conversation history was included in
# this request. The API is still stateless; the context was sent again manually.


# --- Prompt Engineering ---

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow.",
]


# Prompt Q1 - Zero-shot: instructions and reviews, with no examples
zero_shot_prompt = f"""
Classify each review as positive, negative, or mixed.
Return one line per review in the format: Review number: sentiment.

Review 1: "{reviews[0]}"
Review 2: "{reviews[1]}"
Review 3: "{reviews[2]}"
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": zero_shot_prompt}],
)
print("\nPrompt Q1 - Zero-Shot Results:")
print(response.choices[0].message.content)


# Prompt Q2 - One-shot: one example demonstrates the expected classification
one_shot_prompt = f"""
Classify each review as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now classify the following reviews. Label each result with its review number.

Review 1: "{reviews[0]}"
Review 2: "{reviews[1]}"
Review 3: "{reviews[2]}"
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": one_shot_prompt}],
)
print("\nPrompt Q2 - One-Shot Results:")
print(response.choices[0].message.content)

# The example makes the desired label and output pattern more explicit. Even if
# this run looks similar to zero-shot, one-shot prompting can improve consistency.


# Prompt Q3 - Few-shot: examples demonstrate all three possible classes
few_shot_prompt = f"""
Classify each review as positive, negative, or mixed.

Examples:
Review: "The product exceeded all expectations."
Sentiment: positive

Review: "Customer service never replied to my emails."
Sentiment: negative

Review: "The features are excellent, but setup was frustrating."
Sentiment: mixed

Now classify the following reviews. Label each result with its review number.

Review 1: "{reviews[0]}"
Review 2: "{reviews[1]}"
Review 3: "{reviews[2]}"
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": few_shot_prompt}],
)
print("\nPrompt Q3 - Few-Shot Results:")
print(response.choices[0].message.content)

# I would use zero-shot for a simple task the model already understands, one-shot
# when one example can clarify the expected format, and few-shot when the model
# needs examples of several classes or greater consistency.


# Prompt Q4 - Chain of thought
reasoning_prompt = """
Solve the problem below. Show your calculations step by step, then label the
final answer clearly.

A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": reasoning_prompt}],
)
print("\nPrompt Q4 - Step-by-Step Response:")
print(response.choices[0].message.content)

# Breaking a multi-step problem into smaller calculations makes skipped steps
# less likely and makes the final answer easier for the user to verify.


# Prompt Q5 - Structured output
review = (
    "I've been using this tool for three months. It handles large datasets well, "
    "but the UI is clunky and the export options are limited."
)

structured_prompt = f"""
Analyze the review below. Return only valid JSON with these keys:
"sentiment", "confidence", and "reason". Confidence must be a float from 0 to 1,
and reason must be one sentence.

Review: "{review}"
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": structured_prompt}],
)
raw_response = response.choices[0].message.content
print("\nPrompt Q5 - Raw Response:")
print(raw_response)

try:
    result = json.loads(raw_response)
    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])
except (json.JSONDecodeError, KeyError, TypeError) as error:
    print("The response was not valid JSON in the required structure:", error)
    print("Raw response for debugging:", raw_response)


# Prompt Q6 - Delimiters
user_text = (
    "First boil a pot of water. Once boiling, add a handful of salt and the "
    "pasta. Cook for 8-10 minutes until al dente. Drain and toss with your "
    "sauce of choice."
)

delimiter_prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": delimiter_prompt}],
)
print("\nPrompt Q6 - Instructional Text:")
print(response.choices[0].message.content)

non_instruction_text = (
    "The weather was beautiful today. Many people visited the park to enjoy "
    "the sunshine."
)

non_instruction_prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{non_instruction_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": non_instruction_prompt}],
)
print("\nPrompt Q6 - Non-Instructional Text:")
print(response.choices[0].message.content)

# Delimiters separate the data being analyzed from the instructions. This reduces
# ambiguity and helps prevent user text from being mistaken for part of the prompt.


# --- Local Models with Ollama ---
# --- Local Models with Ollama ---

# Ollama Q1

# I ran this command in the terminal:
# ollama run qwen3:0.6b "Explain what a large language model is in two sentences."

ollama_output = """
A large language model (LLM) is a type of artificial intelligence trained on
massive amounts of text data to understand and generate human-like language.
It works by predicting the most likely next word in a sequence, allowing it to
perform tasks like answering questions, summarizing text, and writing code.
"""

print("\nOllama Q1 - Saved Ollama Output:")
print(ollama_output.strip())

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain what a large language model is in two sentences.",
        }
    ],
)

print("\nOllama Q1 - OpenAI Response:")
print(response.choices[0].message.content)

# Both models explained that an LLM learns patterns from large amounts of text to
# understand and generate language. The Ollama response was simpler and explained
# next-word prediction, while the OpenAI response was more technical and mentioned
# deep learning and neural networks. One advantage of using a local model is greater
# privacy because the prompt can remain on the user's computer. One disadvantage is
# that it requires local storage, memory, computing power, and additional setup.