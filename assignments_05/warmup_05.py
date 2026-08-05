from dotenv import load_dotenv
from openai import OpenAI
import json

# --- API ---

# API Q1

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print(f"Response: {response.choices[0].message.content}")

print(f"Model name: {response.model}")

print(f"Usage of tokens: {repsonse.usage.total_tokens}")

# API Q2

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temp
    )

    print(f"\nTemperature: {temp}")
    print(response.choices[0].message.content)

# I would assume the answer would change because the more access to probable tokens based on temperture. If you want predictable answers use a lower temperature.

# API Q3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."
        }
    ],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response.choices, start=1):
    print(f"\nCompletion {i}:")
    print(choice.message.content)


# API Q4

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain how neural networks work."
        }
    ],
    max_tokens=15
)

print(response.choices[0].message.content)

# It could reduce the lenght of an answer to just what is needed and not have the model go on for longer costing more

# --- API ---


# System Messages and Personas

# System Messages and Personas Q1


from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# Personality 1: Tutor
messages = [
    {
        "role": "system",
        "content": (
            "You are a patient, encouraging Python tutor. "
            "You always explain things simply and end with a word of encouragement."
        )
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("=== Tutor Personality ===")
print(response.choices[0].message.content)

# Personality 2: Pirate
messages = [
    {
        "role": "system",
        "content": (
            "You are stern like an army sergent"
            "using army patriotic language."
        )
    },
    {
        "role": "user",
        "content": "I don't understand what a list comprehension is."
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\n=== Pirate Personality ===")
print(response.choices[0].message.content)

# I don't have the model to see

# System Messages and Personas Q2


from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "My name is Jordan and I'm learning Python."
    },
    {
        "role": "assistant",
        "content": (
            "Nice to meet you, Jordan! Python is a great choice. "
            "What would you like to work on?"
        )
    },
    {
        "role": "user",
        "content": "Can you remind me what my name is?"
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print(response.choices[0].message.content)

# It is due to the model having access to the message history and who answered

# Prompt engineering

# Prompt engineering Q1

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Classify each review as positive, negative, or mixed.

Example:

Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now classify these reviews:

Review 1: "{reviews[0]}"
Review 2: "{reviews[1]}"
Review 3: "{reviews[2]}"
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)

# Prompt engineering Q2

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Classify each review as positive, negative, or mixed.

Example:

Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now classify these reviews:

Review 1: "{reviews[0]}"
Review 2: "{reviews[1]}"
Review 3: "{reviews[2]}"
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)

# I assume the model would take on the information or structure of the example

# Prompt engineering Q3

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Classify reviews as positive, negative, or mixed.

Examples:

Review: "The product exceeded all expectations."
Sentiment: positive

Review: "Customer service never replied to my emails."
Sentiment: negative

Review: "The features are excellent, but setup was frustrating."
Sentiment: mixed

Now classify these reviews:

Review 1: "{reviews[0]}"

Review 2: "{reviews[1]}"

Review 3: "{reviews[2]}"
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)

# I would choose zero shot on answer tasks, one shot if there is need for a small guide,
# and few shot if there is a need for consistency for a certain structure of response

# Prompt Engineering Q4

prompt = """
Solve the following problem.

Show your reasoning step by step before giving the final answer.

A data engineer earns $85,000 per year.
She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year
than her post-raise salary.

What is her final annual salary?

Label the final answer clearly.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)

# It is able to go through every step without skipping anything and it adds a form of checks for itself along with visibility in it's steps

# Prompt Engineering Q5

review = (
    "I've been using this tool for three months. "
    "It handles large datasets well, "
    "but the UI is clunky and the export options are limited."
)

prompt = f"""
Analyze the following review.

Return ONLY valid JSON with these keys:

sentiment
confidence
reason

Review:
"{review}"
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

raw_response = response.choices[0].message.content

print("Raw Response:")
print(raw_response)

try:
    result = json.loads(raw_response)

    print("\nSentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])

except json.JSONDecodeError:
    print("\nInvalid JSON received.")
    print(raw_response)


# Prompt Engineering Q6

user_text = (
    "First boil a pot of water. Once boiling, add a handful of salt "
    "and the pasta. Cook for 8-10 minutes until al dente. "
    "Drain and toss with your sauce of choice."
)

prompt = f"""
You will be given text inside triple backticks.

If it contains step-by-step instructions,
rewrite them as a numbered list.

If it does not contain instructions,
respond with exactly:

"No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)


non_instruction = (
    "The weather was beautiful today. "
    "Many people visited the park to enjoy the sunshine."
)

prompt = f"""
You will be given text inside triple backticks.

If it contains step-by-step instructions,
rewrite them as a numbered list.

If it does not contain instructions,
respond with exactly:

"No steps provided."

```{non_instruction}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)

# It keeps a seperation so the model knows what is the item to follow along
# and what is the item that needs some work or consideration on

# Olama

# Olama 1

# I did not have enough space to download olama but here is openAI

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain what a large language model is in two sentences."
        }
    ]
)

print(response.choices[0].message.content)
