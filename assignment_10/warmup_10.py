import time


# --- ML vs. LLM in Pipelines ---

# ML/LLM Question 1
#


# ML/LLM Question 2
#
# 1. I would use deterministic code to convert "2023-07-04" to a day of the
#    week because calendar conversion follows an exact rule.
# 2. I would use an LLM to classify job seniority from freeform text because it
#    can interpret varied titles, requirements, and wording without fixed rules.
# 3. I would use a trained ML model to predict churn because labeled examples
#    and numeric features allow it to learn and evaluate a repeatable pattern.
# 4. I would use deterministic code with a maintained alias-to-canonical-city
#    lookup because known city-name variants should always produce the same name.
# 5. I would use deterministic code to sum revenue because addition is exact and
#    does not require prediction or language understanding.


# ML/LLM Question 3
#
# Incremental processing means processing only raw records that do not already
# have a corresponding enriched record. It is important here because it avoids
# repeating the same classifier work, LLM calls, and database writes each time
# the pipeline runs. If all 365 records were reprocessed every time, repeated LLM
# calls would unnecessarily increase API cost and runtime. The LLM could also
# return slightly different wording on each run, causing already-correct rows to
# change even though their source data did not. Incremental processing therefore
# lowers cost while making the stored results more stable.


# --- Prompt Design ---

# Prompt Question 1

TWO_SENTENCE_SYSTEM_PROMPT = """
You are a running-weather assistant. Write exactly two sentences. In the first
sentence, clearly state the supplied ML prediction: either that the day is good
for running or that the runner should skip it. In the second sentence, explain
the prediction using the supplied temperature, precipitation, wind, and model
confidence. Do not contradict or override the ML prediction.
""".strip()

# The validation logic would need to require exactly two complete sentences
# instead of one. It should also verify that sentence one states the supplied
# prediction and sentence two gives weather-based reasoning; an invalid response
# could then be retried or replaced with a fallback.


# Prompt Question 2

def call_with_retry(client, messages, max_retries=3):
    """Call the chat completion API, returning None after the final failure."""
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1.")

    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)

    return None


# I would use retry logic in production for temporary failures such as rate
# limits, network interruptions, or short API outages. The retry count prevents
# one temporary problem from losing a record while also preventing the pipeline
# from retrying forever during a longer outage.
