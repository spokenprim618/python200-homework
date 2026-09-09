"""Warmup 10: choosing ML, LLM, and deterministic pipeline tools."""

import time


# --- ML vs. LLM in Pipelines ---

# ML/LLM Question 1
# The ML model classifies each day as either good for running or a day to skip,
# and it provides a confidence score for that prediction. It does this well
# because it was trained on labeled weather data and learned how the numeric
# features relate to the two classes. The LLM then uses the weather information
# and the ML prediction to write a recommendation that is easier for a person to
# understand. If I used the LLM for classification, it would be slower, cost
# more, and might not give the same answer every time. If I used the ML model to
# write the recommendation, it would not work because the model was only trained
# to return a class and a probability, not to understand and write natural
# language.


# ML/LLM Question 2
# 1. I would use deterministic code for the date because a date library can
#    calculate the correct day of the week every time.
# 2. I would use an LLM for the job posting because it needs to read freeform
#    text and understand what experience level the posting is describing.
# 3. I would use a trained ML model for customer churn because I have labeled
#    data and a fixed group of numeric features to make the prediction from.
# 4. I would use an LLM for the city names because it can recognize that
#    different versions of a city name can still refer to the same place.
# 5. I would use deterministic code to add the revenue because there is one
#    correct answer and there is no reason to use an ML model or an LLM for it.


# ML/LLM Question 3
# Incremental processing means that I only process records that have not already
# been completed. This is important because there is no reason to run the model
# and LLM again on dates that are already in weather_enriched. If I processed all
# 365 records every time, I would keep paying for the same LLM calls and the
# pipeline would take longer to finish. The LLM could also create different
# wording for records that were already completed. If the database were not set
# up to prevent duplicates, I could also end up with repeated dates and
# incorrect counts. Incremental processing keeps the pipeline cheaper, faster,
# and more consistent.


# --- Prompt Design ---

# Prompt Question 1
TWO_SENTENCE_SYSTEM_PROMPT = (
    "You are writing a running recommendation for a daily weather summary app. "
    "You will receive one day's weather features and a machine learning "
    "prediction about whether the day is good for running. Write exactly two "
    "sentences. In the first sentence, clearly state whether the prediction is "
    "good for running or not ideal for running. In the second sentence, explain "
    "the prediction using specific temperature, precipitation, wind, and "
    "confidence details provided in the message. Do not use bullet points, "
    "headers, or extra commentary."
)

# I would need to change the validation so it expects exactly two sentences
# instead of one. It should reject the response if it finds fewer or more
# than two sentences or if either sentence is empty. I would also want to use a
# better sentence parser instead of only splitting on periods because a decimal
# or abbreviation could be mistaken for the end of a sentence.


# Prompt Question 2
def call_with_retry(client, messages, max_retries=3):
    """Call the chat completion API and retry after temporary failures."""
    retries = 0

    while True:
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=150,
            )
        except Exception as error:
            if retries >= max_retries:
                print(f"OpenAI request failed after final attempt: {error}")
                return None

            retries += 1
            print(
                f"OpenAI request failed; retrying "
                f"({retries}/{max_retries}): {error}"
            )
            time.sleep(2)


# I would use this in a production pipeline so one temporary API error does not
# cause the entire pipeline to stop. After a timeout, rate limit, or brief
# network issue, the request may succeed when it is tried again. The retry limit
# is also important because I do not want the pipeline to get stuck on one
# record forever. For a larger pipeline, I would also log the failed records and
# save the progress so I could try those records again later.
