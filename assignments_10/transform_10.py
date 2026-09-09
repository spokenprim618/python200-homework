"""Run the incremental weather ML + LLM enrichment pipeline."""
#https://screenrec.com/share/4bOiy0xNMp
import json
import os

import joblib
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client


MODEL_PATH = "models/weather_classifier.pkl"
METADATA_PATH = "models/weather_classifier_metadata.json"
FALLBACK_SUMMARY = "Recommendation unavailable."

SYSTEM_PROMPT = (
    "You are writing a one-sentence running recommendation for a daily weather "
    "summary app. You will receive weather conditions for one day and a machine "
    "learning prediction about whether the day is good for running. Follow the "
    "classifier's prediction, but use the weather details and confidence to make "
    "the advice useful. Write exactly one sentence that is direct, practical, "
    "and specific to the conditions. Do not use bullet points, headers, or "
    "phrases such as 'Based on the data'."
)

def make_user_message(row, good_for_running, confidence):
    """Create the weather message sent to the LLM for one date."""
    prediction_text = (
        "good for running" if good_for_running else "not ideal for running"
    )
    return (
        f"Date: {row['date']}\n"
        f"High: {row['temperature_2m_max']}°C\n"
        f"Low: {row['temperature_2m_min']}°C\n"
        f"Precipitation: {row['precipitation_sum']} mm\n"
        f"Maximum wind speed: {row['wind_speed_10m_max']} km/h\n"
        f"Model prediction: {prediction_text} (confidence: {confidence:.0%})"
    )


def validate_summary(text):
    """Return a cleaned summary, or None when the API response is unusable."""
    if not text or not text.strip():
        return None

    cleaned = " ".join(text.split())
    if len(cleaned) > 500:
        return None
    return cleaned


def main():
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    supabase = create_client(supabase_url, supabase_key)
    openai_client = OpenAI(api_key=openai_api_key)

    # --- Step 1: Incremental Read ---
    with open(METADATA_PATH, encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    features = metadata["feature_names"]

    raw_rows = supabase.table("weather_raw").select("*").execute().data

    enriched_response = (
        supabase.table("weather_enriched").select("date").execute()
    )
    already_done = {row["date"] for row in enriched_response.data}
    to_classify = [
        row for row in raw_rows if row["date"] not in already_done
    ]

    print("\n--- Incremental Read ---")
    print(f"Raw records: {len(raw_rows)}")
    print(f"Already enriched: {len(already_done)}")
    print(f"Records to process this run: {len(to_classify)}")

    enrichment_records = []

    if to_classify:
        # --- Step 2: ML Transform ---
        classifier = joblib.load(MODEL_PATH)

        weather_df = pd.DataFrame(to_classify)
        missing_features = [
            feature for feature in features if feature not in weather_df.columns
        ]
        if missing_features:
            raise ValueError(
                f"weather_raw is missing model features: {missing_features}"
            )

        feature_data = weather_df[features]
        predictions = classifier.predict(feature_data)
        probabilities = classifier.predict_proba(feature_data)[:, 1]

        enrichment_records = [
            {
                "date": row["date"],
                "good_for_running": bool(predictions[index]),
                "confidence": round(float(probabilities[index]), 4),
            }
            for index, row in enumerate(to_classify)
        ]

        print("\n--- ML Transform ---")
        print(
            f"Good days predicted: {int(predictions.sum())} / "
            f"{len(predictions)}"
        )
        print(
            f"Confidence range: {probabilities.min():.2f} - "
            f"{probabilities.max():.2f}"
        )

        # --- Step 3: LLM Transform ---
        print("\n--- LLM Transform ---")
        for index, record in enumerate(enrichment_records):
            raw_row = to_classify[index]

            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": make_user_message(
                                raw_row,
                                record["good_for_running"],
                                record["confidence"],
                            ),
                        },
                    ],
                    max_tokens=100,
                )
                raw_summary = response.choices[0].message.content
                record["llm_summary"] = (
                    validate_summary(raw_summary) or FALLBACK_SUMMARY
                )
            except Exception as error:
                print(f"API error on {record['date']}: {error}")
                record["llm_summary"] = FALLBACK_SUMMARY

            completed = index + 1
            if completed % 50 == 0 or completed == len(enrichment_records):
                print(
                    f"Enriched {completed} / {len(enrichment_records)} records"
                )

        # --- Step 4: Load ---
        load_response = (
            supabase.table("weather_enriched")
            .upsert(enrichment_records, on_conflict="date")
            .execute()
        )
        print("\n--- Load ---")
        print(f"Rows upserted: {len(load_response.data)}")
    else:
        print("\nAll raw records are already enriched; no transforms were needed.")
        print("Rows upserted: 0")

    # --- Step 5: Verify ---
    verification_response = (
        supabase.table("weather_enriched")
        .select("date,good_for_running,confidence,llm_summary")
        .order("date")
        .execute()
    )
    verified_rows = verification_response.data
    good_day_count = sum(
        bool(row["good_for_running"]) for row in verified_rows
    )

    print("\n--- Verify ---")
    print(f"Total rows in weather_enriched: {len(verified_rows)}")
    print("Five sample rows:")
    for row in verified_rows[:5]:
        print(
            f"{row['date']} | good={row['good_for_running']} | "
            f"confidence={float(row['confidence']):.2f}"
        )
        print(f"  {row['llm_summary']}")
    print(f"Days classified as good for running: {good_day_count}")

    # LLM SUMMARY REVIEW:
    # Most of the summaries accurately reflect the weather features and the ML
    # prediction. The January 5 summary is particularly good because it agrees
    # with the True prediction and high confidence of 0.98. It also supports the
    # recommendation with specific details about the 13.2°C temperature, lack of
    # precipitation, and mild wind speed. The January 4 summary seems weaker
    # because it says there is a "chance of rain," even though the feature gives
    # an amount of precipitation rather than the probability of rain. It also
    # does not explain that the confidence of 0.42 makes the prediction fairly
    # close to the classification threshold. This may have happened because the
    # LLM loosely interpreted the weather values instead of describing the exact
    # information it received.


if __name__ == "__main__":
    main()


# --- Step 6: Reflection ---
# A classifier trained on Charlotte, NC weather may be less accurate when used
# for a different city because the new city's weather can have a different
# distribution of temperatures, precipitation, wind, and seasonal patterns.
# The meaning of comfortable running weather may also change by location, so I
# would test the classifier on labeled data from the new city before trusting it.
# The LLM cannot override the stored classifier result because it only writes the
# llm_summary field; good_for_running and confidence come directly from the ML
# model. Therefore, the LLM is additive: it explains the prediction in natural
# language, even when the classifier's prediction itself may be questionable.
# This makes it important not to treat a convincing summary as proof that the
# underlying prediction is correct. With 50,000 records, my main concerns would
# be API cost, latency, and rate limits because this design makes one LLM request
# per row. I would address them by processing records in controlled batches,
# adding retries and checkpoints, monitoring token usage, and using asynchronous
# requests where allowed. I could also use a cheaper model, shorten the prompt,
# cache repeated results, or generate rule-based summaries when an LLM is not
# necessary.
