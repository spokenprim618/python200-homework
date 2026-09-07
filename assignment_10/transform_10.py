import json
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "weather_classifier.pkl"
METADATA_PATH = BASE_DIR / "models" / "weather_classifier_metadata.json"
LLM_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """
You are a concise running-weather assistant. Write exactly one sentence that
states whether the person should run based on the supplied ML prediction and
briefly supports that prediction with the supplied weather measurements. Treat
the ML prediction as final, do not override it, and do not invent information.
""".strip()


# --- Connections and Files ---

def get_supabase_client():
    load_dotenv(BASE_DIR / ".env")
    load_dotenv()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url:
        raise ValueError("SUPABASE_URL is missing from the environment.")
    if not key:
        raise ValueError("SUPABASE_KEY is missing from the environment.")

    return create_client(url, key)


def get_openai_client():
    load_dotenv(BASE_DIR / ".env")
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is missing; LLM fallbacks will be used.")
        return None

    try:
        return OpenAI()
    except Exception as error:
        print(f"Warning: OpenAI client could not be created: {error}")
        return None


def load_model_files():
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    model = joblib.load(MODEL_PATH)
    return model, metadata


def fetch_all_rows(supabase, table_name, columns="*", page_size=1000):
    """Fetch every row even when the database API applies a page limit."""
    rows = []
    start = 0

    while True:
        response = (
            supabase
            .table(table_name)
            .select(columns)
            .range(start, start + page_size - 1)
            .execute()
        )
        page = response.data or []
        rows.extend(page)

        if len(page) < page_size:
            break
        start += page_size

    return rows


# --- Step 1: Incremental Read ---

def get_unprocessed_records(supabase):
    raw_records = fetch_all_rows(supabase, "weather_raw")
    enriched_rows = fetch_all_rows(supabase, "weather_enriched", "date")
    enriched_dates = {row["date"] for row in enriched_rows}

    unprocessed = [
        row for row in raw_records
        if row["date"] not in enriched_dates
    ]

    print("--- Incremental Read ---")
    print(f"Raw records: {len(raw_records)}")
    print(f"Already enriched: {len(enriched_dates)}")
    print(f"Processing this run: {len(unprocessed)}")

    return unprocessed


# --- Step 2: ML Transform ---

def run_ml_transform(model, metadata, records):
    if not records:
        print("\n--- ML Transform ---")
        print("No new records to classify.")
        return []

    feature_names = metadata["feature_names"]
    frame = pd.DataFrame(records)
    features = frame[feature_names]

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    confidences = probabilities.max(axis=1)

    enrichment_records = []
    for index, prediction in enumerate(predictions):
        enrichment_records.append(
            {
                "date": records[index]["date"],
                "good_for_running": bool(prediction),
                "confidence": round(float(confidences[index]), 4),
            }
        )

    good_count = sum(row["good_for_running"] for row in enrichment_records)

    print("\n--- ML Transform ---")
    print(f"Good running days: {good_count}")
    print(
        "Confidence range: "
        f"{min(confidences):.4f} to {max(confidences):.4f}"
    )

    return enrichment_records


# --- Step 3: LLM Transform ---

def build_user_message(raw_record, enrichment_record):
    prediction = (
        "good for running"
        if enrichment_record["good_for_running"]
        else "skip running"
    )

    return (
        f"Date: {raw_record['date']}\n"
        f"Maximum temperature: {raw_record['temperature_2m_max']} C\n"
        f"Minimum temperature: {raw_record['temperature_2m_min']} C\n"
        f"Precipitation: {raw_record['precipitation_sum']} mm\n"
        f"Maximum wind speed: {raw_record['wind_speed_10m_max']} km/h\n"
        f"ML prediction: {prediction}\n"
        f"Model confidence: {enrichment_record['confidence']:.2%}"
    )


def fallback_summary(enrichment_record):
    recommendation = (
        "a good day for running"
        if enrichment_record["good_for_running"]
        else "a day to skip running"
    )
    return (
        f"The model predicts {recommendation}, but an AI-generated weather "
        "explanation could not be created."
    )


def enrich_with_llm(client, raw_records, enrichment_records):
    raw_by_date = {row["date"]: row for row in raw_records}
    total = len(enrichment_records)

    print("\n--- LLM Transform ---")

    for index, enrichment_record in enumerate(enrichment_records, start=1):
        raw_record = raw_by_date[enrichment_record["date"]]
        summary = None

        if client is not None:
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": build_user_message(
                                raw_record,
                                enrichment_record,
                            ),
                        },
                    ],
                    temperature=0.2,
                )
                summary = response.choices[0].message.content
                if summary:
                    summary = " ".join(summary.split())
            except Exception as error:
                print(
                    f"LLM error for {enrichment_record['date']}: "
                    f"{error}; using fallback."
                )

        enrichment_record["llm_summary"] = (
            summary or fallback_summary(enrichment_record)
        )

        if index % 50 == 0 or index == total:
            print(f"LLM progress: {index}/{total}")

    return enrichment_records


# --- Step 4: Load ---

def load_enriched_records(supabase, records):
    if not records:
        print("\n--- Load ---")
        print("Upserted 0 rows.")
        return []

    response = (
        supabase
        .table("weather_enriched")
        .upsert(records, on_conflict="date")
        .execute()
    )

    print("\n--- Load ---")
    print(f"Upserted {len(response.data or [])} rows.")
    return response.data or []


# --- Step 5: Verify ---

def verify_enriched_records(supabase):
    count_response = (
        supabase
        .table("weather_enriched")
        .select("*", count="exact")
        .execute()
    )
    sample_response = (
        supabase
        .table("weather_enriched")
        .select("date,good_for_running,confidence,llm_summary")
        .order("date")
        .limit(5)
        .execute()
    )
    good_response = (
        supabase
        .table("weather_enriched")
        .select("*", count="exact")
        .eq("good_for_running", True)
        .execute()
    )

    print("\n--- Verification ---")
    print(f"Total enriched rows: {count_response.count}")
    print("Five sample rows:")
    for row in sample_response.data or []:
        print(row)
    print(f"Good running days: {good_response.count}")


# After running the complete pipeline, inspect several generated summaries here.
# Record one strong summary and one weak summary, then explain whether each one
# reflects both the weather features and ML prediction. A weaker response could
# be caused by an ambiguous prompt, a borderline confidence score, unusual input
# values, or normal variation in LLM generation.


# --- Step 6: Reflect ---
#
# The assignment describes the classifier as trained on Charlotte, NC weather,
# while my Week 9 raw records are for New York City, so I would not assume that
# its predictions transfer perfectly. The cities have different temperature and
# precipitation distributions, creating data drift from the classifier's
# training environment. The LLM cannot override the classifier in this pipeline
# because the prediction is supplied as final and the LLM only turns that result
# and the weather features into a readable recommendation. This makes the LLM
# additive: it improves communication but cannot correct a bad ML classification.
# With 50,000 records, my main concerns would be LLM API cost, rate limits, and
# latency because every unprocessed row creates a separate request. I would use
# incremental processing, batch or concurrent calls within the API's limits,
# exponential-backoff retries, checkpoints, and monitoring, and I would consider
# generating summaries only when a user requests them or using deterministic
# templates for routine cases.


# --- Pipeline ---

def main():
    supabase = get_supabase_client()
    model, metadata = load_model_files()

    unprocessed_records = get_unprocessed_records(supabase)
    enrichment_records = run_ml_transform(
        model,
        metadata,
        unprocessed_records,
    )

    if enrichment_records:
        openai_client = get_openai_client()
        enrich_with_llm(
            openai_client,
            unprocessed_records,
            enrichment_records,
        )

    load_enriched_records(supabase, enrichment_records)
    verify_enriched_records(supabase)


if __name__ == "__main__":
    main()
