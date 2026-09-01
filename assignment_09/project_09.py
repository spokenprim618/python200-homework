import os
from datetime import datetime

import requests
from dotenv import load_dotenv
from supabase import create_client


# --- Supabase Connection ---

def get_client():
    load_dotenv()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url:
        raise ValueError("SUPABASE_URL is missing from the environment.")

    if not key:
        raise ValueError(
            "SUPABASE_KEY is missing from the environment."
        )

    return create_client(url, key)


# --- Step 1: Extract ---

def extract_weather():
    # New York City
    latitude = 40.7128
    longitude = -74.0060

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ],
        "timezone": "America/New_York",
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    print("--- Extract Summary ---")
    print("Status code:", response.status_code)
    print("Latitude:", data.get("latitude"))
    print("Longitude:", data.get("longitude"))
    print("Timezone:", data.get("timezone"))
    print("Days returned:", len(data["daily"]["time"]))

    return data


# --- Step 2: Transform ---
# my own notes not to be graded
# This transform is done to turn the connected column data into row data so every record is toghther and handled by insert
def transform_weather(data):
    daily = data["daily"]

    records = []

    for i in range(len(daily["time"])):
        record = {
            "date": daily["time"][i],
            "temperature_2m_max": daily["temperature_2m_max"][i],
            "temperature_2m_min": daily["temperature_2m_min"][i],
            "precipitation_sum": daily["precipitation_sum"][i],
            "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
        }

        records.append(record)

    print("\n--- Transform Summary ---")
    print("First record:")
    print(records[0])

    print("\nLast record:")
    print(records[-1])

    print("\nRecords created:", len(records))

    # I expect to 365 days and I recieved 365 rows.
    # Due to working with year data there could be difference due to leap years and would instead create 366 days.

    return records


# --- Step 3: Load ---

def load_weather(supabase, records):
    response = (
        supabase
        .table("weather_raw")
        .upsert(
            records,
            on_conflict="date"
        )
        .execute()
    )

    print("\n--- Load Summary ---")
    print(f"Upserted {len(response.data)} rows.")
# This tells me idempotency is good practice when loading data into a db and removes possiblity of duplicates

    return response.data


# --- Step 4: Verify ---

def verify_weather(supabase):
    print("\n--- Verification ---")

    # Total row count
    count_response = (
        supabase
        .table("weather_raw")
        .select("*", count="exact")
        .execute()
    )

    print("Total rows:", count_response.count)

    # Earliest date
    earliest_response = (
        supabase
        .table("weather_raw")
        .select("date")
        .order("date")
        .limit(1)
        .execute()
    )

    if earliest_response.data:
        earliest_date = earliest_response.data[0]["date"]
        print("Earliest date:", earliest_date)
    else:
        print("Earliest date: No data")

    # Latest date
    latest_response = (
        supabase
        .table("weather_raw")
        .select("date")
        .order("date", desc=True)
        .limit(1)
        .execute()
    )

    if latest_response.data:
        latest_date = latest_response.data[0]["date"]
        print("Latest date:", latest_date)
    else:
        print("Latest date: No data")

    # Look for July 4, 2023
    target_date = "2023-07-04"

    july_fourth_response = (
        supabase
        .table("weather_raw")
        .select("*")
        .eq("date", target_date)
        .execute()
    )

    if july_fourth_response.data:
        print("\n2023-07-04 record:")
        print(july_fourth_response.data[0])

    else:
        print("\n2023-07-04 was not found.")

        nearby_response = (
            supabase
            .table("weather_raw")
            .select("*")
            .gte("date", "2023-01-01")
            .lte("date", "2023-12-31")
            .execute()
        )

        if nearby_response.data:
            target = datetime.strptime(
                target_date,
                "%Y-%m-%d"
            )

            nearest = min(
                nearby_response.data,
                key=lambda row: abs(
                    datetime.strptime(
                        row["date"],
                        "%Y-%m-%d"
                    ) - target
                )
            )

            print("Nearest available record:")
            print(nearest)
        else:
            print("No 2023 records were found.")


# --- Pipeline ---

def main():
    supabase = get_client()

    weather_data = extract_weather()

    records = transform_weather(weather_data)

    load_weather(
        supabase,
        records
    )

    verify_weather(supabase)


if __name__ == "__main__":
    main()