import os
from datetime import date

from dotenv import load_dotenv
from supabase import create_client, Client


# --- Supabase Connection ---

# Q1

# supabase-py needs two pieces of information:
#
# 1. The Supabase project URL
# 2. The Supabase project API key
# I find the API key in project settings in API Keys and I use the secret key.
# i can't find the project url but I use the ID within the link and the template url.
# This information should never be hardcoded because others could use bots to scrape my repo to use my api without my knowledge which would cost me along with other attacks.

# Q2
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


# Q3
# RLS is a security policy to make sure only the users you want can use CRUD actions on the db.
# I disabled RLS for this course because it is just me working on this project and I won't be having others or my key visable. THerefore isn't need for this security measure.
# In the real world I would turn on RLS if I were to have multiple users accessing the db and I would need to make sure the right people have access to the DB along with user accounts.

# --- supabase-py CRUD ---

# Q1
def insert_test_record(supabase):
    record = {
        "date": date.today().isoformat(),
        "temperature_2m_max": 27.0,
        "temperature_2m_min": 19.0,
        "precipitation_sum": 0.5,
        "wind_speed_10m_max": 18.0,
    }

    response = (
        supabase
        .table("weather_raw")
        .insert(record)
        .execute()
    )

    print("Inserted record:")
    print(response.data)


# If I did run this function twice I don't have any checks for having duplicate data. Due to not having any checks if a row exists ALso I don't have any other variables for the temperature and so on so it will always create some duplicate data except for the date automatically updating.


# Q2
def get_records_by_date_range(supabase, start, end):
    response = (
        supabase
        .table("weather_raw")
        .select("*")
        .gte("date", start)
        .lte("date", end)
        .execute()
    )

    return response.data


# Q3

# Insert in supabase creates a new row with the information provided whether it exists or not.
# Upsert in supabase is insert and update at the same time. If the information already exists then the it will just update the record but if it doesn't it will insert the row like normal



def safe_upsert(supabase, records):
    response = (
        supabase
        .table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )

    rows_affected = len(response.data)

    print(f"Rows affected: {rows_affected}")

    return response.data


# --- Idempotency ---

# Q1

# Idemotency matters for a pipeline because it can handle issues like mentioned above with just using insert could duplicate data with no checks.
# IF the pipeline crashes midway and it restarts from the begining it will complete and duplicate the data from where it crashed.


# --- Tests ---

if __name__ == "__main__":
    supabase = get_client()

    # Q1
    insert_test_record(supabase)

    # Q2
    today = date.today().isoformat()

    records = get_records_by_date_range(
        supabase,
        today,
        today
    )

    print("\nRecords in date range:")
    print(records)

    # Q3 example
    test_records = [
        {
            "date": today,
            "temperature_2m_max": 28.0,
            "temperature_2m_min": 20.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 17.0,
        }
    ]

    safe_upsert(supabase, test_records)