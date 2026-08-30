# --- Prefect Pipeline ---

# Pipeline Q2
import numpy as np
import pandas as pd
from prefect import flow, task


@task
def create_series(arr):
    return pd.Series(arr, name = 'values')

@task
def clean_data(series):
    return series.dropna(ignore_index=True)

@task
def summarize_data(series):
    return {"mean":series.agg("mean"),"median":series.agg("median"),"std":series.agg("std"),"mode":series.mode()[0]}

@flow
def pipeline_flow():
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

    new_series = create_series(arr)
    cleaned_series = clean_data(new_series)
    summary = summarize_data(cleaned_series)
    for key in summary:
        print(f"Key {key} and value {summary[key]}")
    return summary

if __name__ == "__main__":
    pipeline_flow()

#Q1
#Prefect is too much overhead here because there isn't any dynamic
#data coming in that needs a step by step check of each task in the flow.
#Many of the features are completely unused

#Q2
# It could be used for the quality of life additions such as explainability of the pipeline
# and auditability. There can be logging and caching to keep records and make reruns faster.