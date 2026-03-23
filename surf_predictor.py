#!/usr/bin/env python3
"""
Surf Quality Predictor for Ocean Beach, San Francisco.

Fetches live NOAA buoy data and Surfline forecasts, then uses an LLM
(via vLLM or HuggingFace transformers) to predict surf quality ratings.
Outputs a comparison table of LLM vs Surfline predictions.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime

import pandas as pd
import requests
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NOAA_STATION = "46237"  # San Francisco Bar buoy
NOAA_URL = f"https://www.ndbc.noaa.gov/data/realtime2/{NOAA_STATION}.txt"
SURFLINE_SPOT_ID = "5842041f4e65fad6a77087f8"  # Ocean Beach SF
SURFLINE_RATING_URL = (
    f"https://services.surfline.com/kbyg/spots/forecasts/rating"
    f"?spotId={SURFLINE_SPOT_ID}&days=6&intervalHours=1"
)
SURFLINE_WAVE_URL = (
    f"https://services.surfline.com/kbyg/spots/forecasts/wave"
    f"?spotId={SURFLINE_SPOT_ID}&days=6&intervalHours=1"
)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
RATING_LABELS = {1: "Poor", 2: "Poor+", 3: "Fair", 4: "Good", 5: "Epic"}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_noaa_buoy_data() -> pd.DataFrame:
    """Fetch recent observations from NOAA buoy."""
    print("  Fetching NOAA buoy data...")
    df = pd.read_csv(NOAA_URL, sep=r"\s+", skiprows=[1], nrows=50)
    df["datetime"] = pd.to_datetime(
        df[["#YY", "MM", "DD", "hh", "mm"]].rename(
            columns={"#YY": "year", "MM": "month", "DD": "day", "hh": "hour", "mm": "minute"}
        )
    )
    ob_data = df[["datetime", "WVHT", "DPD", "MWD", "WTMP"]].copy()
    ob_data["Wave_Height_ft"] = pd.to_numeric(ob_data["WVHT"], errors="coerce") * 3.28084
    ob_data["Water_Temp_F"] = (pd.to_numeric(ob_data["WTMP"], errors="coerce") * 9 / 5) + 32
    ob_data["datetime"] = ob_data["datetime"].dt.tz_localize("UTC")
    ob_data["datetime_local"] = ob_data["datetime"].dt.tz_convert("America/Los_Angeles")
    return ob_data


def _surfline_headers():
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.surfline.com/",
    }


def fetch_surfline_ratings() -> pd.DataFrame:
    """Fetch Surfline hourly quality ratings."""
    print("  Fetching Surfline ratings...")
    resp = requests.get(SURFLINE_RATING_URL, headers=_surfline_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()

    ratings_df = pd.DataFrame(data["data"]["rating"])
    ratings_df["datetime_utc"] = pd.to_datetime(ratings_df["timestamp"], unit="s", utc=True)
    ratings_df["datetime_local"] = ratings_df.apply(
        lambda row: row["datetime_utc"] + pd.Timedelta(hours=row["utcOffset"]), axis=1
    )
    ratings_df["datetime_local"] = ratings_df["datetime_local"].dt.tz_convert("America/Los_Angeles")
    ratings_df["rating_int"] = ratings_df["rating"].apply(lambda x: int(x["value"]))
    return ratings_df


def fetch_surfline_wave() -> pd.DataFrame:
    """Fetch Surfline hourly wave height forecast."""
    print("  Fetching Surfline wave forecast...")
    resp = requests.get(SURFLINE_WAVE_URL, headers=_surfline_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()

    wave_df = pd.DataFrame(data["data"]["wave"])
    wave_df["datetime_utc"] = pd.to_datetime(wave_df["timestamp"], unit="s", utc=True)
    wave_df["datetime_local"] = wave_df.apply(
        lambda row: row["datetime_utc"] + pd.Timedelta(hours=row["utcOffset"]), axis=1
    )
    wave_df["datetime_local"] = wave_df["datetime_local"].dt.tz_convert("America/Los_Angeles")
    wave_df["surf_min"] = wave_df["surf"].apply(lambda x: x["min"])
    wave_df["surf_max"] = wave_df["surf"].apply(lambda x: x["max"])
    return wave_df


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_prompt_data(ob_data: pd.DataFrame, ratings_df: pd.DataFrame, hours: int):
    """Merge buoy + surfline historical data and create future prediction rows.

    Returns (final_prompt_data, surfline_future) where surfline_future contains
    Surfline's own predictions for the future hours (for comparison).
    """
    now = pd.Timestamp.now(tz="America/Los_Angeles")

    surfline_past = ratings_df.loc[ratings_df["datetime_local"] <= now]
    surfline_future = ratings_df.loc[ratings_df["datetime_local"] > now]

    # Merge historical NOAA + Surfline
    prompt_data = ob_data.merge(
        surfline_past[["datetime_local", "rating_int"]], on="datetime_local", how="inner"
    )

    if prompt_data.empty:
        # If no overlap (buoy reports every 30 min, surfline hourly), do nearest merge
        ob_data_sorted = ob_data.sort_values("datetime_local")
        surfline_past_sorted = surfline_past.sort_values("datetime_local")
        prompt_data = pd.merge_asof(
            ob_data_sorted,
            surfline_past_sorted[["datetime_local", "rating_int"]],
            on="datetime_local",
            direction="nearest",
            tolerance=pd.Timedelta("1h"),
        )
        prompt_data = prompt_data.dropna(subset=["rating_int"])

    if prompt_data.empty:
        print("  Warning: No overlapping historical data. Using buoy data only.")
        prompt_data = ob_data.copy()
        prompt_data["rating_int"] = None

    # Create future rows
    last_ts = prompt_data["datetime_local"].max()
    future_dates = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=hours, freq="h")
    future_df = pd.DataFrame({"datetime_local": future_dates})
    final = pd.concat([future_df, prompt_data], ignore_index=True)

    if "datetime" in final.columns:
        final.drop(columns=["datetime"], inplace=True)

    final = final.astype(object).fillna("None")
    final = final.sort_values(by="datetime_local").reset_index(drop=True)

    return final, surfline_future


def build_prompt(final_prompt_data: pd.DataFrame, hours: int) -> str:
    """Build the LLM prompt from the prepared data."""
    df_clean = final_prompt_data.copy()
    numeric_cols = ["WVHT", "DPD", "MWD", "WTMP", "Wave_Height_ft", "Water_Temp_F"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").round(1)

    table_str = df_clean.to_markdown(index=False)

    prompt = f"""You are the lead surf forecaster for Ocean Beach, San Francisco.
Below is buoy data and observed surf quality ratings (rating_int: 1=Poor to 5=Epic).
Wave_Height_ft is the wave height in feet.

{table_str}

### Task:
The last {hours} rows have rating_int and Wave_Height_ft = 'None'.
Predict both for each hour.
Consider: wave height trends, dominant period (DPD), wave direction (MWD), water temp.
Longer period swells (>12s) from the W/NW (270-310) produce better surf at Ocean Beach.

Output ONLY {hours} lines, each formatted as: rating,height
where rating is an integer 1-5 and height is wave height in feet (one decimal).
Example line: 3,4.2
Predictions:"""
    return prompt


# ---------------------------------------------------------------------------
# LLM inference
# ---------------------------------------------------------------------------
def run_vllm(prompt: str) -> str:
    """Run inference using vLLM."""
    from vllm import LLM, SamplingParams

    print("  Loading model with vLLM...")
    llm = LLM(model=MODEL_NAME)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=150)
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()


def run_huggingface(prompt: str) -> str:
    """Fallback: run inference using HuggingFace transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  Loading model with HuggingFace transformers...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16 if device == "mps" else torch.float32
    ).to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=150, temperature=0.1, do_sample=True
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_inference(prompt: str, backend: str) -> str:
    """Run LLM inference with the chosen backend."""
    if backend == "vllm":
        try:
            return run_vllm(prompt)
        except ImportError:
            print("  vLLM not available, falling back to HuggingFace transformers...")
            return run_huggingface(prompt)
    else:
        return run_huggingface(prompt)


def parse_predictions(raw: str, hours: int) -> list[tuple[int, float]]:
    """Extract (rating, wave_height_ft) pairs from LLM output."""
    # Match lines like "3,4.2" or "3, 4.2"
    pairs = re.findall(r"([1-5])\s*,\s*(\d+\.?\d*)", raw)
    predictions = [(int(r), round(float(h), 1)) for r, h in pairs[:hours]]

    # Pad if we didn't get enough
    while len(predictions) < hours:
        predictions.append((3, 3.0))

    return predictions[:hours]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def display_results(
    future_timestamps: list,
    llm_predictions: list[tuple[int, float]],
    surfline_future: pd.DataFrame,
    surfline_wave: pd.DataFrame,
):
    """Print a pretty ASCII table comparing LLM vs Surfline predictions."""
    rows = []
    for ts, (llm_rating, llm_height) in zip(future_timestamps, llm_predictions):
        ts_aware = ts if ts.tzinfo else ts.tz_localize("America/Los_Angeles")
        ts_floored = ts_aware.floor("h")

        # Find matching Surfline rating
        rating_match = surfline_future.loc[
            surfline_future["datetime_local"].dt.floor("h") == ts_floored
        ]
        if not rating_match.empty:
            sl_rating = int(rating_match.iloc[0]["rating_int"])
            sl_qual = f"{sl_rating} ({RATING_LABELS.get(sl_rating, '?')})"
        else:
            sl_qual = "N/A"

        # Find matching Surfline wave height
        wave_match = surfline_wave.loc[
            surfline_wave["datetime_local"].dt.floor("h") == ts_floored
        ]
        if not wave_match.empty:
            sl_min = wave_match.iloc[0]["surf_min"]
            sl_max = wave_match.iloc[0]["surf_max"]
            sl_wave = f"{sl_min}-{sl_max} ft"
        else:
            sl_wave = "N/A"

        llm_qual = f"{llm_rating} ({RATING_LABELS.get(llm_rating, '?')})"
        llm_wave = f"{llm_height} ft"
        ts_str = ts_aware.strftime("%Y-%m-%d %H:%M %Z")
        rows.append([ts_str, llm_qual, sl_qual, llm_wave, sl_wave])

    print()
    print("=" * 95)
    print("  SURF QUALITY FORECAST — Ocean Beach, San Francisco")
    print("=" * 95)
    print()
    print(
        tabulate(
            rows,
            headers=["Timestamp", "LLM Quality", "Surfline Quality", "LLM Wave Ht", "Surfline Wave Ht"],
            tablefmt="fancy_grid",
            colalign=("left", "center", "center", "center", "center"),
        )
    )
    print()
    print(f"  Model: {MODEL_NAME}")
    print(f"  Buoy:  NOAA Station {NOAA_STATION} (SF Bar)")
    print("=" * 95)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Predict surf quality at Ocean Beach, SF using an LLM."
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=12,
        help="Number of hours to predict (default: 12)",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "huggingface"],
        default="vllm",
        help="Inference backend (default: vllm, falls back to huggingface if unavailable)",
    )
    args = parser.parse_args()

    print()
    print("Surf Predictor — Ocean Beach, SF")
    print("-" * 40)

    # 1. Fetch data
    ob_data = fetch_noaa_buoy_data()
    ratings_df = fetch_surfline_ratings()
    wave_df = fetch_surfline_wave()

    # 2. Prepare prompt data
    final_prompt_data, surfline_future = prepare_prompt_data(ob_data, ratings_df, args.hours)

    # 3. Build prompt and run inference
    prompt = build_prompt(final_prompt_data, args.hours)
    print(f"  Running inference ({args.backend})...")
    t0 = time.time()
    raw_output = run_inference(prompt, args.backend)
    elapsed = time.time() - t0
    print(f"  Inference completed in {elapsed:.1f}s")

    # 4. Parse predictions
    predictions = parse_predictions(raw_output, args.hours)

    # 5. Get future timestamps
    last_ts = final_prompt_data.loc[
        final_prompt_data["rating_int"] != "None", "datetime_local"
    ]
    if last_ts.empty:
        last_ts = pd.Timestamp.now(tz="America/Los_Angeles")
    else:
        last_ts = last_ts.max()
    future_timestamps = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1), periods=args.hours, freq="h"
    )

    # 6. Display
    display_results(future_timestamps, predictions, surfline_future, wave_df)


if __name__ == "__main__":
    main()
