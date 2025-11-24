import os
import pandas as pd
import numpy as np

PROCESSED_DIR = "data/processed"
OUT_DIR = "data/processed"   # saving in the same folder for now

def add_basic_features(df):
    # Ensure datetime index
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # === 1. Daily Returns ===
    df["Return"] = df["Close"].pct_change()

    # === 2. Log Returns ===
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))

    # === 3. Lag Features ===
    df["Lag1"] = df["Close"].shift(1)
    df["Lag3"] = df["Close"].shift(3)
    df["Lag5"] = df["Close"].shift(5)

    # === 4. Rolling Means ===
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()

    return df


def main():
    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith("_clean.parquet")]

    for filename in files:
        print(f"\nAdding features to {filename}")

        path = os.path.join(PROCESSED_DIR, filename)
        df = pd.read_parquet(path)

        df_feat = add_basic_features(df)

        ticker = filename.split("_")[0].upper()
        out_path = os.path.join(OUT_DIR, f"{ticker}_features.parquet")

        df_feat.to_parquet(out_path)

        print(f"Saved feature file -> {out_path}")


if __name__ == "__main__":
    main()
