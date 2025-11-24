import os
import pandas as pd
import numpy as np

PROCESSED_DIR = "data/processed"
OUT_DIR = "data/processed"   # saving in the same folder for now

def add_features(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # --- Basic Features ---
    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Lag1"] = df["Close"].shift(1)
    df["Lag3"] = df["Close"].shift(3)
    df["Lag5"] = df["Close"].shift(5)
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()

    # --- Rolling Volatility ---
    df["Volatility5"] = df["LogReturn"].rolling(5).std()
    df["Volatility10"] = df["LogReturn"].rolling(10).std()
    df["Volatility20"] = df["LogReturn"].rolling(20).std()

    # --- Rolling Volume ---
    df["VolMA5"] = df["Volume"].rolling(5).mean()
    df["VolMA10"] = df["Volume"].rolling(10).mean()
    df["VolMA20"] = df["Volume"].rolling(20).mean()

    # --- RSI (14) SAFE VERSION ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()

    # Avoid divide by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI14"] = 100 - (100 / (1 + rs))

    # --- MACD ---
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]

    # --- Target Variable: tomorrow's close ---
    df["Target"] = df["Close"].shift(-1)

    return df


def main():
    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith("_clean.parquet")]

    for filename in files:
        print(f"\nAdding features to {filename}")

        path = os.path.join(PROCESSED_DIR, filename)
        df = pd.read_parquet(path)

        df_feat = add_features(df)

        ticker = filename.split("_")[0].upper()
        out_path = os.path.join(OUT_DIR, f"{ticker}_features.parquet")

        df_feat.to_parquet(out_path)

        print(f"Saved feature file -> {out_path}")


if __name__ == "__main__":
    main()
