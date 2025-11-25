import os
import pandas as pd

INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/processed"

def prepare_ml_dataset(df):
    # Drop rows with any NaN (due to rolling windows, lags, etc.)
    df = df.dropna()

    # Select ML features
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "Return", "LogReturn",
        "Lag1", "Lag3", "Lag5",
        "MA5", "MA10", "MA20",
        "Volatility5", "Volatility10", "Volatility20",
        "VolMA5", "VolMA10", "VolMA20",
        "RSI14",
        "MACD", "Signal", "MACD_Hist"
    ]

    # Add target
    feature_cols.append("Target")

    # Return selected columns only
    return df[feature_cols]


def main():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith("_features.parquet")]

    for filename in files:
        print(f"\nPreparing ML dataset for {filename}")
        
        path = os.path.join(INPUT_DIR, filename)
        df = pd.read_parquet(path)

        ml_df = prepare_ml_dataset(df)

        ticker = filename.split("_")[0].upper()
        out_path = os.path.join(OUTPUT_DIR, f"{ticker}_ml.parquet")

        ml_df.to_parquet(out_path)

        print(f"Saved ML-ready dataset -> {out_path}")


if __name__ == "__main__":
    main()
