import os
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

def clean_df(df):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if pd.api.types.is_datetime64_any_dtype(df["Date"]) and df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_convert(None)

    df = df.sort_values("Date")
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[numeric_cols] = df[numeric_cols].fillna(method="ffill")
    df["Dividends"] = df["Dividends"].fillna(0)
    df["Stock Splits"] = df["Stock Splits"].fillna(0)
    negative_mask = (df[numeric_cols] < 0).any(axis=1)
    if negative_mask.sum() > 0:
        print(f"{negative_mask.sum()} rows with negative values removed.")
        df = df[~negative_mask]
    df = df.reset_index(drop=True)

    return df

def main():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]

    if not files:
        print("No raw CSV files found.")
        return

    for filename in files:
        print(f"\nCleaning {filename}")
        in_path = os.path.join(RAW_DIR, filename)

        df = pd.read_csv(in_path)
        cleaned = clean_df(df)

        ticker = filename.split("_")[0].upper()
        out_path = os.path.join(PROCESSED_DIR, f"{ticker}_clean.parquet")

        cleaned.to_parquet(out_path, index=False)

        print(f"Saved cleaned file to {out_path}")

if __name__ == "__main__":
    main()
