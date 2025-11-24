import os
import pandas as pd

RAW_DIR = "data/raw"
def validate_file(path):
    print(f"\n=== Validating: {os.path.basename(path)} ===")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            print(f" Missing required column: {col}")
        else:
            print(f"Found column: {col}")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        print("Date range:", df["Date"].min(), "→", df["Date"].max())
        print("Invalid date entries:", df["Date"].isnull().sum())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    if "Date" in df.columns:
        dup = df["Date"].duplicated().sum()
        print("Duplicate dates:", dup)
    
    print("\nValidation completed.")

def main():
    files = os.listdir(RAW_DIR)
    csv_files = [f for f in files if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in data/raw/")
        return

    for filename in csv_files:
        path = os.path.join(RAW_DIR, filename)
        validate_file(path)


if __name__ == "__main__":
    main()
