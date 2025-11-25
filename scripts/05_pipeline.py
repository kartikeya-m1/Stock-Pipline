import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


DATA_DIR = "data/processed"


def load_dataset(ticker):
    """Load ML-ready dataset."""
    path = os.path.join(DATA_DIR, f"{ticker}_ml.parquet")
    df = pd.read_parquet(path)
    df = df.sort_index()
    return df


def train_test_split(df, split_ratio=0.8):
    split = int(len(df) * split_ratio)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    feature_cols = [col for col in df.columns if col != "Target"]

    X_train = train_df[feature_cols]
    y_train = train_df["Target"]

    X_test = test_df[feature_cols]
    y_test = test_df["Target"]

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    """Train Gradient Boosting model."""
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    mape = (abs(y_test - preds) / y_test).mean() * 100

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape
    }

    return preds, metrics


def save_model(model, ticker):
    """Save trained model object."""
    out_path = f"models/{ticker}_gbr_model.pkl"
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, out_path)
    print(f"Saved model to {out_path}")


def main():
    tickers = ["TSLA", "AAPL", "AMZN", "MSFT", "JPM"]

    for ticker in tickers:
        print(f"\n===== Running Pipeline for {ticker} =====")

        df = load_dataset(ticker)

        X_train, y_train, X_test, y_test = train_test_split(df)

        model = train_model(X_train, y_train)

        preds, metrics = evaluate_model(model, X_test, y_test)

        print(f"Metrics for {ticker}: {metrics}")

        save_model(model, ticker)

        # save predictions
        pred_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": preds
        }, index=y_test.index)

        os.makedirs("results", exist_ok=True)
        pred_df.to_parquet(f"results/{ticker}_predictions.parquet")

        print(f"Saved predictions for {ticker}")


if __name__ == "__main__":
    main()
