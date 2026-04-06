import pandas as pd
from src.utils.config import INPUT_CSV, BRAND_KEYWORD

REQUIRED_COLUMNS = ["post_id", "platform", "date", "brand", "text", "engagement"]


def load_raw_data():
    
    df = pd.read_csv(INPUT_CSV)

    # Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    df = df.copy()

    df["brand"] = df["brand"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # Filter brand
    filtered_df = df[df["brand"].str.lower() == BRAND_KEYWORD.lower()].copy()

    filtered_df["date"] = pd.to_datetime(filtered_df["date"], errors="coerce")

    return filtered_df


if __name__ == "__main__":
    
    df = load_raw_data()
    
    print(df.head())
    print("\nRows loaded:", len(df))