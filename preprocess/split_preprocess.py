# preprocess/split_preprocess.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from text_cleaning import clean_text

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INPUT_FILE = os.path.join(DATA_DIR, "car_absa_labeled.csv")

def prepare_dataset(input_path=INPUT_FILE, out_train="train.csv", out_test="test.csv", test_size=0.2, random_state=42):
    os.makedirs(os.path.dirname(out_train) or ".", exist_ok=True)
    df = pd.read_csv(input_path)
    if "review_text" not in df.columns:
        raise KeyError("Input dataset must have a 'review_text' column.")

    df["clean_text"] = df["review_text"].apply(clean_text)

    # If rule-based ABSA output file exists with 'detected_sentiment', prefer that for labels
    if "detected_sentiment" not in df.columns:
        # fallback: create coarse labels from rating if available
        if "rating" in df.columns:
            df["detected_sentiment"] = df["rating"].apply(lambda r: "positive" if r >= 4 else ("neutral" if r == 3 else "negative"))
        else:
            df["detected_sentiment"] = "neutral"

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["detected_sentiment"] if df["detected_sentiment"].nunique()>1 else None)

    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test, index=False)
    print(f"Saved {len(train_df)} train rows → {out_train}")
    print(f"Saved {len(test_df)} test rows → {out_test}")
    return train_df, test_df

if __name__ == "__main__":
    # default outputs to data/ directory
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    train_fp = os.path.join(out_dir, "train.csv")
    test_fp = os.path.join(out_dir, "test.csv")
    prepare_dataset(out_train=train_fp, out_test=test_fp)
