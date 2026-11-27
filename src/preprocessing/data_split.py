
"""
Prepare THAR binary train/validation/test splits.
label: 0 = non_communal, 1 = communal
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_splits(raw_csv_path, output_dir):
    print(f"Loading THAR data from: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path, encoding="utf-8-sig")

    print(f"Raw rows: {len(df)}")

    # Keep rows with usable comment text and binary labels
    df = df.dropna(subset=["Comment", "label"]).copy()
    df["Comment"] = df["Comment"].astype(str).str.strip()
    df = df[df["Comment"] != ""]

    # Ensure labels are integers: 0 = non-communal, 1 = communal
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]

    # Remove exact duplicate comments before splitting
    before = len(df)
    df = df.drop_duplicates(subset=["Comment"]).reset_index(drop=True)
    print(f"Removed duplicate comments: {before - len(df)}")
    print(f"Rows used: {len(df)}")

    # Keep religion/community annotation for reference.
    # It is NOT a model input; only Comment and label are used.
    keep_cols = [
        col for col in
        ["Identifier", "Comment", "SubTask2", "label", "label_name"]
        if col in df.columns
    ]
    df = df[keep_cols]

    # 80% train, 20% temporary set
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    # Split temporary set into 10% validation and 10% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print("\n=== THAR SPLIT SUMMARY ===")
    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    summary = []
    for split_name, split_df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df)
    ]:
        counts = split_df["label"].value_counts()

        communal = int(counts.get(1, 0))
        non_communal = int(counts.get(0, 0))

        print(f"\n{split_name}:")
        print(f"  Non-communal (0): {non_communal}")
        print(f"  Communal (1):     {communal}")

        summary.append({
            "Split": split_name,
            "Total": len(split_df),
            "Communal": communal,
            "Non_Communal": non_communal
        })

    pd.DataFrame(summary).to_csv(
        os.path.join(output_dir, "split_summary.csv"),
        index=False
    )

    print("\nSplits saved successfully.")


if __name__ == "__main__":
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    raw_path = os.path.join(
        project_dir,
        "data",
        "raw",
        "Thar_Dataset.csv"
    )

    output_dir = os.path.join(
        project_dir,
        "data",
        "processed"
    )

    prepare_splits(raw_path, output_dir)
