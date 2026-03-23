import numpy as np
import nibabel as nib
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import sys
import json

def create_splits(data_dir: Path, output_dir: Path):
    """
    Takes processed data and splits it into stratified training, validation, and testing sets.

    Args:
        data_dir (Path): Directory of the target case
        output_dir (Path): Output target directory to save the split dataset

    Returns:
        None. Saves stratified dataset to output_dir
    """

    # Create a sorted list of case names from the processed data directory
    case_names = sorted(case.name for case in data_dir.iterdir())
    strat_labels = []

    # Build stratification label set for each case based on tumor subregions present
    for case in case_names:
        seg = nib.load(data_dir / case / f"{case}-seg.nii.gz").get_fdata()
        labels = np.unique(seg)
        tumor_labels = [str(l) for l in [1, 2, 3] if l in labels]
        strat_labels.append("_".join(tumor_labels))

    label_counts = Counter(strat_labels)

    # Remove cases whose label combination appears only once — too rare to stratify
    filtered = [(case, label) for case, label in zip(case_names, strat_labels) if label_counts[label] > 1]

    filtered_case_names, filtered_strat = zip(*filtered)
    filtered_case_names = list(filtered_case_names)
    filtered_strat = list(filtered_strat)

    # Create data splits - stratified 70/15/15 train/val/test
    data_train, data_split_temp, train_strat, temp_strat= train_test_split(filtered_case_names, filtered_strat, test_size=0.3, random_state=42, stratify=filtered_strat)
    data_val, data_test, val_strat, test_strat = train_test_split(data_split_temp, temp_strat, test_size=0.5, random_state=42, stratify=temp_strat)

    final_split = {"train":data_train, "test": data_test, "val": data_val}

    # Save split case names to JSON file for reproducibility
    with open(output_dir / "splits.json", "w") as f:
        json.dump(final_split, f)

    return None

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "splits"

    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        create_splits(PROCESSED_DATA_DIR, OUTPUT_DATA_DIR)
        print("\nDataset split successfully.")
    except Exception as e:
        print(f"\nSplitting failed: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main())