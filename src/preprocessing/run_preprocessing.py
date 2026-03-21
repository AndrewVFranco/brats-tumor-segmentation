from src.preprocessing.preprocess import preprocess_case
from pathlib import Path
from tqdm import tqdm
import sys

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

    cases = sorted(DATA_DIR.iterdir())
    failed_cases = []

    cases = cases[:1]
    for case in tqdm(cases):
        try:
            preprocess_case(case, OUTPUT_DIR)
        except Exception as e:
            print(f"case:{case} failed, error {e}")
            failed_cases.append(case)
            continue

    if failed_cases:
        print(f"\n{len(failed_cases)} cases failed: {failed_cases}")
    else:
        print("\nAll cases processed successfully.")

    return 0

if __name__ == "__main__":
    sys.exit(main())