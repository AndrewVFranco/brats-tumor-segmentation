from src.preprocessing.preprocess import preprocess_case
from pathlib import Path
from tqdm import tqdm
from functools import partial
import multiprocessing
import sys

def main():
    multiprocessing.set_start_method('fork', force=True)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

    preprocess_with_output = partial(preprocess_case, output_dir=OUTPUT_DIR)

    cases = sorted(DATA_DIR.iterdir())

    N = multiprocessing.cpu_count() - 2

    with multiprocessing.Pool(processes=N) as pool:
        try:
            results = list(tqdm(
                pool.imap(preprocess_with_output, cases),
                total=len(cases)
            ))
            print("\nAll cases processed successfully.")
        except Exception as e:
            print(f"\nPreprocessing failed: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main())