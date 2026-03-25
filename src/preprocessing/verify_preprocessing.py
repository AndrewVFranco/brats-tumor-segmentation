import sys
import nibabel as nib
from pathlib import Path

def verify_dataset_processing(data_dir: Path):
    """
    Iterates through the processed dataset directory to ensure that all data has been successfully processed.

    Args:
        data_dir (Path): Directory of the preprocessed data
    """

    files = ["seg", "t1c", "t1n", "t2f", "t2w"]
    passed = 0
    failed = 0
    failed_cases = []

    for case in data_dir.iterdir():
        try:
            case_name = case.name
            for modality in files:
                shape = nib.load(data_dir / case_name / f"{case_name}-{modality}.nii.gz").shape
                if any(dim == 0 for dim in shape):
                    raise ValueError(f"Invalid shape {shape} for {modality}")
            passed += 1

        except Exception as e:
            print(f"\nCase loading failed: {e}")
            failed_cases.append(case)
            failed += 1

    print(f"Cases passed: {passed}\nCases failed: {failed}")

    if failed_cases:
        print(f"Failed cases: {failed_cases}")
    else:
        print("All cases passed verification.")

    return None

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data" / "processed"

    verify_dataset_processing(DATA_DIR)
    return 0

if __name__ =="__main__":
    sys.exit(main())