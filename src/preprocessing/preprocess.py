import nibabel as nib
from pathlib import Path
import SimpleITK as sitk
import numpy as np

def preprocess_case(case_dir: Path, output_dir: Path):
    """
    Preprocesses a case to prepare it for training.

    Args:
        case_dir (Path): Directory of the target case
        output_dir (Path): Output target directory

    Returns:
        None. Saves preprocessed NIfTI volumes to output_dir/case_name/
    """

    # Load NIfTI volumes and convert to numpy arrays
    case_name = case_dir.name
    case_files = {
        "seg":nib.load(case_dir / f"{case_name}-seg.nii.gz"),
        "t1c":nib.load(case_dir / f"{case_name}-t1c.nii.gz"),
        "t1n":nib.load(case_dir / f"{case_name}-t1n.nii.gz"),
        "t2f":nib.load(case_dir / f"{case_name}-t2f.nii.gz"),
        "t2w":nib.load(case_dir / f"{case_name}-t2w.nii.gz")
    }

    modality_arrays = {
        "seg": case_files["seg"].get_fdata(),
        "t1c": case_files["t1c"].get_fdata(),
        "t1n": case_files["t1n"].get_fdata(),
        "t2f": case_files["t2f"].get_fdata(),
        "t2w": case_files["t2w"].get_fdata(),
    }

    # Typecast seg mask labels to int
    modality_arrays["seg"] = modality_arrays["seg"].astype(np.uint8)

    # Store contrast modality affine matrix for later use
    affine = case_files["t1c"].affine

    # Delete case_files dictionary to save memory
    del case_files

    # Instantiate bias field corrector
    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    # Apply bias field correction to all modalities besides segmentation
    for modality in modality_arrays:
        if modality == "seg":
            continue
        corrected_modality = modality_arrays[modality]
        corrected_modality = sitk.GetImageFromArray(corrected_modality.astype(np.float32))
        corrected_modality = corrector.Execute(corrected_modality)
        modality_arrays[modality] = sitk.GetArrayFromImage(corrected_modality)

    # Apply Z-score normalization to all modalities besides segmentation
    for modality in modality_arrays:
        if modality == "seg":
            continue
        nonzero_mask = modality_arrays[modality] != 0
        brain_voxels = modality_arrays[modality][nonzero_mask]
        modality_arrays[modality] = (modality_arrays[modality] - brain_voxels.mean()) / brain_voxels.std()
        modality_arrays[modality][~nonzero_mask] = 0

    # Create the combined brain mask
    combined_brain_mask = np.stack([
        modality_arrays["t1c"],
        modality_arrays["t1n"],
        modality_arrays["t2f"],
        modality_arrays["t2w"]])

    # Remove voxels outside the bounding box
    combined_brain_mask = np.any(combined_brain_mask, axis=0)
    bounding_box = np.where(combined_brain_mask)

    # Get the bounding box dimensions
    x_min, x_max = bounding_box[0].min(), bounding_box[0].max() + 1
    y_min, y_max = bounding_box[1].min(), bounding_box[1].max() + 1
    z_min, z_max = bounding_box[2].min(), bounding_box[2].max() + 1

    # Crop the modalities to the bounding box area
    for modality in modality_arrays:
        modality_arrays[modality] = modality_arrays[modality][x_min:x_max, y_min:y_max, z_min:z_max]

    # Save the preprocessed modalities
    case_output_dir = output_dir / case_name
    case_output_dir.mkdir(parents=True, exist_ok=True)

    for modality in modality_arrays:
        nib.save(nib.Nifti1Image(modality_arrays[modality], affine), case_output_dir / f"{case_name}-{modality}.nii.gz")

    return None