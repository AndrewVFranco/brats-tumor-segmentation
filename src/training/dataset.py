import nibabel as nib
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, data_dir: Path, case_names: list, transforms=None):
        """
        PyTorch Dataset for BraTS 2023 GLI brain tumor segmentation.
        Loads the preprocessed multimodal MRI volumes and segmentation masks for a given list of cases.

         Args:
             data_dir (Path): Directory containing preprocessed case folders
             case_names: List of case name strings for this split
             transforms: Optional MONAI transforms pipeline to apply
         """

        self.data_dir = data_dir
        self.case_names = case_names
        self.transforms = transforms

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, idx):
        """
        Receives an index value and returns a complete single data sample at that index

        Args:
            idx: The index for the data item that will be accessed

        Returns:
            Dictionary containing the image tensor and segmentation mask
        """

        # Get the case name at the specified index and store numpy modality arrays into modality_arrays
        case_name = self.case_names[idx]

        modality_arrays = {
            "seg": nib.load(self.data_dir / case_name / f"{case_name}-seg.nii.gz").get_fdata(),
            "t1c": nib.load(self.data_dir / case_name / f"{case_name}-t1c.nii.gz").get_fdata(),
            "t1n": nib.load(self.data_dir / case_name / f"{case_name}-t1n.nii.gz").get_fdata(),
            "t2f": nib.load(self.data_dir / case_name / f"{case_name}-t2f.nii.gz").get_fdata(),
            "t2w": nib.load(self.data_dir / case_name / f"{case_name}-t2w.nii.gz").get_fdata(),
        }

        # Stack modality arrays into (4, H, W, D) structure
        image = np.stack(
            (
                modality_arrays["t1c"],
                modality_arrays["t1n"],
                modality_arrays["t2f"],
                modality_arrays["t2w"]
            ),
            axis=0
        )

        # Create segmentation map array and add dimension to fit (1, H, W, D) structure
        label = np.uint8(modality_arrays["seg"])
        label = np.expand_dims(label, axis=0)

        image_tensor = torch.tensor(image, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Create the final output dictionary
        tensors = {"image": image_tensor, "label": label_tensor}

        # Apply transforms if available
        if self.transforms:
            tensors = self.transforms(tensors)

        return tensors

        
