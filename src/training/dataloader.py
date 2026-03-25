from torch.utils.data import DataLoader
from pathlib import Path
from src.training.dataset import BraTSDataset
from src.training.dataset import collate_skip_none

def get_dataloader(data_dir: Path, case_names, transforms, batch_size=1, shuffle=False):
    """
    Takes a list of cases and applies transformation before saving them to a dataloader object for training or validation.

     Args:
        data_dir: The directory of the preprocessed dataset
        case_names: The target cases for transformation
        transforms: The selected transformations to be applied to the selected data category
        batch_size: Size of the batch to be processed during training
        shuffle (Bool): Determine whether the dataset should be shuffled in each epoch

    Returns:
        Dataloader: Pytorch dataloader object ready for training or evaluation
     """

    dataset = BraTSDataset(data_dir, case_names, transforms)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True, collate_fn=collate_skip_none)