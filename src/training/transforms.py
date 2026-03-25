from monai.transforms import (
    Compose,
    EnsureTyped,
    Orientationd,
    Spacingd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandCropByPosNegLabeld
)

def get_train_transforms():
    """
    Apply data augmentation techniques to the training dataset

     Returns:
        Compose: Compose of all transforms including augmentation
     """

    return Compose([
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        RandFlipd(keys=["image", "label"], prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.5),
        RandScaleIntensityd(keys=["image"], prob=0.5, factors=0.1),
        RandShiftIntensityd(keys=["image"], prob=0.5, offsets=0.1),
        RandGaussianNoised(keys=["image"], prob=0.2),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", pos=1, neg=1, spatial_size=(128, 128, 128), num_samples=2)
    ])

def get_val_transforms():
    """
    Apply deterministic transformations including rotation and flipping to the validation data

     Returns:
        Compose: Compose of deterministic transforms only
     """

    return Compose([
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"))
    ])
