from monai.networks.nets import UNet

def get_model():
    """
    Configures the UNet algorithm that will be used to train the model

     Returns:
         UNet: The configured UNet algorithm
     """
    return UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    )