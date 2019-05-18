import numpy as np
import torch


def tensor2image(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        # get the data from a variable
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image

        # convert it into a numpy array
        image_numpy = image_tensor[0].cpu().float().numpy()

        # grayscale to RGB
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        # post-processing: transpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # if it is a numpy array, do nothing
    else:
        image_numpy = input_image

    return image_numpy.astype(imtype)