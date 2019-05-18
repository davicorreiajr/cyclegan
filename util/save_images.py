import os
import ntpath
from PIL import Image
from util.tensor2image import tensor2image


def save_images(image_dir, visuals, image_path):
    """Save images to the disk.

    Parameters:
        image_dir (str)          -- path to save images
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths

    """
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    for label, im_data in visuals.items():
        im = tensor2image(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        save_image(im, save_path)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)