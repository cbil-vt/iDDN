"""Load resources in the iDDN package

Include some example data and images. They are used in the tutorial and the testing part.
"""

import importlib.resources
from pathlib import Path
import numpy as np

data_dir = importlib.resources.files("iddn_data")


def load_example(file_name="example.npz"):
    """Load example and testing data of iDDN

    Parameters
    ----------
    file_name : str
        The name of the data file to load. It should be a NumPy npz file.

    Returns
    -------
    The loaded NumPy object

    """
    dat_file = Path(data_dir, file_name)
    return np.load(dat_file)


def get_image_path(img_name="three_layers.png"):
    """Get the full path of an image in the resource folder of the iDDN package

    Parameters
    ----------
    img_name : str
        The name of the image to load

    Returns
    -------
    The full path of the image

    """
    return Path(data_dir, img_name)
