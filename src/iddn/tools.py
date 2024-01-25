"""Utility functions of DDN
"""

import numpy as np


def standardize_data(data):
    """Standadize each column of the input data"""

    dat_mean = np.mean(data, axis=0)
    dat_std = np.std(data, axis=0)
    return (data - dat_mean) / dat_std
