import pandas as pd
import numpy as np

"""
This file contains all the methods needed to take an ascii (tabular) formatted file and return data.
"""

# Some filler variables in case you just want to see how the Reader works.
filename = 'keck_vels/0748-01711-1_KECK.txt'

cols = ["Julian Date", "Radial Velocity (m/s)", "Error (m/s)", "S Value", "H Alpha",
        "Median photons per pixel", "Exposure Time (seconds)"]


def read(file_path=filename, column_names=cols):
    """
    Reads an ascii file and returns its pandas data frame given the file path it's located at and the
    column names of the data frame. The default column_names is the default column names of the HiRES
    radial velocity data (https://ebps.carnegiescience.edu/data/hireskeck-data). The default filepath is
    the first HiRES dataset in the HiRES (unbinned) data.

    :type column_names: list
    :param file_path: The file path (string) at which the ascii file is located at.
    :param column_names: The list of column names of the returned DataFrame.
    :return: The pandas DataFrame filled with the data from the ascii file with the specified columns.
    """
    data = pd.DataFrame(columns=column_names)
    with open(file_path, 'r') as f:
        # f = open(url, 'r')
        for line in f:
            line = str(line.strip())
            col = line.split()
            col = np.asarray(col)
            col = col.astype('float64')
            col = np.transpose(col)
            df = pd.DataFrame(data=col).T
            df.columns = column_names
            data = data.append(df, ignore_index=True)
        f.close()

    return data
