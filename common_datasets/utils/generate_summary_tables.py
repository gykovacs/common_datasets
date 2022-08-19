import logging
import sys

import numpy as np
import pandas as pd

import common_datasets.binary_classification
import common_datasets.regression
import common_datasets.multiclass_classification

logging.basicConfig(level=logging.INFO)

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=200)
np.set_printoptions(precision=2)

columns, data= common_datasets.binary_classification.generate_summary_table()

for row in data:
    row[0]= np.round(row[0], 2)
    row[1]= np.round(row[1], 2)
    row[3]= np.round(row[3], 2)
    row[4]= np.round(row[4], 2)

data

columns, data= common_datasets.regression.generate_summary_table()

data

columns, data= common_datasets.multiclass_classification.generate_summary_table()

data
