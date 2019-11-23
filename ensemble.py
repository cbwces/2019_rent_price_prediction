#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np

prediction_path = os.environ['PREDICTION_FILE']
np.around((pd.read_csv(prediction_path + '//prediction1.csv') + pd.read_csv(prediction_path + '//prediction2.csv')) / 2).to_csv(prediction_path + '//final_prediction_result.csv', index=None, header=None)
