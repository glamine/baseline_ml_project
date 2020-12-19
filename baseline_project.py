# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:58:24 2020

@author: guill
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

path = os.getcwd() + '/datasets/house_price_competition/'

# Read the data
X_full = pd.read_csv(path + 'train.csv', index_col='Id')
X_test_full = pd.read_csv(path + 'test.csv', index_col='Id')

#%%

