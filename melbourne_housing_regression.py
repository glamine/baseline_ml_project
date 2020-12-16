# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:04:34 2020

@author: guill
"""
import os
import pandas as pd

path = os.getcwd()

melbourne_file_path = path + '/datasets/melb_house_prices.csv'

# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 

#%% print a summary of the data in Melbourne data
print(melbourne_data.describe())
print(melbourne_data.columns)

print(melbourne_data.Suburb.value_counts())
print(melbourne_data.Suburb.unique())
print(melbourne_data.Suburb.mode()) # median, #mean

print(melbourne_data.dtypes)

#%%

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)



#%%



y = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']

X = melbourne_data[melbourne_features]

print(X.describe())
print(X.head())

#%%

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

#%%
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Define model
melbourne_model = DecisionTreeRegressor(random_state=1)
# Fit model
melbourne_model.fit(train_X, train_y)


forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)



#%% 

from sklearn.metrics import mean_absolute_error

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))


#%%

print("Making predictions for the following 5 houses:")
print(val_X.head())
print("The predictions are")
print(melbourne_model.predict(val_X.head()))
print(val_y.head())

#%% parameter tuning :
    
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
    
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

