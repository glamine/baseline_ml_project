# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:02:38 2020

@author: guill
"""

import os
import pandas as pd

path = os.getcwd()

melbourne_file_path = path + '/datasets/melb_house_prices.csv'

# read the data and store data in DataFrame titled melbourne_data
df = pd.read_csv(melbourne_file_path)

df1 = pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
df2 = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})
df3 = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])
print(df3)

ds1 = pd.Series([1, 2, 3, 4, 5])
ds2 = pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')

print(ds1)
print(ds2)

# specify index_col 
#df = pd.read_csv("myData.csv", index_col=0)

df = pd.read_csv(melbourne_file_path)

df.head()
df.shape
#df.columnName
print(df.columns)
#Or
#df['columnName']
#df['columnName'][0] # column then row

#%% indexing in pandas

# loc and iloc
# ATTENTION, they do ROW first, COLUMN second here

# ILOC : position based
df.iloc[0] # first row
df.iloc[:,0] # first col
df.iloc[-5:]
df.iloc[[3,5,7]:]

# LOC : label based (index and column name)
# usually more use loc in pandas

df.loc[0, 'columnName']
df.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]
df.loc['Apples':'Potatoes']

# warning : iloc uses the Python stdlib indexing scheme, where the first element of the range is included and the last one excluded.
# loc, meanwhile, indexes inclusively. So 0:10 will select entries 0,...,10

#%% manipulate index:
    
df.set_index("columnName")

#%% conditional selection

df.loc[df.country == 'Italy']
df.loc[(df.country == 'Italy') & (df.points >= 90)]
df.loc[(df.country == 'Italy') | (df.points >= 90)]


# isin
# isin is lets you select data whose value "is in" a list of values
df.loc[df.country.isin(['Italy', 'France'])]

#isnull(), notnull()
df.loc[df.price.notnull()] # NaN numbers

# assign values
df['critic'] = 'everyone' # new col critic with all everyone values
df['index_backwards'] = range(len(df), 0, -1) # reverse incr index

#%% summary function

df.points.describe()
df.points.mean() # median(), mode() min() max() len()
df.points.unique()
df.points.value_counts()

#%% maps

review_points_mean = df.points.mean()
df.points.map(lambda p: p - review_points_mean)

# The function you pass to map() should expect a single value from the Series 
# (a point value, in the above example), and return a transformed version of 
# that value. map() returns a new Series where all the values have been 
# transformed by your function.

# map takes a column

# apply() is the equivalent method if we want to transform a whole DataFrame 
# by calling a custom method on each row.

# apply takes a row, for all rows OR each column with : axis='index'

def remean_points(row):
    row.points = row.points - review_points_mean
    return row

df.apply(remean_points, axis='columns') # axis='index'

# Note that map() and apply() return NEW !!!, transformed Series and DataFrames, 
# respectively.

# operation on series, return new series
newds = df.country + " - " + df.region_1 # understand if strings
# All of the standard Python operators (>, <, ==, and so on) work in this manner.
# NOTE : faster than map and apply

# find max value in column
bargain_idx = (df.points / df.price).idxmax()
bargain_wine = df.loc[bargain_idx, 'title']

# find string in other string : substring in string !
# sum on booleans : true = 1, false = 0, because int(True) = 1

n_trop = df.description.map(lambda desc: "tropical" in desc).sum()
n_fruity = df.description.map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])

# more complex functions :
    
def stars(row):
    if row.country == 'Canada':
        return 3
    elif row.points >= 95:
        return 3
    elif row.points >= 85:
        return 2
    else:
        return 1

star_ratings = df.apply(stars, axis='columns')

#%% groupwise analysis

df.groupby('points').points.count() # equivalent to value_counts()
# creates groups of same value for points column
# index names takes the 'points' name as index

# get min priced wine per category
df.groupby('points').price.min()

# groupby returns a slice of dataframe on which any fun can be applied
df.groupby('winery').apply(lambda df: df.title.iloc[0])
# get first wine in group for each winery

# mutliple indexes for groups :
# get best wine for each country AND province
df.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])
mi = df.index
type(mi)
# pandas.core.indexes.multi.MultiIndex

# reset multi index :
df.reset_index()    

# agg(), which lets you run a bunch of different functions on your DataFrame simultaneously
df.groupby(['country']).price.agg([len, min, max])
# returns dataframe with 'country' index and columns len, min, max

#%% sorting

df.sort_values(by='len') # by 'column name' # sort_values(by='len', ascending=False)
df.sort_index() # sort by index
df.sort_values(by=['country', 'len']) # multiple values, order of priority

#%% Dtypes

df.price.dtype
# dtype('float64')

# get all data types
df.dtypes

# convert type
df.points.astype('float64')
# strings are objects in pandas :
# convert as strings    .astype(str)

# note : Pandas also supports more exotic data types, such as categorical data 
# and timeseries data

#%% missing data

# replace na (float64) by string "Unknown"
df.region_2.fillna("Unknown")

#♠ replace value by another
df.taster_twitter_handle.replace("@kerinokeefe", "@kerino")
# pratique also when NaN is not noted NaN

#%% renaming

#rename(), which lets you change index names and/or column names
df.rename(columns={'points': 'score'})
df.rename(index={0: 'firstEntry', 1: 'secondEntry'})

# give names to columns and rows axes
df.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')

#%% combining : concat and join

canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")

pd.concat([canadian_youtube, british_youtube]) # only if same type of dataset, same columns rows

# join
# join() lets you combine different DataFrame objects which have an index in common

# The lsuffix and rsuffix parameters are necessary here because the data has 
# the same column names in both British and Canadian datasets. 
# If this wasn't true (because, say, we'd renamed them beforehand) we wouldn't
# need them.

left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])

left.join(right, lsuffix='_CAN', rsuffix='_UK')
