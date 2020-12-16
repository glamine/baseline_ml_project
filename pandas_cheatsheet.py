# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:02:38 2020

@author: guill
"""

import pandas as pd

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
df = pd.read_csv("myData.csv", index_col=0)
df.head()
df.shape
df.columnName
#Or
df['columnName']
df['columnName'][0] # column then row

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
