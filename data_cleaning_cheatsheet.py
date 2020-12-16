# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 00:08:57 2020

@author: guill
"""

import pandas as pd
import numpy as np

df = pd.read_csv("myCSV.csv")

# get the number of missing data points per column
missing_values_count = df.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]

#%%

# how many total missing values do we have?
total_cells = np.product(df.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
percent_missing = (total_missing/total_cells) * 100
print(percent_missing)

# look at the # of missing points in the first ten columns
missing_values_count[0:10]

# remove all the rows that contain a missing value
df.dropna()

# remove all columns with at least one missing value
columns_with_na_dropped = df.dropna(axis=1)
columns_with_na_dropped.head()

sf_permits_with_na_imputed = df.fillna(method='bfill', axis=0).fillna(0)

#%%

# modules we'll use
import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# set seed for reproducibility
np.random.seed(0)

#%%

# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size=1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns=[0])

# plot both together to compare
fig, ax = plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")

#%%

# In general, you'll normalize your data if you're going to be using 
# a machine learning or statistics technique that assumes your data is normally 
# distributed. Some examples of these include linear discriminant analysis (LDA) 
# and Gaussian naive Bayes. (Pro tip: any method with "Gaussian" in the name 
# probably assumes normality.)

# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")

#%%

# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

# If you check the pandas dtype documentation here, you'll notice that there's
# also a specific datetime64 dtypes. Because the dtype of our column is object 
# rather than datetime64, we can tell that Python doesn't know that this column 
# contains dates.

#%%

# create a new column, date_parsed, with the parsed dates
df['date_parsed'] = pd.to_datetime(df['date'], format="%m/%d/%y")

df['date_parsed'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

# get the day of the month from the date_parsed column
day_of_month_landslides = df['date_parsed'].dt.day
day_of_month_landslides.head()

# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)

#%%

date_lengths = df.Date.str.len()
date_lengths.value_counts()

indices = np.where([date_lengths == 24])[1]
print('Indices with corrupted data:', indices)
df.loc[indices]

df.loc[3378, "Date"] = "02/23/1975"
df.loc[7512, "Date"] = "04/28/1985"
df.loc[20650, "Date"] = "03/13/2011"
df['date_parsed'] = pd.to_datetime(df['Date'], format="%m/%d/%Y")

#%%

df['Last Known Eruption'].sample(5)

#%%

# modules we'll use
import pandas as pd
import numpy as np

# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)

#%%

# start with a string
before = "This is the euro symbol: €"

# check to see what datatype it is
type(before)

#%%
# encode it to a different encoding, replacing characters that raise errors
after = before.encode("utf-8", errors="replace")

# check the type
type(after)

# convert it back to utf-8
print(after.decode("utf-8"))

#%%

# start with a string
before = "This is the euro symbol: €"

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")

# convert it back to utf-8
print(after.decode("ascii"))

# We've lost the original underlying byte string! It's been 
# replaced with the underlying byte string for the unknown character :(

#%%

# Notice that we get the same UnicodeDecodeError we got when we tried to decode 
# UTF-8 bytes as if they were ASCII! This tells us that this file isn't actually UTF-8. 
# We don't know what encoding it actually is though. One way to figure it out is 
# to try and test a bunch of different character encodings and see if any of them work. 
# A better way, though, is to use the chardet module to try and automatically guess 
# what the right encoding is. It's not 100% guaranteed to be right, but it's 
# usually faster than just trying to guess.

#%%

# look at the first ten thousand bytes to guess the character encoding
with open("../input/kickstarter-projects/ks-projects-201801.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)

# read in the file with the encoding detected by chardet
kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv", encoding='Windows-1252')

# look at the first few lines
kickstarter_2016.head()

# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")

#%%

sample_entry = b'\xa7A\xa6n'
print(sample_entry)
print('data type:', type(sample_entry))

#%%

# modules we'll use
import pandas as pd
import numpy as np

# helpful modules
import fuzzywuzzy
from fuzzywuzzy import process
import chardet

# read in all our data
professors = pd.read_csv("../input/pakistan-intellectual-capital/pakistan_intellectual_capital.csv")

# set seed for reproducibility
np.random.seed(0)

#%%

# get all the unique values in the 'Country' column
countries = professors['Country'].unique()

# sort them alphabetically and then take a closer look
countries.sort()
countries

#%%

# convert to lower case
professors['Country'] = professors['Country'].str.lower()
# remove trailing white spaces
professors['Country'] = professors['Country'].str.strip()

#%%

# get the top 10 closest matches to "south korea"
matches = fuzzywuzzy.process.extract("south korea", countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
matches

#%%

# function to replace rows in the provided column of the provided dataframe
# that match the provided string above the provided ratio with the provided string
def replace_matches_in_column(df, column, string_to_match, min_ratio = 47):
    # get a list of unique strings
    strings = df[column].unique()
    
    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches 
    df.loc[rows_with_matches, column] = string_to_match
    
    # let us know the function's done
    print("All done!")