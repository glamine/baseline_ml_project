# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 00:05:11 2020

@author: guill
"""

import matplotlib.pyplot as plt
import pandas as pd
# pd.plotting.register_matplotlib_converters() ? needed for ?
# %matplotlib inline
import seaborn as sns

fifa_filepath = "../input/fifa.csv"
# Read the file into a variable fifa_data
fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)

# Set the width and height of the figure
plt.figure(figsize=(16,6))

# Add title
plt.title("Daily Global Streams of Popular Songs in 2017-2018")

# Line chart showing how FIFA rankings evolved over time 
sns.lineplot(data=fifa_data)
# apparently takes index as x axis ref (date here)

# Add label for horizontal axis
plt.xlabel("Date")

#%%  bar chart

df = pd.read_csv()

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=df.index, y=df['NK'])
# apparently here no bins, y = 1 value per bar (NOT HISTOGRAM)
# compare unique values (diff box plot because no range) where order does not matter

# Bar chart showing average score for racing games by platform
sns.barplot(x=df['Racing'], y=df.index)
# can do horizontal bar plot for having the text labels straight

#%% heatmap:
    
# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=df, annot=True)
# todo 2d plots with numeric values # unique matrix of values
# directly works with pandas : 2D array, col and rows

#%% scatter plot

sns.scatterplot(x=df['bmi'], y=df['charges'])
# plot large dataset, along two features numeric

# same than scatter but add regression line
sns.regplot(x=df['bmi'], y=df['charges'])

#%% color coded scatter plot : three variables (color, not more than 3, boolean ideally)

sns.scatterplot(x=df['bmi'], y=df['charges'], hue=df['smoker'])

# same and add one reg line per color
sns.lmplot(x="bmi", y="charges", hue="smoker", data=df)

#%% swarm plot :  categorical scatter plot

# no overlap in the dots, x axis categorical

sns.swarmplot(x=df['smoker'],y=df['charges'])

#%% histogram

# Histogram 
sns.distplot(a=df['Petal Length (cm)'], kde=False)

#%% KDE

# KDE plot 
sns.kdeplot(data=df['Petal Length (cm)'], shade=True)

# tres classe 2D KDE plot

# 2D KDE plot
sns.jointplot(x=df['Petal Length (cm)'], y=df['Sepal Width (cm)'], kind="kde")

# can do hold on plots
# KDE plots for each species
sns.kdeplot(data=df['Petal Length (cm)'], label="Iris-setosa", shade=True)
sns.kdeplot(data=df['Petal Length (cm)'], label="Iris-versicolor", shade=True)
sns.kdeplot(data=df['Petal Length (cm)'], label="Iris-virginica", shade=True)
