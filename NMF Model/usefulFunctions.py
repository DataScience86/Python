# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 21:13:53 2019

@author: Mohit Sharma

Collection of useful functions
"""
#%%
# function adds the two numbers which are separated by a defined interval. 
# first number is the last number in the dataset 
# second number is the number at the said interval
def interval_add(data, varName, interval = 2):
    nrow = data.shape[0] - 1
    value = data.loc[[nrow, (nrow - interval)], varName]
    return value.sum()

#%%

# A different version of above function
# function adds the two numbers which are separated by a defined interval in series. 
# first number is the last number in the dataset 
# second number is the number at the said interval
def interval_add_series(Series, interval = 2):
    Series = df.V4
    nelem = len(Series) - 1
    value = Series[[nelem, (nelem - interval)]]
    return value.sum()

# Applying the function of grouped data
# Group df by df.V4, then apply a interval_add_series function
df.groupby('ID')['V4'].apply(lambda x: interval_add_series(x, 2))
#%%