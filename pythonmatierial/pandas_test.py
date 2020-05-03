













import pandas as pd
import numpy as np


#Creating a basic dataframe
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, 1, 5, 2, 4.5, 7.4, 4, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

df=pd.DataFrame(data)


#reading csv files
df_temp = pd.read_csv("/home/nikhil/in_postcodes.csv")

#change the index
df.index=['a','b','c','d','e','f','g','h','i','j']

# resetting index
df.reset_index(drop =True, inplace=True)  # the index is reset with 0,1,2... x
# Note: if drop=True is not added, the old index is added as a column to the dataframe

#Getting list of columns names
df.columns.values
df.columns.tolist()

#getting some info about the dataframe and values in the dataframe
df.info()
df.describe()
df.shape
df.dtypes

#reading specific rows of the dataframe

df.head()  
df.iloc[7]
df.iloc[0:7]

# selecting only a few specific columns / rows

# by row / column index
df.iloc[3:7,[0,3]]
df.iloc[:,[0,3]]

#by row index / column names
df.loc[:,['age','visits']]


#selecting rows based on specific criteria
df[df['animal']=='cat']
df[df['age']>5]
df[(df['animal'] == 'cat') & (df['age'] < 3)]


#Updating values of specific rows
#df['visits'][df['visits']==3]='Nan'
df['animal'] = df['animal'].replace('snake', 'python')

#sums, mean, count
df['animal'].value_counts()
df['visits'].sum()
df['visits'].cumsum()
df['visits'].mean()

#sorting values
df.sort_values(by=['age', 'visits'], ascending=[False, True])

#Group by
df.groupby('animal')['age'].mean()

#dropping rows / columns
#note - rows can be dropped only using an index or by subsetting
df=df[df['animal']=='snake']
df.drop('animal',axis=1)
#df.dropna(inplace=True)
#df.dropna(inplace=True, how='all')
#df.dropna(how = 'any',thresh=3)

#Adding a column
df['New']=df['animal']+ df['priority']

#Pivot table in Pandas
#df.pivot_table(index='animal', columns='visits', values='age', aggfunc='mean')

#filling in missing values
df['age'].fillna(4.0, inplace=True)

#finding number of missing values
df.apply(lambda x: sum(x.isnull()),axis=0) #operation applied acorss rows
df.apply(lambda x: sum(x.isnull()),axis=1) # operation applied across columns

# Merging datasets
data1 = {'sl_no': ['1', '2', '3', '4', '5'],
        'state': ['Karnataka', 'Kerala', 'Rajasthan', 'Bengal', 'Maharashtra'], 
        'capital': ['Bangalore', 'Trivandrum', 'Jaipur', 'Kolkata', 'Mumbai']}
df_data1 = pd.DataFrame(data1, columns = ['sl_no', 'state', 'capital'])


data2 = {'sl_no': ['1', '2', '3'],
        'state': ['Telangana', 'Tamilnadu', 'Assam'], 
        'capital': ['Hyderabad', 'Chennai', 'Dispur']}
df_data2 = pd.DataFrame(data2, columns = ['sl_no', 'state', 'capital'])

pd.concat([df_data1, df_data2], axis=1)  # join across columns
pd.concat([df_data1, df_data2], axis=0)  #join across rows























