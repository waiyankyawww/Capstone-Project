# import pandas
# import os

# df = pandas.read_csv(os.getcwd() +"/dataset/Gold_Price_Data.csv")
# # for i in xrange(len(df)):
# #    df.loc[i,'Day'] = datetime.strptime(df.loc[i,'Date'], '%Y-%m-%d').day


# df['Year']=[d.split('-')[0] for d in df.Date]
# df['Month']=[d.split('-')[1] for d in df.Date]
# df['Day']=[d.split('-')[2] for d in df.Date]

# df.head(5)



#after pip install pandas, import the module
from optparse import Values
import pandas as pd 

#Read your input csv file
df = pd.read_csv('/Users/waiyankyaw/Desktop/SOA Project/dataset/Gold_Price_Data.csv')

#Convert all the string values of date-time column to datetime objects
df['date-time-obj'] = pd.to_datetime(df['Date'])
# df['Price'] = df['Price']

#Create two new columns with date-only and time-only values
df['day'] = df['date-time-obj'].dt.day
df['month'] = df['date-time-obj'].dt.month
df['year'] = df['date-time-obj'].dt.year

# if (df['day'].all() <= 12 and df['month'].all() <= 12):
#     df['day'].values = df['month'].values
#     df['month'].values = df['day'].values

# df['price'] = df['date-time-obj'].dt.

#Deleted temporarily created column
del df['date-time-obj']

#Save your final data to a new csv file
df.to_csv('dataset/Gold_Price_Data_Final.csv', index=False)