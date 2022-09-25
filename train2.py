import os
import pickle
import pandas as pd
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# from sklearn.random_forest import RandomForestRegressor

df = pd.read_csv(os.getcwd() + "/dataset/Gold_Price_Data_Final.csv")


columns = ['Day', 'Month', 'Year']
df = df[columns]
df.loc[:,columns]

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

# X.shape()
# Y.shape()
# X = df[['Date', 'Price']]
X = df.dropna()
# Y = df['Date', 'Price']
Y = df.dropna()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=101)

linear_R = LinearRegression()
linear_R.fit(X_train, Y_train)

# Random_F.fit(X_train, Y_train)
Random_F = RandomForestRegressor()
# Random_F.fit(X_train, Y_train.values.ravel())
Random_F.fit(X_train, Y_train)

pickle.dump(linear_R, open(os.getcwd() + "/models/LinearRegression.pkl", "wb"))
pickle.dump(Random_F, open(os.getcwd() + "/models/RandomForest.pkl", "wb"))
