import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.random_forest import RandomForestRegressor
import pickle

df = pd.read_csv(os.getcwd() + "/dataset/Gold_Price.csv")

columns = ["price"]
df = df[columns]

X = df.iloc[:, 0:4]
Y = df.iloc[:, 4:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

lr = LinearRegression()
lr.fit(X_train, Y_train)

rf = RandomForestRegressor()
rf.fit(X_train, Y_train)

pickle.dump(lr, open(os.getcwd() + "/models/model.pkl", "wb"))
