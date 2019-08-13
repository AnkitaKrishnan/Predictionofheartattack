# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:03:52 2019

@author: ankit
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('heart.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# For making polynomial linear regression model
A = dataset.iloc[:, 4:5].values
b = dataset.iloc[:, 3].values

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
A_poly = poly_reg.fit_transform(A)
poly_reg.fit(A_poly, b)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(A_poly, b)

# Visualising the Polynomial Regression results
plt.scatter(A, b, color = 'red')
plt.plot(A, lin_reg_2.predict(poly_reg.fit_transform(A)), color = 'blue')
plt.title('Age & Heart Attack (Polynomial Regression)')
plt.xlabel('Chol')
plt.ylabel('TrestBps')
plt.show()

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred1 = np.round(y_pred, 0)

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_pred))

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
rmse_val = rmse(np.array(y_test), np.array(y_pred))
print("rms error is: " + str(rmse_val))