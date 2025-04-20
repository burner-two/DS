# Implement Multiple Linear Regression and assess the impact of additional predictors.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data_set = pd.read_csv('/home/shan/Desktop/DSA/50_Startups.csv')

# Independent variables (all except last column)
X = data_set.iloc[:, :-1].values

# Dependent variable (last column)
Y = data_set.iloc[:, -1].values

# Encode categorical data (state column)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Train the Multiple Linear Regression model
regression = LinearRegression()
regression.fit(X_train, Y_train)

# Predict the Test set results
y_pred = regression.predict(X_test)
print("Predictions:", y_pred)

# Create DataFrame for actual vs predicted values
mir_diff = pd.DataFrame({'Actual Value': Y_test, 'Predicted Value': y_pred})
print(mir_diff.head())

# Evaluation of Model
print('R squared: {:.2f}'.format(regression.score(X, Y) * 100))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
#Mannulay compute  Root Mean Squared Error 
# RMSE = sqrt(MSE)
mse = metrics.mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)
#or use the below method 
#print('Root Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred, squared=False))
