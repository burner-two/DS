#Regression and its type 
#Implement Simple Linear Regression using a given dataset.
#Explore and interpret the Regression Model coeffiecients and goodness-of-fit measures.

#Implementing simple linear regression

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

# Loading Dataset
data_set = pd.read_csv('/home/shan/Desktop/DSA/SalaryData.csv')


# Independent variable (all except last column)
X = data_set.iloc[:, :-1].values

# Dependent variable (last column)
Y = data_set.iloc[:, -1].values

# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Fitting the Simple Linear Regression model to the training dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Prediction of Test and Training set result
y_pred = regressor.predict(X_test)
x_pred = regressor.predict(X_train)

# Evaluation of Model
print('R squared: {:.2f}'.format(regressor.score(X, Y) * 100))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
#either use the below line to get Root Mean Squared Error or use the mse , rmse value
#print('Root Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred, squared=False))

mse = metrics.mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)


# Print the coefficients and intercept
print('Coefficients:', regressor.coef_)
print('Intercept:', regressor.intercept_)

plt.scatter(X_train,Y_train,color= "green")
plt.plot(X_train,x_pred,color= 'red')
plt.title("Salary vs Expriecne ")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()

#visulize the Test set result 
plt.scatter(X_test,Y_test,color= "blue")
plt.plot(X_train,x_pred,color= 'red')
plt.title("Salary vs Expriecne ")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()



