#Logistic Regression and Decision Tree
#Build a Logistic Regression Model to predict a binary outcome.
#Evaluate the model’s performance using classification metrics.
# Logistic Regression

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/home/shan/Desktop/DSA/Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Encoding categorical data (Gender)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])  # Female -> 0, Male -> 1

# Splitting the Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


print(X_train)
print(Y_train)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)

# Predicting a New Result
# Predicting result for certain age and salary, person will purchase or not purchase

print(classifier.predict(sc.transform([[12345678, 1, 30, 87000]])))


# Predicting the test results
y_pred = classifier.predict(X_test)
print(np.concatenate(
    (y_pred.reshape(len(y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, y_pred)
print(cm)
accuracy = accuracy_score(Y_test, y_pred)
print(accuracy)
