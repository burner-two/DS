import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import tree

# Load iris data
iris_data = load_iris()
iris = pd.DataFrame(iris_data.data)

print("Feature Names:", iris_data.feature_names)

X = iris_data.data
print(X)

Y = iris_data.target
print(Y)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=50, test_size=0.3)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Train model
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=100)
clf.fit(X_train, y_train)

# Predict
Y_pred = clf.predict(X_test)
print(Y_pred)

# Accuracy
print("Accuracy:", accuracy_score(y_test, Y_pred))

# Confusion matrix plot using modern approach
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
cm = confusion_matrix(y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# Show the plot
plt.show()

tree.plot_tree(clf)
plt.show()