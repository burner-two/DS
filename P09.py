import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder



# Step 1: Load dataset
dataset = pd.read_csv('/home/shan/Desktop/DSA/wine.csv')

# Step 2: Convert object (string) columns to numbers using LabelEncoder
for col in dataset.columns:
    if dataset[col].dtype == 'object':
        dataset[col] = LabelEncoder().fit_transform(dataset[col])

# Step 3: Now fill missing values with the mean (now safe since all columns are numeric)
dataset.fillna(dataset.mean(), inplace=True)

# Step 4: Separate X and y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Step 5: Split and scale
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step 6: PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# Visualize
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA - Wine Dataset')
plt.show()
