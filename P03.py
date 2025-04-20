# Importing Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define the data
data = {
    'Product': [
        'Apple_Juice', 'Banana_Smoothie', 'Orange_Jam', 'Grape_Jelly',
        'Kiwi_Juice', 'Mango_Pickle', 'Pineapple_Sorbet', 'Strawberry_Yoghurt',
        'Blueberry_Pie', 'Cherry_Salsa'
    ],
    'Category': [
        'Apple', 'Banana', 'Orange', 'Grape', 'Kiwi', 'Mango',
        'Pineapple', 'Strawberry', 'Blueberry', 'Cherry'
    ],
    'Sales': [1200, 1700, 2200, 1400, 2000, 1000, 1500, 1800, 1300, 1600],
    'Cost':   [600,  850, 1100,  700, 1000,  500,  750,  900,  650,  800],
    'Profit': [600,  850, 1100,  700, 1000,  500,  750,  900,  650,  800]
}

# Convert to DataFrame
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

# Feature Scaling
numeric_columns = ['Sales', 'Cost', 'Profit']

# Standardization
scaler_std = StandardScaler()
df_scaled_std = pd.DataFrame(scaler_std.fit_transform(df[numeric_columns]), columns=[col + "_std" for col in numeric_columns])

# Normalization
scaler_norm = MinMaxScaler()
df_scaled_norm = pd.DataFrame(scaler_norm.fit_transform(df[numeric_columns]), columns=[col + "_norm" for col in numeric_columns])

print("\nStandardized Features:\n", df_scaled_std)
print("\nNormalized Features:\n", df_scaled_norm)

# Combine scaled features (standardized version) with original categorical data
df_scaled = pd.concat([df_scaled_std, df.drop(numeric_columns, axis=1)], axis=1)
print("\nDataset after Feature Scaling (Standardized + Categorical):\n", df_scaled)

# Feature Dummification
categorical_columns = ['Product', 'Category']

# Create ColumnTransformer for OneHotEncoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns)
    ],
    remainder='passthrough'  # keep the other columns (scaled numerical)
)

# Apply transformation
df_final = pd.DataFrame(preprocessor.fit_transform(df_scaled))

print("\nDataset after Feature Dummification (One-Hot Encoding):\n", df_final)
