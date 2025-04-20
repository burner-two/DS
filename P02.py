import pandas as pd

# Reading the Student CSV
data = pd.read_csv("Student.csv")

# Removing rows with missing values
clean_data = data.dropna(axis=0, how='any')
print("Cleaned Data (no missing values):")
print(clean_data)

# Filling missing values with 0
filled_data = data.fillna(0)
print("\nFilled Data (missing values replaced with 0):")
print(filled_data)

# Reading car data
car_data = pd.read_csv("car.csv")

# Replace outliers in 'Sell Price ($)' with the median
sell_price_col = 'Sell Price ($)'
buy_price_col = 'Buy Price ($)'
median_value = car_data[sell_price_col].median()
mean = car_data[sell_price_col].mean()
std = car_data[sell_price_col].std()

upper_threshold = mean + 2 * std
lower_threshold = mean - 2 * std

car_data[sell_price_col] = car_data[sell_price_col].apply(
    lambda x: median_value if x > upper_threshold or x < lower_threshold else x
)

print("\nCar Data (after replacing outliers in Sell Price):")
print(car_data)

#filering 
print("filtering start from here ")
filled_data1 = car_data[car_data[sell_price_col]>30000]
print(filled_data1)

#sorting the data by decending order
print("sorting start from here ")

sorted_data = car_data.sort_values(by=sell_price_col,ascending=False)
print(sorted_data)


#Grouping and calculating mean for numeric colums 
print("grouping start from here")
numeric_column = [sell_price_col,buy_price_col]
group_data  = car_data.groupby('Make')[numeric_column].mean()
print(group_data)


# Reading JSON data
print("\nFrom here JSON data starts:")
jdata = pd.read_json("animal.json")
print(jdata)
