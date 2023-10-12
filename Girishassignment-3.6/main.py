import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('50_Startups.csv') 
df.head()
df.isnull().sum()
sns.pairplot(df)
# Perform one-hot encoding for the "state" column
df = pd.get_dummies(df, columns=['State'], drop_first=True)
df
# Define the predictor variables (X) excluding the target variable and the original "state" column
X = df.drop(['Profit'], axis=1)
# Define the target variable (y)
y = df['Profit']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Calculate the Mean Squared Error (MSE) to evaluate the model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
# Define a dictionary with input values for prediction
new_data_dict = {
    'R&D Spend': [120542.52,123334.88],
    'Administration': [148718.95, 108679.17],
    'Marketing Spend': [311613.29, 304981.62],
    'State_Florida': [0,1],  # Include the state columns (Florida=0, New York=1)
    'State_New York': [0,0]
}

# Create a DataFrame from the dictionary
new_data = pd.DataFrame(new_data_dict)

# Make predictions
predicted_profit = model.predict(new_data)

# Print the predicted profit values
for i, pred in enumerate(predicted_profit):
    print(f'Prediction {i + 1}: {pred:.2f}')

# Define a dictionary with input values for prediction
new_data_dict = {
    'R&D Spend': [1736783,1233333],
    'Administration': [1482, 10222],
    'Marketing Spend': [3116333, 32322],
    'State_Florida': [0,1],  # Include the state columns (Florida=0, New York=1)
    'State_New York': [0,0]
}

# Create a DataFrame from the dictionary
new_data = pd.DataFrame(new_data_dict)

# Make predictions
predicted_profit = model.predict(new_data)

# Print the predicted profit values
for i, pred in enumerate(predicted_profit):
    print(f'Prediction {i + 1}: {pred:.2f}')

   
