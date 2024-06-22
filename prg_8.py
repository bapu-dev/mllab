import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Iris dataset
iris = load_iris()
x = iris.data
y = iris.target

# Create a DataFrame for visualization (optional)
iris_df = pd.DataFrame(data=x, columns=iris.feature_names)
iris_df['target'] = y

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)

# Make predictions on the test set
y_pred = linear_regression.predict(x_test)

# Calculate Mean Squared Error and R-squared score
nse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", nse)
print("R-Squared Score:", r2)
