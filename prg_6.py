import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Handle missing data (Iris dataset doesn't have missing values, but we'll simulate some)
def preprocess_dataset(df):
    df.iloc[::10, 0] = float('NaN')
    # Simulate missing values in the first column
    imputer = SimpleImputer(strategy="mean")
    df[df.columns] = imputer.fit_transform(df[df.columns])
    # Since Iris dataset doesn't have categorical variables,
    # we'll skip this step
    # Encode categorical variable (if applicable)
    
# Perform feature scaling
    scaler = StandardScaler()
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
    return df

# Preprocessed Iris dataset
preprocessed_df = preprocess_dataset(iris_df)
print("Preprocessed dataset:")
print(preprocessed_df.head())
2j