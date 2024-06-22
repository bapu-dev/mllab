import pandas as pd
def explore_dataset(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv("C:/Users/CSC/Documents/ML Lab/Programs/iris.csv")
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel("C:/Users/CSC/Documents/ML Lab/Programs/iris.xlsx")
    else:
        print("File type not supported")
        return
    print("Dataset Information:")
    print(df.info()) 
    print("\n First Few Rows of the Dataset:")
    print(df.head())
    print("Summary Statistics:")
    print(df.describe())
    print("\n Unique Values for Categorical Columns:") 
    for column in df.select_dtypes(include='object').columns:
        print(f"{column}: {df[column].unique()}")
explore_dataset("C:/Users/CSC/Documents/ML Lab/Programs/iris.csv")
