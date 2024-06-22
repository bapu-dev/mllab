import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_dataset(file_path):
    df = pd.read_csv(file_path)
    
    print("Pairplot of the Dataset")
    sns.pairplot(df)
    plt.show()
    
    if df.iloc[:, 0].dtype == 'object':
        sns.countplot(x=df.columns[0], data=df)
        plt.title("Bar Chart of Categorical Column")
        plt.xlabel(df.columns[0])
        plt.ylabel("Count")
        plt.show()
    else:
        print("No categorical column found to plot bar chart.")
        
file_path = 'iris.csv'
visualize_dataset("C:/Users/CSC/Documents/ML Lab/Programs/iris.csv")
