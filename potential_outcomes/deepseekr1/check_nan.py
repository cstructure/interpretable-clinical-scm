import pandas as pd
from utils import load_data

# Load the data
df = load_data("sdy1662.csv")

# Check for any remaining NaN values
print("Total NaN values in DataFrame:", df.isnull().sum().sum())

# If there are NaN values, print the columns that have them
if df.isnull().sum().sum() > 0:
    nan_cols = df.columns[df.isnull().any()].tolist()
    print("Columns with NaN:", nan_cols)
    print("Number of NaN per column:")
    print(df[nan_cols].isnull().sum())
else:
    print("No NaN values found.")
