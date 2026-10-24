import pandas as pd

df = pd.read_csv("dataset/fake reviews dataset.csv")

print("\nFIRST FIVE ROWS:\n")
print(df.head())

print("\n-------------------------------------")

print("\nCOLUMN NAMES:\n")
print(df.columns)

print("\n-------------------------------------")

print("\nDATASET SHAPE:\n")
print(df.shape)

print("\n-------------------------------------")

print("\nMISSING VALUES:\n")
print(df.isnull().sum())

print("\n-------------------------------------")

print("\nUNIQUE LABELS:\n")

# This will help us find which column contains the labels
for column in df.columns:
    print(f"\n{column}")
    print(df[column].value_counts().head())