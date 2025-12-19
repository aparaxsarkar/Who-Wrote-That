import pandas as pd

df = pd.read_csv("raid_data/extra.csv")

print("All models:")
print(df["model"].unique())

print("\nModel counts:")
print(df["model"].value_counts())

print("\nAll domains:")
print(df["domain"].unique())

print("\nDomain counts:")
print(df["domain"].value_counts())
