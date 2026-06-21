import pandas as pd

df = pd.read_csv(
    r"C:\Users\AMD\Downloads\CUAD_v1\CUAD_v1\master_clauses.csv"
)

print("Renewal Term:")
print(df["Renewal Term"].iloc[0])

print("\nRenewal Term-Answer:")
print(df["Renewal Term-Answer"].iloc[0])

print("\nMost Favored Nation:")
print(df["Most Favored Nation"].iloc[0])

print("\nMost Favored Nation-Answer:")
print(df["Most Favored Nation-Answer"].iloc[0])