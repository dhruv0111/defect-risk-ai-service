import pandas as pd

datasets = ["cm1.csv", "kc1.csv", "pc1.csv", "jm1.csv"]

for file in datasets:
    df = pd.read_csv(f"data/{file}")
    print(f"\n{file}")
    print("Columns:")
    print(df.columns.tolist())