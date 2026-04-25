import pandas as pd

csv_path = input("Enter CSV path: ").strip()
df = pd.read_csv(csv_path)
counts = df['target_emotion'].value_counts()
print(counts)