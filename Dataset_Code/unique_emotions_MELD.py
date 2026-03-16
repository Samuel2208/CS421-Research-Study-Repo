import pandas as pd

df = pd.read_csv('../Dataset/MELD.csv')

print(df['Emotion'].unique())