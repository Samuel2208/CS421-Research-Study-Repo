import pandas as pd

csv_files = [
    "../Dataset/iemocap_session1.csv",
    "../Dataset/iemocap_session2.csv",
    "../Dataset/iemocap_session3.csv",
    "../Dataset/iemocap_session4.csv",
    "../Dataset/iemocap_session5.csv",
]

unique_emotions = set()

for file in csv_files:
    df = pd.read_csv(file)
    if "emotion" in df.columns:
        emotions = df["emotion"].unique()
    else:
        possible_cols = [col for col in df.columns if "emotion" in col.lower()]
        if possible_cols:
            emotions = df[possible_cols[0]].unique()
        else:
            continue
    unique_emotions.update(emotions)

unique_emotions_list = sorted(list(unique_emotions))
print(unique_emotions_list)
