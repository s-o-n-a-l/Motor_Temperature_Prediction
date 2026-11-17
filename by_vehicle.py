import pandas as pd
df = pd.read_pickle("prepared/prepared_month5_ALL_12M.pkl")
t = "Motor_Temperature"
df["abs_change"] = (df[f"{t}_future"] - df[t]).abs()
print(df.groupby("device_id", observed=True)["abs_change"]
        .agg(["mean", "median", "max", "size"])
        .sort_values("mean", ascending=False))