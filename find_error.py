import pandas as pd
df = pd.read_pickle("prepared/prepared_month5_ALL_12M.pkl")
t = "Motor_Temperature"
df["abs_change"] = (df[f"{t}_future"] - df[t]).abs()

b = pd.cut(df["abs_change"], [-0.1, 1, 3, 6, 10, 20, 30, 200])
g = df.groupby(b, observed=True)["abs_change"].agg(rows="size", total="sum")
g["% of rows"] = 100 * g["rows"] / g["rows"].sum()
g["% of total error"] = 100 * g["total"] / g["total"].sum()
print(g)

cols = ["device_id", "_t", t, f"{t}_future", "abs_change"] + [c for c in df.columns if c.startswith("ignition")]
print(df.sort_values("abs_change", ascending=False).head(20)[cols])

v = df[(df["device_id"].astype(str) == "861409075137969") & (df["_t"] >= "2026-05-22 11:10") & (df["_t"] <= "2026-05-22 12:00")]
print(v[["_t", t, f"{t}_future"]].iloc[::6].to_string())