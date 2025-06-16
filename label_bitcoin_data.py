import pandas as pd

btc_df = pd.read_csv("bitcoin_data_raw/bitcoin_data.csv", parse_dates=["Date"])
trends_df = pd.read_csv("bitcoin_data_raw/bitcoin_trends.csv", parse_dates=["Date"])
btc_df["Date"] = btc_df["Date"].dt.normalize()
trends_df["Date"] = trends_df["Date"].dt.normalize()
merged_df = pd.merge(btc_df, trends_df, on="Date", how="left")
merged_df.to_csv("bitcoin_data_labeled.csv", index=False)
