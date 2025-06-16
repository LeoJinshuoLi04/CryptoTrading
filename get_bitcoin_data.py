import ccxt
import pandas as pd
import requests
from tqdm import tqdm
import time
from datetime import datetime

START_DATE = "2018-01-01T00:00:00Z"
END_DATE = "2022-12-31T23:00:00Z"
INTERVAL = "1h"
SYMBOL = "BTC/USDT"


def fetch_binance_ohlcv(symbol=SYMBOL, interval=INTERVAL, since=None):
    exchange = ccxt.binance()
    since_ts = exchange.parse8601(since)
    end_ts = exchange.parse8601(END_DATE)

    all_ohlcv = []
    limit = 1000
    print("Fetching Binance OHLCV data...")

    while since_ts < end_ts:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, since=since_ts, limit=limit)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        since_ts = ohlcv[-1][0] + 1
        time.sleep(0.1)

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop("timestamp", axis=1)
    df.to_csv("btc_ohlcv.csv", index=False)
    return df


def fetch_coinmetrics_metric(asset="btc", metric="AdrActCnt", frequency="1d"):
    url = f"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {
        "assets": asset,
        "metrics": metric,
        "frequency": frequency,
        "start_time": START_DATE,
        "end_time": END_DATE,
        "page_size": 10000,
    }
    print(f"Fetching Coin Metrics: {metric}")
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()["data"]
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["time"])
    df = df.rename(columns={metric: metric})
    df = df[["Date", metric]].set_index("Date")
    return df


def get_all_metrics():
    metrics = {
        "AdrActCnt": "AdrActCnt",
        "TxTfrValAdjUSD": "TxVal",
        "TxCnt": "TxCnt",
        "FeeMeanUSD": "FeeMean",
        "HashRate": "HashRate",
    }
    dfs = []
    for api_name, label in tqdm(metrics.items()):
        df = fetch_coinmetrics_metric(metric=api_name)
        df = df.rename(columns={api_name: label})
        dfs.append(df)
    combined = pd.concat(dfs, axis=1)
    combined.to_csv("coinmetrics_combined.csv")
    return combined


def merge_and_save(ohlcv_df, metrics_df):
    ohlcv_df["Date"] = pd.to_datetime(ohlcv_df["Date"]).dt.tz_localize(None)
    metrics_df.index = pd.to_datetime(metrics_df.index).tz_localize(None)
    df = ohlcv_df.set_index("Date").join(metrics_df, how="left")
    df = df.reset_index().sort_values("Date")
    df = df.dropna()
    df.to_csv("bitcoin_data.csv", index=False)


def main():
    btc_ohlcv = fetch_binance_ohlcv(since=START_DATE)
    btc_metrics = get_all_metrics()
    merge_and_save(btc_ohlcv, btc_metrics)


if __name__ == "__main__":
    main()
