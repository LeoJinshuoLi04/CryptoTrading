import time
import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError


def fetch_bitcoin_google_trends(save_path="bitcoin_trends.csv"):
    pytrends = TrendReq(hl="en-US", tz=0)
    kw_list = ["bitcoin"]
    date_ranges = pd.date_range(start="2018-01-01", end="2022-12-31", freq="240D")
    date_ranges = list(date_ranges) + [pd.Timestamp("2022-12-31")]

    all_data = []
    try:
        saved_df = pd.read_csv(save_path, parse_dates=["Date"])
        if not saved_df.empty:
            last_date = saved_df["Date"].max()
            print(f"Resuming from last saved date: {last_date}")
            resume_index = 0
            for i in range(len(date_ranges) - 1):
                if date_ranges[i] > last_date:
                    resume_index = i
                    break
            else:
                print("All data already fetched.")
                return saved_df
            all_data.append(saved_df)
        else:
            resume_index = 0
    except FileNotFoundError:
        resume_index = 0

    for i in range(resume_index, len(date_ranges) - 1):
        start = date_ranges[i].strftime("%Y-%m-%d")
        end = date_ranges[i + 1].strftime("%Y-%m-%d")
        timeframe = f"{start} {end}"

        while True:
            try:
                print(f"Fetching trends for: {timeframe}")
                pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo="", gprop="")
                df = pytrends.interest_over_time()

                if df.empty:
                    print(f"No data returned for {timeframe}. Skipping.")
                    break

                all_data.append(df)
                full_df = pd.concat(all_data)
                full_df = full_df[~full_df.index.duplicated()]
                full_df = full_df.rename(columns={"bitcoin": "GoogleTrends"})
                full_df = full_df.drop(columns=["isPartial"])
                full_df = full_df.reset_index().rename(columns={"date": "Date"})
                full_df.to_csv(save_path, index=False)

                time.sleep(5)
                break

            except TooManyRequestsError:
                print("Too many requests error caught, sleeping for 60 seconds...")
                time.sleep(60)
            except Exception as e:
                print(f"Unexpected error: {e}. Retrying in 60 seconds...")
                time.sleep(60)

    final_df = pd.concat(all_data)
    final_df = final_df[~final_df.index.duplicated()]
    final_df = final_df.rename(columns={"bitcoin": "GoogleTrends"})
    final_df = final_df.drop(columns=["isPartial"])
    final_df = final_df.reset_index().rename(columns={"date": "Date"})
    return final_df


if __name__ == "__main__":
    fetch_bitcoin_google_trends()
