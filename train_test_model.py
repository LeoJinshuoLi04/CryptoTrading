import os
from lstm_model import LSTMModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from sklearn.metrics import confusion_matrix


def load_data(
    altcoin_path,
    bitcoin_path=None,
    bitcoin_feature=None,
    bitcoin_fillna=0,
    window_size=30,
    ret_threshold=0.01,
    train_frac=0.8,
):
    df = pd.read_csv(altcoin_path, parse_dates=["Date"])

    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "RSI",
        "ULTOSC",
        "boll",
        "pct_change",
        "zsVol",
        "MA_Ratio",
        "PR_MA_Ratio",
        "Asset_name",
    ]

    df = df[["Date"] + feature_cols].sort_values(["Asset_name", "Date"]).reset_index(drop=True)

    if bitcoin_path and bitcoin_feature:
        btc_df = pd.read_csv(bitcoin_path, parse_dates=["Date"])
        btc_df["Date"] = btc_df["Date"].dt.normalize()
        btc_df = btc_df[["Date", bitcoin_feature]].set_index("Date")

        merged_dfs = []

        for asset in df["Asset_name"].unique():
            asset_df = df[df["Asset_name"] == asset].copy()
            asset_df["Date"] = asset_df["Date"].dt.normalize()

            merged = asset_df.set_index("Date").join(btc_df, how="left")

            merged[bitcoin_feature] = merged[bitcoin_feature].fillna(bitcoin_fillna)

            merged.reset_index(inplace=True)

            merged["Asset_name"] = asset

            merged_dfs.append(merged)

        df = pd.concat(merged_dfs, ignore_index=True)

        print(f"Combined DataFrame preview with Bitcoin feature '{bitcoin_feature}':")

    all_data_per_coin = {}
    all_X, all_y = [], []

    for asset in df["Asset_name"].unique():
        if asset == "BTCUSDT":
            continue
        asset_df = df[df["Asset_name"] == asset].copy().reset_index(drop=True)

        features_to_use = [col for col in feature_cols if col != "Asset_name"]
        if bitcoin_feature and bitcoin_feature not in features_to_use:
            features_to_use.append(bitcoin_feature)

        features = asset_df[features_to_use]

        features = features.replace([np.inf, -np.inf], 0)
        features = features.dropna()

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        close_prices = asset_df["Close"].values

        X, y = [], []

        for i in range(len(scaled_features) - window_size - 1):
            X_seq = scaled_features[i : i + window_size]
            ret = (close_prices[i + window_size + 1] - close_prices[i + window_size]) / close_prices[i + window_size]

            if ret > ret_threshold:
                y_seq = 1
            elif ret < -ret_threshold:
                y_seq = -1
            else:
                y_seq = 0

            X.append(X_seq)
            y.append(y_seq)

        all_X.extend(X)
        all_y.extend(y)

        y = np.array(y)
        X = np.array(X)

        n_train = int(len(X) * train_frac)
        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:], y[n_train:]
        test_prices = close_prices[window_size + 1 + n_train : len(X) + window_size + 1]

        if len(X_train) > 500:
            all_data_per_coin[asset] = (X_train, y_train, X_test, y_test, test_prices)

    all_y = np.array(all_y)
    label_counts = Counter(all_y)
    print("Label counts before split:")
    print(f"  -1: {label_counts.get(-1, 0)}")
    print(f"   0: {label_counts.get(0, 0)}")
    print(f"   1: {label_counts.get(1, 0)}")

    n_all = len(all_X)
    n_train = int(n_all * train_frac)
    print(f"Training samples: {n_train}, Testing samples: {n_all - n_train}")
    print(f"Number of coins: {len(all_data_per_coin)}")
    return all_data_per_coin


def train_and_test_models():
    bitcoin_only_features = [
        None,
        "AdrActCnt",
        "TxVal",
        "TxCnt",
        "FeeMean",
        "HashRate",
        "GoogleTrends",
    ]

    os.makedirs("models", exist_ok=True)
    os.makedirs("models/v2", exist_ok=True)

    for feature in bitcoin_only_features:
        print(f"\n=== Training LSTM model with Bitcoin feature: {feature} ===")

        data_per_coin = load_data(
            altcoin_path="processed_data/Altcoin.csv",
            bitcoin_path="processed_data/bitcoin_data_labeled.csv",
            bitcoin_feature=feature,
        )

        model_filename = f"models/v2/model_{feature if feature else 'no_btc'}.h5"

        lstm_model = LSTMModel(input_shape=next(iter(data_per_coin.values()))[0].shape[1:])

        if os.path.exists(model_filename):
            print(f"Loading saved model from {model_filename}")
            lstm_model.load(model_filename)
        else:
            print("Training LSTM model...")

            X_all = np.concatenate([X_train for X_train, _, _, _, _ in data_per_coin.values()])
            y_all = np.concatenate([y_train for _, y_train, _, _, _ in data_per_coin.values()])

            lstm_model.train(X_all, y_all)

            lstm_model.save(model_filename)
            print(f"Saved trained model to {model_filename}")

        all_X_test = np.concatenate([splits[2] for splits in data_per_coin.values()])
        all_y_test = np.concatenate([splits[3] for splits in data_per_coin.values()])

        print("Testing LSTM model on concatenated test sets...")
        y_pred = lstm_model.predict(all_X_test)

        print(f"Accuracy: {accuracy_score(all_y_test, y_pred):.4f}")
        print("Classification report:")
        print(classification_report(all_y_test, y_pred))

        print("\nDummy classifier baseline:")
        all_X_train = np.concatenate([splits[0] for splits in data_per_coin.values()])
        all_y_train = np.concatenate([splits[1] for splits in data_per_coin.values()])
        lstm_model.dummy_train(all_X_train, all_y_train)
        dummy_preds = lstm_model.dummy_predict(all_X_test)
        print(f"Dummy Accuracy: {accuracy_score(all_y_test, dummy_preds):.4f}")
        print(classification_report(all_y_test, dummy_preds))

    print("\nAll models processed.")


def train_and_test_models_individual():

    os.makedirs("models", exist_ok=True)
    os.makedirs("models/adrActCnt", exist_ok=True)
    data_per_coin = load_data(
        altcoin_path="processed_data/Altcoin.csv",
        bitcoin_path="processed_data/bitcoin_data_labeled.csv",
        bitcoin_feature="AdrActCnt",
    )

    for coin, (X_train, y_train, X_test, y_test, test_prices) in data_per_coin.items():
        print(f"\n=== Training LSTM model for coin: {coin} ===")

        model_filename = f"models/adrActCnt/model_{coin}.h5"

        lstm_model = LSTMModel(input_shape=X_train.shape[1:])

        print("Training LSTM model...")

        lstm_model.train(X_train, y_train)

        lstm_model.save(model_filename)
        print(f"Saved trained model to {model_filename}")

        print("Testing LSTM model on test set...")
        y_pred = lstm_model.predict(X_test)

        print("\nDummy classifier baseline:")

        lstm_model.dummy_train(X_train, y_train)
        dummy_preds = lstm_model.dummy_predict(X_test)
        print(f"Dummy Accuracy: {accuracy_score(y_test, dummy_preds):.4f}")
        print(classification_report(y_test, dummy_preds))

        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        initial_cash = 50000
        cash = initial_cash
        for i in range(len(y_pred) - 1):
            if y_pred[i] == 1:  # Buy signal
                buy_price = test_prices[i]
                next_price = test_prices[i + 1]

                # Buy as many units as possible
                units_bought = cash * 0.5 // buy_price
                if units_bought > 0:
                    cash -= units_bought * buy_price
                    cash += units_bought * next_price  # Sell all next day

        with open(f"models/adrActCnt/report_{coin}.txt", "w") as f:
            f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_pred))
            f.write("\nConfusion Matrix:\n")
            np.savetxt(f, confusion_matrix(y_test, y_pred), fmt="%d")
            f.write(f"final profit and loss: ${(cash-initial_cash):.2f}")

    print("\nAll models processed.")


def run_simulated_trades_per_coin(bitcoin_feature=None):
    altcoin_path = "processed_data/Altcoin.csv"
    bitcoin_path = "processed_data/bitcoin_data_labeled.csv"

    data_per_coin = load_data(
        altcoin_path=altcoin_path,
        bitcoin_path=bitcoin_path,
        bitcoin_feature=bitcoin_feature,
    )

    results = []
    model_name = f"models/v1/model_{bitcoin_feature if bitcoin_feature else 'no_btc'}.h5"
    model = LSTMModel(input_shape=next(iter(data_per_coin.values()))[0].shape[1:])
    model.load(model_name)

    model.encoder.fit(np.array([-1, 0, 1]))
    final_pnl = 0
    for asset, (X_train, y_train, X_test, y_test, close_prices) in data_per_coin.items():
        if len(y_test) == 0:
            print(f"Warning: No test data for {asset}. Skipping.")
            continue

        y_pred = model.predict(X_test)

        initial_cash = 50000
        cash = initial_cash
        for i in range(len(y_pred) - 1):
            if y_pred[i] == 1:
                buy_price = close_prices[i]
                next_price = close_prices[i + 1]

                units_bought = cash * 0.5 // buy_price
                if units_bought > 0:
                    cash -= units_bought * buy_price
                    cash += units_bought * next_price

        profit_loss = cash - initial_cash

        final_pnl += profit_loss
        results.append(f"{asset} ({bitcoin_feature if bitcoin_feature else 'no_btc'}): ${profit_loss:.2f}")

    os.makedirs("trade_results", exist_ok=True)
    os.makedirs("trade_results/v4", exist_ok=True)

    output_file = f"trade_results/v4/trade_results_{bitcoin_feature if bitcoin_feature else 'no_btc'}.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(results))
        f.write("\n")
        f.write(f"final profit and loss: ${final_pnl:.2f}")

    print(f"Saved trade results to {output_file}")


def run_simulated_trades():
    bitcoin_only_features = [
        None,
        "AdrActCnt",
        "TxVal",
        "TxCnt",
        "FeeMean",
        "HashRate",
        "GoogleTrends",
    ]
    for feature in bitcoin_only_features:
        run_simulated_trades_per_coin(feature)


# Run the function
if __name__ == "__main__":
    # run_simulated_trades()
    train_and_test_models_individual()
