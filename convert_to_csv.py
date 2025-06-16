import os
import csv
import re
from collections import defaultdict


def convert_txts_to_feature_pivot_csv(input_dir, output_file):
    data = defaultdict(dict)
    features = set()

    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(input_dir, filename)
        with open(file_path, "r") as file:
            lines = file.readlines()

        feature = None
        for line in lines:
            line = line.strip()
            match = re.match(r".+?\s+\((.+?)\):\s+\$(.+)", line)
            if match:
                feature = match.group(1)
                break

        if feature is None:
            print(f"Could not determine feature in file: {filename}")
            continue

        features.add(feature)

        for line in lines:
            line = line.strip()
            if line.lower().startswith("final profit and loss"):
                # Extract value for final PnL
                match = re.search(r"\$([\d,.]+)", line)
                if match:
                    value = float(match.group(1).replace(",", ""))
                    data["TotalPnL"][feature] = value
                continue

            match = re.match(r"(.+?)\s+\((.+?)\):\s+\$(.+)", line)
            if match:
                coin, _, value = match.groups()
                try:
                    data[coin][feature] = float(value.replace(",", ""))
                except ValueError:
                    print(f"Warning: could not parse value in {filename}: {line}")

    sorted_features = sorted(features)

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Coin"] + sorted_features)

        for coin in sorted(data.keys()):
            row = [coin] + [data[coin].get(feature, "") for feature in sorted_features]
            writer.writerow(row)

    print(f"Combined pivoted CSV written to: {output_file}")


# Run it
convert_txts_to_feature_pivot_csv(input_dir="trade_results/v4", output_file="combined_results.csv")
