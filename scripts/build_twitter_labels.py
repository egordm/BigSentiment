# %%
import os
import pathlib

import pandas as pd
import pymongo
from config import mongodb
import pendulum
import numpy as np

# %%
from utils.datasets import ensure_dataset

HOUR = 3600
DAY = HOUR * 24
CURRENCIES = [
    'USD-BTC',
    'USD-ETH',
    'USD-XRP',
    # 'USD-LINK',
]

TIMEFRAMES = {
    'n12h': -HOUR * 8,
    '12h': HOUR * 12,
    '1d': DAY,
    '2d': DAY * 2,
    '7d': DAY * 7,
    '14d': DAY * 14,
}
OUTPUT_PATH = '../data/bitcoin_twitter_test_labeled/'

# %%

client = mongodb()
collection = client['market']['coinmarketcap']

# %%
ensure_dataset(OUTPUT_PATH, delete=True)
files = pathlib.Path("../data/bitcoin_twitter_test_processed/").glob("part_*.parquet")
for chunk, file in enumerate(files):
    data = pd.read_parquet(file)
    datasets = [data[['_id', 'text', 'follower_count', 'retweet_count', 'created_at']]]

    for CURRENCY in CURRENCIES:
        # Retrieve collection for given currency
        print(f'Processing: chunk:{chunk}, currency:{CURRENCY}')
        query = collection[CURRENCY]['1h'] \
            .find({}, {'_id': 0, 'timestamp': 1, 'close': 1}) \
            .sort([('timestamp', pymongo.ASCENDING)])
        market = pd.DataFrame(list(query))
        # Find it's min and max tracked date
        max_date = pendulum.from_timestamp(market.iloc[-1].timestamp)
        min_date = pendulum.from_timestamp(market.iloc[0].timestamp)

        # Create row mask for rows which match the start and date by a mergin
        mask = (data['created_at'] < np.datetime64(max_date.subtract(seconds=TIMEFRAMES['14d']))) & \
               (data['created_at'] > np.datetime64(min_date.subtract(seconds=TIMEFRAMES['n12h'])))

        # Parse timetamp and search for their nearest price (for each datapoint)
        row_timestamps = data['created_at'].astype('int64') // 10 ** 9
        current_value = np.array(market['close'].iloc[market['timestamp'].searchsorted(row_timestamps)])

        labels = {}
        for label, delta in TIMEFRAMES.items():
            print(f'Creating labels: {label}')
            # Search the closest price for given price term
            max_idx = len(market['close']) - 1
            indices = np.minimum(market['timestamp'].searchsorted(row_timestamps + delta), max_idx)
            new_value = market['close'].iloc[indices]
            result = np.array(new_value - current_value) / current_value
            result[~mask] = None
            labels[f'feat_{CURRENCY.replace("-", "")}_change_{label}'] = result

        labels = pd.DataFrame(labels)
        datasets.append(labels)

    # reset indexes first so that we join as is
    for ds in datasets:
        ds.reset_index(drop=True, inplace=True)

    # Horizontally stack all the datasets
    data = pd.concat(datasets, axis=1)
    if len(data) > 0:
        data.to_parquet(os.path.join(OUTPUT_PATH, f"part_{chunk}.parquet"))
    else:
        print(f'Skipping part: {chunk}')
    print(f'Processed part: {chunk}')
    print(data.describe())
