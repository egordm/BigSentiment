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
    # 'USD-XRP',
    # 'USD-LINK',
]

TIMEFRAMES = {
    '-12h': -HOUR * 8,
    '12h': HOUR * 12,
    '1d': DAY,
    '2d': DAY * 2,
    '7d': DAY * 7,
    '14d': DAY * 14,
}
OUTPUT_PATH = '../../data/bitcoin_twitter_labeled/'

# %%

client = mongodb()
collection = client['market']['coinmarketcap']

# %%
ensure_dataset(OUTPUT_PATH, delete=True)
files = pathlib.Path("../../data/bitcoin_twitter_processed/").glob("part_*.parquet")
for chunk, file in enumerate(files):
    data = pd.read_parquet(file)
    datasets = [data]

    for CURRENCY in CURRENCIES:
        print(f'Processing: chunk:{chunk}, currency:{CURRENCY}')
        # %%
        query = collection[CURRENCY]['1h'] \
            .find({}, {'_id': 0, 'timestamp': 1, 'close': 1}) \
            .sort([('timestamp', pymongo.ASCENDING)])
        market = pd.DataFrame(list(query))
        max_date = pendulum.from_timestamp(market.iloc[-1].timestamp)
        min_date = pendulum.from_timestamp(market.iloc[0].timestamp)

        # %%
        mask = (data['created_at'] < np.datetime64(max_date.subtract(seconds=TIMEFRAMES['14d']))) & \
               (data['created_at'] > np.datetime64(min_date.subtract(seconds=TIMEFRAMES['-12h'])))

        # %%
        row_timestamps = data['created_at'].astype('int64') // 10 ** 9
        current_value = np.array(market['close'].iloc[market['timestamp'].searchsorted(row_timestamps)])

        labels = {}
        for label, delta in TIMEFRAMES.items():
            print(f'Creating labels: {label}')
            max_idx = len(market['close']) - 1
            indices = np.minimum(market['timestamp'].searchsorted(row_timestamps + delta), max_idx)
            new_value = market['close'].iloc[indices]
            result = np.array(new_value - current_value) / current_value
            result[~mask] = None
            labels[f'feat-{CURRENCY}-change-{label}'] = result

        labels = pd.DataFrame(labels)
        datasets.append(labels)

    data = pd.concat(datasets, axis=1)
    if len(data) > 0:
        data.to_parquet(os.path.join(OUTPUT_PATH, f"part_{chunk}.parquet"))
    else:
        print(f'Skipping part: {chunk}')
    print(f'Processed part: {chunk}')
    print(data.describe())
