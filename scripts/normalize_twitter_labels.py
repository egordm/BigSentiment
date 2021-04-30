import os
import pathlib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle

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

FEATURES = [
    f'feat_{CURRENCY.replace("-", "")}_change_{label}'
    for label, delta in TIMEFRAMES.items()
    for CURRENCY in CURRENCIES
]

LOAD = True

OUTPUT_PATH = '../data/bitcoin_twitter_test_labeled_normalized/'
ensure_dataset(OUTPUT_PATH, delete=True)

# Record mean and standard dist for the split dataset
scalers = {feat: StandardScaler() for feat in FEATURES}

# Loads means if possible
if LOAD:
    with open('normalize_params.pickle', 'rb') as f:
        feature_specs = pickle.load(f)
        for feature, data in feature_specs.items():
            scalers[feature].mean_ = data['mean']
            scalers[feature].var_ = data['var']
            scalers[feature].scale_ = data['scale']
            scalers[feature].n_samples_seen_ = data['n_samples_seen']

else:
    files = pathlib.Path("../data/bitcoin_twitter_labeled/").glob("part_*.parquet")
    for chunk, file in enumerate(files):
        data = pd.read_parquet(file)
        for feature in FEATURES:
            values = np.array(data[feature][data[feature].notna()])
            if len(values) == 0: continue
            scalers[feature].partial_fit(values.reshape(-1, 1))

for feature in FEATURES:
    print(f'{feature}: mean={scalers[feature].mean_}, var={scalers[feature].var_}')

# Save the means for other datasets
if not LOAD:
    feature_specs = {feature: {
        'mean': scalers[feature].mean_,
        'var': scalers[feature].var_,
        'scale': scalers[feature].scale_,
        'n_samples_seen': scalers[feature].n_samples_seen_,
    } for feature in FEATURES}
    with open('normalize_params.pickle', 'wb') as f:
        pickle.dump(feature_specs, f, protocol=pickle.HIGHEST_PROTOCOL)

# %%

# Normalize each feature given it's distribution
files = pathlib.Path("../data/bitcoin_twitter_test_labeled/").glob("part_*.parquet")
for chunk, file in enumerate(files):
    data = pd.read_parquet(file)
    print(f'Processing chunk {chunk}')
    for feature in FEATURES:
        data = data[data[feature].notna()]
        if len(data) == 0: break
        values = np.array(data[feature]).reshape(-1, 1)
        data[feature] = scalers[feature].transform(values).reshape(-1)
    if len(data) == 0:
        continue
    data.to_parquet(os.path.join(OUTPUT_PATH, f"part_{chunk}.parquet"))
