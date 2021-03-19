import logging
import os

import click
import pendulum as dt
import pymongo
import pandas as pd
from pandarallel import pandarallel

from config import mongodb
from utils.datasets import ensure_dataset, DATASET_DIR

pandarallel.initialize()

VALID_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']


def bitstamp_transform(df: pd.DataFrame):
    df.drop(columns=[c for c in df.columns if c not in VALID_COLUMNS], inplace=True)
    df['timestamp'] = df['timestamp'].parallel_map(lambda x: dt.from_timestamp(int(x)))
    return df


def binance_transform(df: pd.DataFrame):
    df.drop(columns=[c for c in df.columns if c not in VALID_COLUMNS], inplace=True)
    df['timestamp'] = df['timestamp'].parallel_map(lambda x: dt.from_timestamp(x / 1000))
    return df


@click.command()
def prepare():
    TIMESCALE = '5m'
    DATASETS = {
        'bitstamp': [
            'btcusd', 'btcusdc', 'ethusd', 'ethusdc', 'bchusd',
            'linkusd', 'ltcusd', 'omgusd', 'paxusd', 'xlmusd', 'xrpusd'
        ],
        'binance': [
            'BTCUSDT', 'ETHUSDT'
        ]
    }
    CUTOFF_TICKS = 200

    dataset_path = os.path.join(DATASET_DIR, 'market')
    ensure_dataset(dataset_path, delete=True)

    market_collection = mongodb()['market']
    for source, pairs in DATASETS.items():
        for pair in pairs:
            logging.debug(f'Preprocessing {source}/{pair}')
            collection = market_collection[source][pair][TIMESCALE].find()
            df = pd.DataFrame(list(collection))
            df = bitstamp_transform(df) if source == 'bitstamp' else binance_transform(df)
            # Convert fields to float
            for c in VALID_COLUMNS:
                if c != 'timestamp':
                    df[c] = df[c].astype(float)
            # Presort data
            df.sort_values(by=['timestamp'], ascending=True, inplace=True)
            # Find start cutoff value
            counter, cutoff = 0, df['timestamp'].iloc[-1]
            for i, row in df[['volume', 'timestamp']].iterrows():
                counter = counter + 1 if row['volume'] > 0.0001 else 0
                if counter > CUTOFF_TICKS:
                    cutoff = df['timestamp'].iloc[i - CUTOFF_TICKS + 1]
                    break
            logging.debug(f'- Found cutoff date at {cutoff.isoformat()}. Removing {len(df[df["timestamp"] <= cutoff])}')
            df = df[df["timestamp"] >= cutoff].reindex(columns=VALID_COLUMNS)
            # Save the dataset
            logging.debug(f'Writing {source}-{pair}.parquet with {len(df)} records')
            df.to_parquet(os.path.join(dataset_path, f'{source}-{pair}.parquet'))
