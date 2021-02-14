import logging
import pathlib

import click
import os.path

import pymongo
from binance.client import Client
import pendulum
from pymongo.errors import BulkWriteError
import pandas as pd

from config import mongodb

DATA_FOLDER = 'data/cryptodata'

TIMESTEPS = {
    '5m': 300,
    '15m': 900,
    '1h': 3600,
    '4h': 14400,
    '1d': 86400
}


def aggregate_bin(data):
    return {
        'timestamp': data[0]['unix'],
        'open': data[0]['open'],
        'high': max(map(lambda x: x['high'], data)),
        'low': min(map(lambda x: x['low'], data)),
        'close': data[-1]['close'],
        'volume': sum(map(lambda x: x['volume'], data)),
        'volume_USD': sum(map(lambda x: x['Volume USD'], data)),
    }


@click.command()
def bitstamp_file():
    logging.info(f'Loading bitstamp data.')

    db = mongodb()
    scrapperdb = db['market']
    group = scrapperdb['bitstamp']

    # Assume importing 1m files
    interval = 60
    bincounts = {k: int(v / interval) for k, v in TIMESTEPS.items()}
    files = pathlib.Path(DATA_FOLDER).glob("Bitstamp_*.csv")
    for file in files:
        _, symbol, _ = os.path.splitext(os.path.basename(file))[0].split('_')
        logging.debug(f'Processing {symbol}')
        df = pd.read_csv(file, skiprows=1)
        df['unix'] = df['unix'].astype(int)
        df['volume'] = df[df.columns[-2]]
        df.sort_values(by='unix', ascending=True, inplace=True)

        data = {k: [] for k, _ in TIMESTEPS.items()}
        bins = {k: [] for k, _ in TIMESTEPS.items()}

        for i, row in df.iterrows():
            for k in TIMESTEPS.keys():
                bins[k].append(row)

            for k, bincount in bincounts.items():
                if len(bins[k]) >= bincount:
                    data[k].append(aggregate_bin([bins[k].pop(0) for _ in range(bincount)]))

        for k, lines in data.items():
            output = group[symbol][k]
            output.ensure_index([
                ('timestamp', pymongo.ASCENDING),
            ])

            try:
                output.insert_many(
                    [{**line, '_id': line['timestamp']} for line in lines],
                    ordered=False
                )
            except BulkWriteError as e:
                pass