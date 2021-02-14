import logging
import pathlib

import click
import os.path

import pymongo
import requests
from binance.client import Client
import pendulum
from pymongo.errors import BulkWriteError
import pandas as pd
import time

from config import mongodb

DATA_FOLDER = 'data/cryptodata'

TIMESTEPS = {
    '5m': 300,
    '15m': 900,
    '1h': 3600,
    '4h': 14400,
    '1d': 86400
}

ITEM_LIMIT = 1000

USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2656.18 Safari/537.36'

ALLOWED_PAIRS = ['btcusd', 'btcusdc', 'xrpusd', 'ltcusd', 'ethusd', 'ethusdc', 'bchusd', 'paxusd', 'xlmusd', 'linkusd',
                 'omgusd']


@click.command()
@click.option('--symbols', default=ALLOWED_PAIRS, help='Symbols to fetch data for', required=True, multiple=True)
def bitstamp(symbols):
    logging.info(f'Scraping bitstamp symbols. Symbols: {symbols}, Timeframes: {TIMESTEPS}')

    db = mongodb()
    scrapperdb = db['market']
    group = scrapperdb['bitstamp']

    for symbol in symbols:
        for step, delta in reversed(TIMESTEPS.items()):
            output = group[symbol][step]
            output.ensure_index([
                ('timestamp', pymongo.ASCENDING),
            ])

            last_item = next(output.find().sort("timestamp", pymongo.DESCENDING).limit(1), None)
            start_dt = pendulum.DateTime(year=2013, month=1, day=1).replace(tzinfo=None) if not last_item \
                else pendulum.from_timestamp(int(last_item['timestamp'])).replace(tzinfo=None)

            end_dt = pendulum.now().replace(tzinfo=None)
            period = pendulum.period(start_dt, end_dt)
            for from_dt in period.range('seconds', delta * ITEM_LIMIT):
                till_dt = from_dt.add(seconds=delta * ITEM_LIMIT)

                time.sleep(1)
                result = requests.get(f'https://www.bitstamp.net/api/v2/ohlc/{symbol}/', params={
                    'step': delta,
                    'start': from_dt.replace(tzinfo=pendulum.UTC).int_timestamp,
                    'end': till_dt.replace(tzinfo=pendulum.UTC).int_timestamp,
                    'limit': 1000
                }, headers={
                    'User-Agent': USER_AGENT,
                }).json()

                klines = []
                for tick_data in result['data']['ohlc']:
                    klines.append(tick_data)

                if len(klines) == 0:
                    continue

                try:
                    output.insert_many(
                        [{**line, '_id': line['timestamp']} for line in klines],
                        ordered=False
                    )
                except BulkWriteError as e:
                    pass

    logging.info(f'Finished scraping bitstamp. Symbols: {symbols}, Timeframes: {TIMESTEPS}')
