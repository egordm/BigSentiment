import logging

import click
import os.path

import pymongo
import requests
from binance.client import Client
import pendulum
from pymongo.errors import BulkWriteError

from config import mongodb

TIMESTEPS = {
    '5m': 300,
    '15m': 900,
    '1h': 3600,
    '4h': 14400,
    '1d': 86400
}
KEYS = [
    'timestamp','close', 'volume_24hr', 'market_cap'
]

ITEM_LIMIT = 10000 - 1000

USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2656.18 Safari/537.36'



@click.command()
@click.option('--symbols', default=['USD-BTC'], help='Symbols to fetch data for', required=True, multiple=True)
def coinmarketcap(symbols):
    logging.info(f'Scraping coinmerketcap symbols. Symbols: {symbols}, Timeframes: {TIMESTEPS}')

    db = mongodb()
    scrapperdb = db['market']
    group = scrapperdb['coinmarketcap']

    for symbol in symbols:
        sym_to, sym_from = symbol.split('-')

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

                result = requests.get('https://web-api.coinmarketcap.com/v1.1/cryptocurrency/quotes/historical', params={
                    'convert': f'{sym_to},{sym_from}',
                    'format': 'chart_crypto_details',
                    'id': 1,
                    'interval': step,
                    'time_end': till_dt.replace(tzinfo=pendulum.UTC).int_timestamp,
                    'time_start': from_dt.replace(tzinfo=pendulum.UTC).int_timestamp,
                    'skip_invalid': True
                }, headers={
                    'User-Agent': USER_AGENT,
                }).json()

                if 'data' not in result:
                    if 'older than' in result['status']['error_message']:
                        continue
                    iu = 0

                klines = []
                for tick_dt, tick_data in result['data'].items():
                    klines.append([
                        pendulum.parse(tick_dt).int_timestamp,
                        *tick_data[sym_to]
                    ])

                if len(klines) == 0:
                    continue

                try:
                    output.insert_many(
                        [{**dict(zip(KEYS, line)), '_id': line[0]} for line in klines],
                        ordered=False
                    )
                except BulkWriteError as e:
                    pass

    logging.info(f'Finished scraping coinmarketcap. Symbols: {symbols}, Timeframes: {TIMESTEPS}')