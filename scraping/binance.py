import logging

import click
import os.path

import pymongo
from binance.client import Client
import pendulum
from pymongo.errors import BulkWriteError

from config import mongodb

TIMESTEPS = ['5m', '15m', '1h', '4h', '1d']
KEYS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av',
    'tb_quote_av', 'ignore'
]


@click.command()
@click.option('--symbols', default=['BTCUSDT'], help='Symbols to fetch data for', required=True)
def binance(symbols):
    logging.info(f'Scraping binance symbols. Symbols: {symbols}, Timeframes: {TIMESTEPS}')

    client = Client(
        api_key=os.getenv('BINANCE_API_KEY'),
        api_secret=os.getenv('BINANCE_SECRET_KEY')
    )

    db = mongodb()
    scrapperdb = db['market']
    group = scrapperdb['binance']

    for symbol in symbols:
        for step in TIMESTEPS:
            output = group[symbol][step]
            output.ensure_index([
                ('timestamp', pymongo.ASCENDING),
            ])

            last_item = next(output.find().sort("timestamp", pymongo.DESCENDING).limit(1), None)

            start_dt = pendulum.DateTime(year=2017, month=1, day=1) if not last_item \
                else pendulum.from_timestamp(int(last_item['timestamp']/1000)).replace(tzinfo=None)
            end_dt = pendulum.now().replace(tzinfo=None)
            period = pendulum.period(start_dt, end_dt)
            for from_dt in period.range('months'):
                till_dt = from_dt.add(months=1)

                klines = client.get_historical_klines(
                    symbol,
                    step,
                    from_dt.strftime("%d %b %Y %H:%M:%S"),
                    till_dt.strftime("%d %b %Y %H:%M:%S")
                )

                if len(klines) == 0:
                    continue

                try:
                    output.insert_many(
                        [{**dict(zip(KEYS, line)), '_id': line[0]} for line in klines],
                        ordered=False
                    )
                except BulkWriteError as e:
                    pass

    logging.info(f'Finished scraping binance. Symbols: {symbols}, Timeframes: {TIMESTEPS}')