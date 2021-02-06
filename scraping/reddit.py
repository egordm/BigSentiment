import logging

import pendulum
import pymongo
from tqdm import tqdm
from twython import Twython
from datetime import datetime
import re
import os
import click
import praw

from config import mongodb


@click.command()
@click.option('--reddits', default=['bitcoin'], help='Query to fetch data from reddit from', required=True)
def reddit(reddits):
    # logging.info(f'Starting Reddit scraping. Query: {query}, Depth: {depth}')

    raise Exception('Reddit api is horseshit')

    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_APP'),
        client_secret=os.getenv('REDDIT_SECRET'),
        password=os.getenv('REDDIT_PASS'),
        username=os.getenv('REDDIT_USER')
    )

    client = mongodb()
    scrapperdb = client['scrapper']
    output = scrapperdb['tweets']
    output.ensure_index([
        ('created_at', pymongo.DESCENDING),
        ('topics', pymongo.ASCENDING)
    ])

    # for subredit in reddits:
    #     since = pendulum.now().start_of('day')
    #
    #     while True:
    #         try:
    #             ts1 = since.format('DD.MM.YYYY-HH HH:mm:ss')
    #             searchResults = list(
    #                 reddit.subreddit(subredit).search('timestamp:{}..{}'.format(cts1, cts2), syntax='cloudsearch')
    #             )
    #             # If it errors for whatever reason, log the error and continue
    #         except Exception as e:
    #             logging.exception(e)
    #             continue
    #
