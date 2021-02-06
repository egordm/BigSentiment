import logging

import pymongo
from tqdm import tqdm
from twython import Twython
from datetime import datetime
import re
import os
import click

from config import mongodb


@click.command()
@click.option('--query', help='Query to fetch data from twitter from', required=True)
@click.option('--type', type=click.Choice(['popular', 'recent', 'mixed']),
              default='mixed', help='Type of the scrape page')
@click.option('--depth', type=int, default=20, help='Amount of pages to visit')
def twitter(query, type, depth):
    APP_KEY = os.getenv('TWITTER_APP_KEY')
    APP_SECRET = os.getenv('TWITTER_APP_SECRET')
    query += ' -filter:retweets AND -filter:replies'
    logging.info(f'Starting Twitter scraping. Query: {query}, Depth: {depth}')

    twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
    ACCESS_TOKEN = twitter.obtain_access_token()
    twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)

    client = mongodb()
    scrapperdb = client['scrapper']
    output = scrapperdb['tweets']
    output.ensure_index([
        ('created_at', pymongo.DESCENDING),
        ('topics', pymongo.ASCENDING)
    ])

    counter = 0
    next_id = None
    for _ in tqdm(range(depth)):
        data = twitter.search(
            q=query,
            lang='en',
            result_type=type,
            tweet_mode='extended',
            count="100",
            next_id=next_id
        )['statuses']

        if len(data) == 0 or next_id == data[-1]['id']:
            break

        next_id = data[-1]['id']
        for tweet in data:
            id = tweet['id']
            result = {
                '_id': id,
                'text': tweet['full_text'],
                'user_id': tweet['user']['id'],
                'follower_count': tweet['user']['followers_count'],
                'verified': tweet['user']['verified'],
                'created_at': datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y'),
                'retweet_count': tweet['retweet_count'],
                'favorite_count': tweet['favorite_count'],
                'topics': [
                    tag.lower()
                    for tag in re.findall(r"#(\w*[0-9a-zA-Z]+\w*[0-9a-zA-Z])", tweet['full_text'])
                ]
            }
            output.update({'_id': id}, result, upsert=True)
            counter += 1

    logging.info(f'Finished scraping twitter. Results: {counter} Query: {query}, Depth: {depth}')
