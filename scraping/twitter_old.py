import logging

import pymongo
import requests
from tqdm import tqdm
from datetime import datetime
import re
import click
import pendulum

from config import mongodb

USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2656.18 Safari/537.36'
GUEST_AUTH = 'AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA'


def get_guest_token():
    result = requests.get('https://twitter.com/', headers={
        'User-Agent': USER_AGENT
    })
    return re.findall('gt=([0-9]*)', result.text)[-1]


def fetch_data(guest_token, query, cursor, since: pendulum.Date):
    since_s = since.format('YYYY-MM-DD')
    until_s = since.add(days=1).format('YYYY-MM-DD')
    extra = {'cursor': cursor} if cursor else {}

    result = requests.get('https://twitter.com/i/api/2/search/adaptive.json', {
        'q': f"{query} -filter:retweets AND -filter:replies since:{since_s} until:{until_s}",
        'count': 100,
        'tweet_mode': 'extended',
        'include_followed_by': True,
        'include_entities': True,
        'include_user_entities': True,
        **extra
    }, headers={
        'User-Agent': USER_AGENT,
        'Authorization': f'Bearer {GUEST_AUTH}',
        'x-guest-token': guest_token,
        'x-twitter-active-user': 'yes',
        'x-twitter-client-language': 'en',
    }).json()

    if 'globalObjects' not in result:
        return [], None

    tweets = list(result['globalObjects']['tweets'].values())
    instruction = result['timeline']['instructions'][-1]
    if instruction.get('addEntries', None):
        cursor = instruction['addEntries']['entries'][-1]['content']['operation']['cursor']['value']
    else:
        cursor = instruction['replaceEntry']['entry']['content']['operation']['cursor']['value']
    return tweets, cursor


@click.command()
@click.option('--query', help='Query to fetch data from twitter from', required=True)
@click.option('--depth', type=int, default=10, help='Amount of pages to visit')
def twitter_old(query, depth):
    logging.info(f'Starting Historical Twitter scraping. Query: {query}, Depth: {depth}')

    client = mongodb()
    scrapperdb = client['scrapper']
    output = scrapperdb['tweets']
    output.ensure_index([
        ('created_at', pymongo.DESCENDING),
        ('topics', pymongo.ASCENDING)
    ])

    counter = 0
    guest_token = get_guest_token()
    since = pendulum.now().start_of('day')
    for i in tqdm(range(6052)):
        cursor = None

        if i % 20 == 0:
            guest_token = get_guest_token()

        for _ in range(depth):
            old_cursor = cursor
            tweets, cursor = fetch_data(guest_token, query, cursor, since)

            if len(tweets) == 0 or cursor == old_cursor:
                break

            for tweet in tweets:
                id = tweet['id']
                result = {
                    '_id': id,
                    'text': tweet['full_text'],
                    'user_id': tweet['user_id'],
                    'follower_count': None,
                    'verified': None,
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

        since = since.subtract(days=1)

    logging.info(f'Finished scraping twitter. Results: {counter} Query: {query}, Depth: {depth}')