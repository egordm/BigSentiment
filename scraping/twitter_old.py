import logging

import pymongo
import requests
from tqdm import tqdm
from datetime import datetime
import re
import click
import pendulum
import time
from fake_useragent import UserAgent

from config import mongodb

USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2656.18 Safari/537.36'
#USER_AGENT = UserAgent().chrome
GUEST_AUTH = 'AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA'


def get_guest_token():
    for i in range(5):
        try:
            result = requests.get('https://twitter.com/search?q=%23bitcoin', headers={
                'User-Agent': USER_AGENT
            })
            return re.findall('gt=([0-9]*)', result.text)[-1]
        except Exception as e:
            time.sleep(30)


def fetch_data(guest_token, query, cursor, since: pendulum.Date):
    for i in range(5):
        since_s = since.format('YYYY-MM-DD')
        until_s = since.add(days=1).format('YYYY-MM-DD')
        extra = {'cursor': cursor} if cursor else {}

        response = requests.get('https://twitter.com/i/api/2/search/adaptive.json', {
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
        })
        if response.status_code == 401:
            return [], None

        if response.status_code != 200 and (response.status_code % 200) != 1:
            time.sleep(5)
            print('retrying')
            continue

        result = response.json()
        if 'globalObjects' not in result:
            return [], None

        tweets = list(result['globalObjects']['tweets'].values())
        instruction = result['timeline']['instructions'][-1]
        if instruction.get('addEntries', None):
            cursor = instruction['addEntries']['entries'][-1]['content']['operation']['cursor']['value']
        else:
            cursor = instruction['replaceEntry']['entry']['content']['operation']['cursor']['value']
        return tweets, cursor
    return [], None


@click.command()
@click.option('--query', help='Query to fetch data from twitter from', required=True)
@click.option('--depth', type=int, default=10, help='Amount of pages to visit')
@click.option('--offset', type=int, default=0, help='Days offset from now to start scraping')
def twitter_old(query, depth, offset):
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
    since = since.subtract(days=offset)
    for i in tqdm(range(6052)):
        cursor = None

        if i % 20 == 0:
            guest_token = get_guest_token()

        error_count = 0
        for u in range(depth):
            old_cursor = cursor
            try:
                tweets, cursor = fetch_data(guest_token, query, cursor, since)
            except Exception as e:
                logging.exception(str(e))
                if error_count > 5:
                    break
                error_count += 1
                continue

            if u != 0 and (u % 50) == 0:
                guest_token = get_guest_token()

            if len(tweets) == 0 or cursor == old_cursor:
                break

            error_count = 0
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
