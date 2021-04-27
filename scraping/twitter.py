import json
from dataclasses import dataclass

import click
import pymongo
import requests
import re
import time
import datetime as dt
import urllib.parse as u
import logging
from selenium import webdriver
from tqdm import tqdm

from config import mongodb

driver_options = webdriver.FirefoxOptions()
driver_options.headless = True
driver = webdriver.Firefox(firefox_options=driver_options)
session = requests.Session()

DT_FMT = '%Y-%m-%d'

URL = (
    f'https://api.twitter.com/2/search/adaptive.json?'
    f'include_profile_interstitial_type=1'
    f'&include_blocking=1'
    f'&include_blocked_by=1'
    f'&include_followed_by=1'
    f'&include_want_retweets=1'
    f'&include_mute_edge=1'
    f'&include_can_dm=1'
    f'&include_can_media_tag=1'
    f'&skip_status=1'
    f'&cards_platform=Web-12'
    f'&include_cards=1'
    f'&include_ext_alt_text=true'
    f'&include_quote_count=true'
    f'&include_reply_count=1'
    f'&tweet_mode=extended'
    f'&include_entities=true'
    f'&include_user_entities=true'
    f'&include_ext_media_color=true'
    f'&include_ext_media_availability=true'
    f'&send_error_codes=true'
    f'&simple_quoted_tweet=true'
    f'&query_source=typed_query'
    f'&pc=1'
    f'&spelling_corrections=1'
    f'&ext=mediaStats%2ChighlightedLabel'
    f'&count=100'
    f'&tweet_search_mode=live'
    '&q={query}'
)

CURSOR_RE = re.compile('"(scroll:[^"]*)"')


def retry(fn, count=5, delay=1, cb=None):
    ex = None
    for i in range(count):
        try:
            return fn()
        except Exception as e:
            time.sleep(delay)
            if cb: cb()
            ex = e
    raise ex


def update_cookies(state):
    driver.delete_all_cookies()
    driver.get('https://twitter.com/explore')

    # Update cookies
    cookies = driver.get_cookies()
    for cookie in cookies:
        session.cookies.set(cookie['name'], cookie['value'])

    # Update headers
    guest_token = driver.get_cookie('gt')['value']
    # csrf_token = driver.get_cookie('ct0')['value']
    state.headers = {
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0',
        'authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA',
        'x-guest-token': guest_token,
        # 'x-csrf-token': csrf_token,
        'x-twitter-active-user': 'yes',
        'x-twitter-client-language': 'en',
    }


def request_content(state, query: str, date: dt.datetime, cursor=None):
    if cursor:
        url = URL + '&cursor={cursor}'
        url = url.format(query=u.quote(query), cursor=u.quote(cursor))
    else:
        date_filter = f'since:{date.strftime(DT_FMT)} until:{(date + dt.timedelta(days=1)).strftime(DT_FMT)}'
        retweet_filter = '-filter:retweets AND -filter:replies'
        query = f"'{query}' {retweet_filter} {date_filter}"
        url = URL.format(query=u.quote(query))

    response = session.get(url, headers=state.headers)
    data = json.loads(response.text)
    cursor = CURSOR_RE.search(response.text).group(1)
    return data, cursor


def transform_tweet(tweet):
    return {
        '_id': tweet['id'],
        'text': tweet['full_text'],
        'user_id': tweet['user_id'],
        'follower_count': None,
        'verified': None,
        'created_at': dt.datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y'),
        'retweet_count': tweet['retweet_count'],
        'favorite_count': tweet['favorite_count'],
        'topics': [
            tag.lower()
            for tag in re.findall(r"#(\w*[0-9a-zA-Z]+\w*[0-9a-zA-Z])", tweet['full_text'])
        ]
    }


@dataclass
class State:
    headers = None


@click.command()
@click.option('--query', help='Query to fetch data from twitter from', required=True)
@click.option('--since', type=str, default=None, help='Fetch data since date')
@click.option('--until', type=str, default=None, help='Fetch data until date')
@click.option('--depth', type=int, default=600, help='Pages depth to check')
def twitter(query, since, until, depth):
    since = dt.datetime.strptime(since, DT_FMT) if since else dt.datetime.now()
    until = dt.datetime.strptime(until, DT_FMT) if until else dt.datetime.now() - dt.timedelta(days=365)

    logging.info(f'Starting Historical Twitter scraping. '
                 f'Query: {query}, From: {since.strftime(DT_FMT)} Until: {until.strftime(DT_FMT)}')

    # Database config
    client = mongodb()
    scrapperdb = client['scrapper']
    output = scrapperdb['tweets']
    output.ensure_index([
        ('created_at', pymongo.DESCENDING),
        ('topics', pymongo.ASCENDING)
    ])

    state = State()
    cookie_update_fn = lambda: retry(lambda: update_cookies(state), 5, 5)

    p1 = tqdm(total=(until - since).days)
    count = 0
    while since > until:
        logging.debug(f'Scraping Tweets from: {since.strftime(DT_FMT)}')

        # Range request
        cursor = None
        for i in tqdm(range(depth), leave=False, postfix={'date': since.strftime(DT_FMT)}):
            # Update the cookies
            if count % 100 == 0:
                logging.debug('Updating cookies')
                cookie_update_fn()

            try:
                data, cursor = retry(
                    lambda: request_content(state, query, since, cursor=cursor),
                    cb=cookie_update_fn,
                    count=5, delay=5
                )
            except Exception as e:
                print(e)
                break

            # Parse the tweets and push them to the database
            tweets = list(map(transform_tweet, data['globalObjects']['tweets'].values()))
            for tweet in tweets:
                output.update({'_id': tweet['_id']}, tweet, upsert=True)

            # Update counters
            count += 1

        # Update day
        since = since - dt.timedelta(days=1)
        p1.update(1)
