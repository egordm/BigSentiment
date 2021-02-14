import click

from scraping.bitstamp_file import bitstamp_file
from scraping.binance import binance
from scraping.coinmarketcap import coinmarketcap
from scraping.news import news
from scraping.twitter import twitter
from scraping.twitter_old import twitter_old


@click.group()
def scraper():
    pass


scraper.add_command(twitter)
scraper.add_command(twitter_old)
scraper.add_command(news)
scraper.add_command(binance)
scraper.add_command(coinmarketcap)
scraper.add_command(bitstamp_file)
