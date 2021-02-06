import click

from scraping.news import news
from scraping.twitter import twitter
from scraping.twitter_old import twitter_old


@click.group()
def scraper():
    pass


scraper.add_command(twitter)
scraper.add_command(twitter_old)
scraper.add_command(news)
