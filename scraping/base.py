import click

from scraping.news import news
from scraping.twitter import twitter


@click.group()
def scraper():
    pass


scraper.add_command(twitter)
scraper.add_command(news)
