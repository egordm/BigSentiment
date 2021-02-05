import click

from scraping.twitter import twitter


@click.group()
def scraper():
    pass


scraper.add_command(twitter)
