import click
import sys


@click.group()
def cli():
    pass


class CatchAllExceptions(click.Group):
    def __call__(self, *args, **kwargs):
        try:
            return self.main(*args, **kwargs)
        except Exception as exc:
            click.echo('We found %s' % exc)


if sys.argv[1] == 'scraper':
    from scraping.base import scraper

    cli.add_command(scraper)

if sys.argv[1] == 'trading':
    from trading.cli import trading

    cli.add_command(trading)

if __name__ == '__main__':
    cli()
