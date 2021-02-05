import click

from scraping.base import scraper


@click.group()
def cli():
    pass


class CatchAllExceptions(click.Group):
    def __call__(self, *args, **kwargs):
        try:
            return self.main(*args, **kwargs)
        except Exception as exc:
            click.echo('We found %s' % exc)


cli.add_command(scraper)

if __name__ == '__main__':
    cli()
