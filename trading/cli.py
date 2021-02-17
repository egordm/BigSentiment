import click

from trading.runners import test_runner


@click.group()
def trading():
    pass


trading.add_command(test_runner)
