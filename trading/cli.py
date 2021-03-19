import click

from trading.prepare import prepare
from trading.experiment import experiment


@click.group()
def trading():
    pass


trading.add_command(prepare)
trading.add_command(experiment)
