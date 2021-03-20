import logging
import os
import pathlib
import random
import time
from collections import defaultdict
from typing import List

import pandas as pd

import click
import wandb
# from tensortrade.agents import DQNAgent, A2CAgent
from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.env.default import actions, rewards, observers, stoppers, renderers, informers
from tensortrade.env.default.actions import SimpleOrders
from tensortrade.env.default.observers import ObservationHistory
from tensortrade.env.default.rewards import SimpleProfit
from tensortrade.env.generic import TradingEnv
from tensortrade.feed import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet
import datetime as dt

from trading.agents.a2c_agent import A2CAgent
from trading.env.datasource import LocalDatasource
from trading.env.instruments import rsi, macd, percdiff
from trading.env.observers import RollingEpisodicObserver
from trading.env.renderers import PlotlyCustomChart


@click.command()
def experiment():
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    wandb.init(project='crypto_rl')

    # Load dataset
    datasource = LocalDatasource()
    data = datasource.fetch('bitstamp', 'btcusd', '5m')
    print(data.tail())

    features = []
    for c in data.columns[1:]:
        s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
        features += [s]

    cp = Stream.select(features, lambda s: s.name == "close")
    feed = DataFeed([
        Stream.source(list(data['timestamp']), dtype="datetime64").rename('timestamp'),
        percdiff(cp).rename("pd"),
        rsi(cp, period=20).rename("rsi"),
        macd(cp, fast=10, slow=50, signal=5).rename("macd")
    ])
    feed.compile()

    bitstamp = Exchange("bitstamp", service=execute_order)(
        Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
    )
    portfolio = Portfolio(USD, [
        Wallet(bitstamp, 10000 * USD),
        Wallet(bitstamp, 0 * BTC)
    ])
    renderer_feed = DataFeed(datasource.renderer_transform(data))

    action_scheme = SimpleOrders(trade_sizes=1)
    action_scheme.portfolio = portfolio
    reward_scheme = SimpleProfit()

    observer = RollingEpisodicObserver(
        portfolio=portfolio,
        feed=feed,
        renderer_feed=renderer_feed,
        window_size=60,
        episode_duration=dt.timedelta(days=7),
        min_periods=None
    )
    stopper = stoppers.MaxLossStopper(
        max_allowed_loss=0.5
    )
    env = TradingEnv(
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        observer=observer,
        stopper=stopper,
        informer=informers.TensorTradeInformer(),
        # renderer=renderers.EmptyRenderer()
        renderer=PlotlyCustomChart()
    )
    agent = A2CAgent(env)
    agent.train(n_steps=None, n_episodes=100, render_interval=None)

    print(portfolio.ledger.as_frame().head(7))
