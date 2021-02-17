import logging
import random
import time

import click
from tensortrade.agents import DQNAgent, A2CAgent
from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.env.default import actions, rewards, observers, stoppers, renderers, informers
from tensortrade.env.default.actions import SimpleOrders
from tensortrade.env.default.observers import ObservationHistory
from tensortrade.env.generic import TradingEnv
from tensortrade.feed import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet
import tensortrade.env.default as default

from trading.env.observers import FastObservationHistory, RollingEpisodicObserver


def rsi(price: Stream[float], period: float) -> Stream[float]:
    r = price.diff()
    upside = r.clamp_min(0).abs()
    downside = r.clamp_max(0).abs()
    rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
    return 100 * (1 - (1 + rs) ** -1)


def macd(price: Stream[float], fast: float, slow: float, signal: float) -> Stream[float]:
    fm = price.ewm(span=fast, adjust=False).mean()
    sm = price.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    signal = md - md.ewm(span=signal, adjust=False).mean()
    return signal


@click.command()
def test_runner():
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    print('hello')

    cdd = CryptoDataDownload()
    data = cdd.fetch("Bitstamp", "USD", "BTC", "1h")
    print(data.tail())

    features = []
    for c in data.columns[1:]:
        s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
        features += [s]
    cp = Stream.select(features, lambda s: s.name == "close")
    features = [
        Stream.source(list(data['date']), dtype="datetime64").rename('timestamp'),
        cp.log().diff().rename("lr"),
        rsi(cp, period=20).rename("rsi"),
        macd(cp, fast=10, slow=50, signal=5).rename("macd")
    ]
    feed = DataFeed(features)
    feed.compile()

    bitstamp = Exchange("bitstamp", service=execute_order)(
        Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
    )
    portfolio = Portfolio(USD, [
        Wallet(bitstamp, 10000 * USD),
        Wallet(bitstamp, 0 * BTC)
    ])
    renderer_feed = DataFeed([
        Stream.source(list(data["date"])).rename("date"),
        Stream.source(list(data["open"]), dtype="float").rename("open"),
        Stream.source(list(data["high"]), dtype="float").rename("high"),
        Stream.source(list(data["low"]), dtype="float").rename("low"),
        Stream.source(list(data["close"]), dtype="float").rename("close"),
        Stream.source(list(data["volume"]), dtype="float").rename("volume")
    ])

    action_scheme = SimpleOrders(trade_sizes=2)
    reward_scheme = rewards.get('risk-adjusted')
    action_scheme.portfolio = portfolio

    observer = RollingEpisodicObserver(
        portfolio=portfolio,
        feed=feed,
        renderer_feed=renderer_feed,
        window_size=20,
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
        renderer=renderers.PlotlyTradingChart()
    )
    agent = A2CAgent(env)
    agent.train(n_steps=2000, n_episodes=10)

    i = 0
    pass
