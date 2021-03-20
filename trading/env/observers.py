from typing import List

import datetime as dt
import numpy as np
import wandb

from gym.spaces import Box, Space
from random import randrange

from tensortrade.env.default.observers import _create_internal_streams
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.env.generic import Observer

from tensortrade.oms.wallets import Portfolio


class FastObservationHistory(object):
    def __init__(self, window_size: int, n_features: int) -> None:
        self.window_size = window_size
        self.n_features = n_features
        self.rows = np.zeros((self.window_size, self.n_features))
        self.n_observations = 0

    def push(self, row: dict) -> None:
        self.rows = np.roll(self.rows, -1, axis=0)
        self.rows[-1] = list(row.values())
        self.n_observations += 1

    def observe(self) -> 'np.array':
        return np.nan_to_num(self.rows)

    def reset(self) -> None:
        self.rows = np.zeros((self.window_size, self.n_features))
        self.n_observations = 0


class RollingEpisodicObserver(Observer):
    def __init__(
            self,
            portfolio: Portfolio,
            feed: DataFeed = None,
            renderer_feed: DataFeed = None,
            episode_duration: dt.timedelta = dt.timedelta(days=1),
            window_size: int = 1,
            **kwargs
    ) -> None:
        # Data feeds
        internal_group = Stream.group(_create_internal_streams(portfolio)).rename("internal")
        external_group = Stream.group(feed.inputs).rename("external")
        additional_groups = [
            Stream.group(renderer_feed.inputs).rename("renderer")
        ] if renderer_feed else []

        self.feed = DataFeed([
                                 internal_group,
                                 external_group
                             ] + additional_groups)
        self.feed = self.feed.attach(portfolio)

        # Parameters
        self.episode_duration = episode_duration
        self.window_size = window_size

        # Observation config
        self._observation_dtype = kwargs.get('dtype', np.float32)
        self._observation_lows = kwargs.get('observation_lows', -np.inf)
        self._observation_highs = kwargs.get('observation_highs', np.inf)

        initial_obs = self.feed.next()["external"]
        initial_obs.pop('timestamp', None)
        n_features = len(initial_obs.keys())
        self._observation_space = Box(
            low=self._observation_lows,
            high=self._observation_highs,
            shape=(self.window_size, n_features),
            dtype=self._observation_dtype
        )
        self.history = FastObservationHistory(window_size, n_features)

        # State variables
        self.renderer_history = []
        self.stop = False
        self.end_dt = None
        self.first_observation = None
        self.epoch = 0
        self.feed.reset()
        self.warmup()

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    def warmup(self) -> None:
        """Warms up the data feed."""
        if self.history.n_observations < self.window_size:
            for _ in range(self.window_size):
                if self.has_next():
                    obs_row = self.feed.next()["external"]
                    obs_row.pop('timestamp', None)
                    self.history.push(obs_row)

    def observe(self, env: 'TradingEnv') -> np.array:
        """Observes the environment.
        As a consequence of observing the `env`, a new observation is generated
        from the `feed` and stored in the observation history.
        Returns
        -------
        `np.array`
            The current observation of the environment.
        """
        data = self.feed.next()

        # Save renderer information to history
        if "renderer" in data.keys():
            self.renderer_history += [data["renderer"]]
        # Save first observation
        if not self.first_observation:
            self.first_observation = data

        # Push new observation to observation history
        obs_row = data["external"]
        obs_dt = obs_row.pop('timestamp')
        self.history.push(obs_row)

        # Check if episode should be stopped
        if not self.end_dt:
            self.end_dt = obs_dt + self.episode_duration
        elif self.end_dt <= obs_dt:
            self.stop = True

        obs = self.history.observe()
        obs = obs.astype(self._observation_dtype)
        return obs

    def has_next(self) -> bool:
        """Checks if there is another observation to be generated.
        Returns
        -------
        bool
            Whether there is another observation to be generated.
        """
        return self.feed.has_next() and not self.stop

    def reset(self) -> None:
        """Resets the observer"""
        self.renderer_history = []
        self.history.reset()
        if not self.feed.has_next():
            wandb.log(dict(epoch=self.epoch))
            self.epoch += 1
            wandb.log(dict(epoch=self.epoch))
            self.feed.reset()

        self.warmup()
        self.stop = False
        self.end_dt = None
        self.first_observation = None


class FlipFlopObserver(Observer):
    # TODO: implement
    pass
