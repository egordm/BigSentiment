import numpy as np
from tensortrade.agents import Agent


class BaselineAgent(Agent):
    def __init__(
            self,
            env: 'TradingEnv'
    ):
        self.env = env
        pass

    def get_action(self, state: np.ndarray, **kwargs) -> int:
        pass

    def train(
            self,
            n_steps: int = None, n_episodes: int = 10000,
            save_every: int = None, save_path: str = None,
            callback: callable = None, **kwargs
    ) -> float:
        pass

    def save(self, path: str, **kwargs):
        pass

    def restore(self, path: str, **kwargs):
        pass
