import os
import pathlib
from collections import defaultdict
from typing import List

import pandas as pd

from tensortrade.feed import Stream

from utils.datasets import DATASET_DIR


class LocalDatasource:
    def __init__(self) -> None:
        super().__init__()
        files = pathlib.Path(os.path.join(DATASET_DIR, 'market')).glob("*.parquet")
        self.dataset = defaultdict(lambda: defaultdict(lambda: {}))
        for file in files:
            source, pair, timestep = os.path.basename(file).split('.')[0].split('-')
            self.dataset[source][pair][timestep] = file

    def fetch(self, source: str, pair: str, timestep: str) -> pd.DataFrame:
        return pd.read_parquet(self.dataset[source][pair][timestep])

    def renderer_transform(self, data) -> List['Stream']:
        return [
            Stream.source(list(data["timestamp"])).rename("date"),
            Stream.source(list(data["open"]), dtype="float").rename("open"),
            Stream.source(list(data["high"]), dtype="float").rename("high"),
            Stream.source(list(data["low"]), dtype="float").rename("low"),
            Stream.source(list(data["close"]), dtype="float").rename("close"),
            Stream.source(list(data["volume"]), dtype="float").rename("volume")
        ]
