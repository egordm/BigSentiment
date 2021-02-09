import os
import random
import shutil
import re
import sys
from multiprocessing import Pool

import pandas as pd
import numpy as np
import psutil
import torch


def batched(cursor, batch_size):
    batch = []
    for doc in cursor:
        batch.append(doc)
        if batch and not len(batch) % batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def ensure_dataset(dir, delete=False):
    if delete and os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)


def batched_dataset(cursor, dir, dataset_size, batch_size):
    ensure_dataset(dir, delete=True)
    i = 0
    df = pd.DataFrame()
    for batch in batched(cursor, batch_size):
        df = df.append(batch, ignore_index=True)
        if len(df) >= dataset_size:
            df.to_parquet(os.path.join(dir, f'part_{i}.parquet'))
            print(f'Writing part {i}')
            df = pd.DataFrame()
            i += 1

    if len(df) >= 0:
        df.to_parquet(os.path.join(dir, f'part_{i}.parquet'))
        print(f'Writing part {i}')


# Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30, 2)


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


# Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    if 'torch' in sys.modules:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


# Domain Search
re_3986_enhanced = re.compile(r"""
        # Parse and capture RFC-3986 Generic URI components.
        ^                                    # anchor to beginning of string
        (?:  (?P<scheme>    [^:/?#\s]+):// )?  # capture optional scheme
        (?:(?P<authority>  [^/?#\s]*)  )?  # capture optional authority
             (?P<path>        [^?#\s]*)      # capture required path
        (?:\?(?P<query>        [^#\s]*)  )?  # capture optional query
        (?:\#(?P<fragment>      [^\s]*)  )?  # capture optional fragment
        $                                    # anchor to end of string
        """, re.MULTILINE | re.VERBOSE)

re_domain = re.compile(r"""
        # Pick out top two levels of DNS domain from authority.
        (?P<domain>[^.]+\.[A-Za-z]{2,6})  # $domain: top two domain levels.
        (?::[0-9]*)?                      # Optional port number.
        $                                 # Anchor to end of string.
        """,
                       re.MULTILINE | re.VERBOSE)


def domain_search(text):
    try:
        return re_domain.search(re_3986_enhanced.match(text).group('authority')).group('domain')
    except:
        return 'url'


# Multiprocessing Run.
# :df - DataFrame to split                      # type: pandas DataFrame
# :func - Function to apply on each split       # type: python function
# This function is NOT 'bulletproof', be carefull and pass only correct types of variables.
def df_parallelize_run(df, func):
    num_partitions, num_cores = 16, psutil.cpu_count()  # number of partitions and cores
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
