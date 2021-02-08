import os
import shutil
import pandas as pd


def batched(cursor, batch_size):
    batch = []
    for doc in cursor:
        batch.append(doc)
        if batch and not len(batch) % batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def batched_dataset(cursor, dir, dataset_size, batch_size):
    shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)
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
