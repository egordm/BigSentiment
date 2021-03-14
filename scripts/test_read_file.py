from datasets import Dataset, DatasetDict
import os

DATASET_DIR = '../data/sentiment_dataset'
dataset = DatasetDict({
    'train': Dataset.from_parquet(os.path.join(DATASET_DIR, 'train.parquet')),
    'test': Dataset.from_parquet(os.path.join(DATASET_DIR, 'test.parquet')),
    'valid': Dataset.from_parquet(os.path.join(DATASET_DIR, 'valid.parquet'))
})

u = 0