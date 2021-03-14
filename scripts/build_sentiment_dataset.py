import os
import pathlib
import re

import datasets
import emoji
import pandas as pd
from datasets import load_dataset, DatasetDict
from transformers import ElectraTokenizerFast

from utils.datasets import ensure_dataset, load_helper_file

pretokenize = True
# Load helpers for dataset transformation
vocabulary_bert = set(load_helper_file('helper_bert_uncased_vocabulary'))
vocabulary_words = load_helper_file('custom_vocabulary_words')
vocabulary_extra = load_helper_file('custom_vocabulary_extra')
vocabulary = vocabulary_bert.union(set(vocabulary_words)).union(set(vocabulary_extra))
emoji_dict = set(e for lang in emoji.UNICODE_EMOJI.values() for e in lang)

tokenizer_custom = {
    '@HTAG': '[HTAG]',
    '@USR': '[USR]',
    '@CURR': '[CURR]',
    '@EMOJI': '[EMOJI]',
    '@URL': '[URL]',
    '@TIME': '[TIME]',
    '@DATE': '[DATE]',
    '@NUM': '[NUM]'
}

re_num = re.compile('(@NUM|\[|\])')


def process_text(text):
    words = text.split()
    for i, word in enumerate(words):
        if word in vocabulary and not word.startswith('@'):
            continue
        if word.startswith('@NUM'):
            value = round(float(re_num.sub('', word)), 3)
            words[i] = '[NUM] ' + ('%f' % value).rstrip('0').rstrip('.')
            continue
        if prefix := next(filter(words[i].startswith, tokenizer_custom.keys()), None):
            value = words[i].replace(prefix, '').replace(']', '').replace('[', '')
            words[i] = f'{tokenizer_custom[prefix]} {value}'
            continue
        if word in emoji_dict:
            words[i] = '[EMOJI]'
            continue
    return {'text': ' '.join(words)}


# Load the curate vocabulary into the tokenizer
VOCAB_FILE = '../data/models/discriminator/vocab.txt'
tokenizer = ElectraTokenizerFast(vocab_file=VOCAB_FILE)
tokenizer.add_special_tokens({
    'additional_special_tokens': list(tokenizer_custom.values())
})

# Load labelled data
dataset = load_dataset(
    './scripts/ds/parquet.py',
    data_files=list(pathlib.Path("../data/bitcoin_twitter_labeled/").glob("part_*.parquet")),
    cache_dir='./cache',
    keep_in_memory=True,
)
dataset = dataset.sort('created_at', keep_in_memory=True)
features = list(dataset['train'].features.keys())
# Filter away all the records with no values in sentiment columns
dataset = dataset.filter(
    lambda *x: all(x), input_columns=[f for f in features if f.startswith('feat')],
    keep_in_memory=True,
    batch_size=100000
)
# Apply text preprocessing
dataset = dataset.map(process_text, input_columns=['text'], keep_in_memory=True)
# PreTokenize the text
if pretokenize:
    dataset = dataset.map(lambda text: {'input_ids': tokenizer.encode(text)}, input_columns=['text'], keep_in_memory=True)
# Shuffle the dataset thoroughly
dataset = dataset.shuffle(seed=42)
# Split off the training dataset
train_test_dataset = dataset['train'].train_test_split(test_size=0.1)
# Split off the validation and test datasets
test_validate_dataset = train_test_dataset['test'].train_test_split(test_size=0.5)
# Save the finel dataset
train_test_valid_dataset = DatasetDict({
    'train': train_test_dataset['train'],
    'test': test_validate_dataset['test'],
    'valid': test_validate_dataset['train']
})
ensure_dataset('../data/sentiment_dataset')
for k, v in train_test_valid_dataset.items():
    v.to_parquet(os.path.join('../data/sentiment_dataset', f'{k}.parquet'))
