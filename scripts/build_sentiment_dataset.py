import os
import pathlib
import re

import emoji
import pandas as pd
from datasets import DatasetDict, load_dataset
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
    return ' '.join(words)


# Load the curate vocabulary into the tokenizer
VOCAB_FILE = '../data/models/discriminator/vocab.txt'
tokenizer = ElectraTokenizerFast(vocab_file=VOCAB_FILE)
tokenizer.add_special_tokens({
    'additional_special_tokens': list(tokenizer_custom.values())
})

# Preprocess the initial data
OUTPUT_PATH = "../data/bitcoin_twitter_test_labeled_normalized_tokenized/"
ensure_dataset(OUTPUT_PATH, delete=False)
files = pathlib.Path("../data/bitcoin_twitter_test_labeled_normalized/").glob("part_*.parquet")
for chunk, file in enumerate(files):
    print(f'Processing {chunk}')
    df = pd.read_parquet(file)
    features = [str(f) for f in df.columns if f.startswith('feat')]
    print(features)
    df['text'] = df['text'].map(process_text)
    df['input_ids'] = df['text'].map(lambda text: tokenizer.encode(text))
    df['labels'] = df.apply(lambda x: list(x[features].to_numpy()), axis=1)
    df.drop(columns=['text', 'follower_count', 'retweet_count', *features], inplace=True)
    df.to_parquet(os.path.join(OUTPUT_PATH, f"part_{chunk}.parquet"))

# Load labelled data
dataset = load_dataset(
    './scripts/ds/parquet.py',
    data_files=list(pathlib.Path(OUTPUT_PATH).glob("part_*.parquet")),
    cache_dir='./cache',
)
# Shuffle the dataset thoroughly
dataset = dataset.shuffle(seed=42)

ensure_dataset('../data/sentiment_dataset_test')
df = dataset['train'].to_parquet('../data/sentiment_dataset/test2', 4)
exit()

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
ensure_dataset('../data/sentiment_dataset_test')
PARTITONS = [12, 2, 2]
for n_partitons, (k, v) in zip(PARTITONS, train_test_valid_dataset.items()):
    path = os.path.join('../data/sentiment_dataset_test', f'{k}')
    os.makedirs(path, exist_ok=True)
    print(f'Processing: {k}')
    df = v.to_parquet(path, n_partitons)
    uu = 0


#  def to_parquet(
#      self,
#      path_or_buf: Union[PathLike, BinaryIO],
#      n_partitions: int = 1,
#      **to_parquet_kwargs,
# ):
#      """Exports the dataset to parquet
#
#      Args:
#          path_or_buf (``PathLike`` or ``FileOrBuffer``): Either a path to a file or a BinaryIO.
#          to_parquet_kwargs: Parameters to pass to arrow's :func:`arrows.parquet.write_table`
#      """
#      step_size = int(np.ceil(len(self) / float(n_partitions)))
#      for from_range in tqdm(range(0, len(self), step_size)):
#          to_range = np.minimum(from_range + step_size, len(self))
#          pq.write_to_dataset(query_table(
#              pa_table=self._data,
#              key=slice(from_range, to_range),
#              indices=self._indices.column(0) if self._indices is not None else None,
#          ), path_or_buf, **to_parquet_kwargs)