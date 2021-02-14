# Credit for some parts to: https://www.kaggle.com/kyakovlev/preprocessing-bert-public
# Number extraction and hashtags is my baby

# General imports|
import pathlib

import pandas as pd
import re, warnings, pickle, itertools, emoji, unicodedata

# custom imports
from gensim.utils import deaccent
from collections import Counter
from bs4 import BeautifulSoup
from utils.datasets import *
from pandarallel import pandarallel
import fasttext

pandarallel.initialize()
warnings.filterwarnings('ignore')
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 200
# %%

## Initial vars

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data/bitcoin_twitter_processed')
HELPER_PATH = os.path.join(ROOT_DIR, 'data/helpers')
LOCAL_TEST = True  ## Local test - for test performance on part of the train set only
verbose = True
WPLACEHOLDER = 'word_placeholder'
URL_TAG = '@URL'
USER_TAG = '@USR'
NUMBER_TAG = '@NUM'
HASH_TAG = '@HTAG'
CURRENCY_TAG = '@CURR'
IMMUTABLES = [WPLACEHOLDER, URL_TAG, USER_TAG, NUMBER_TAG, HASH_TAG, CURRENCY_TAG]

SEED = 42  ## Seed for enviroment
seed_everything(SEED)  ## Seed everything

ensure_dataset(OUTPUT_PATH, delete=True)


# %%

## Preprocess helpers
def place_hold(w, tag=WPLACEHOLDER):
    return tag + '[' + re.sub(' ', '___', w) + ']'


## Helpers
def check_replace(w):
    return not bool(re.search('|'.join(IMMUTABLES), w))


def make_cleaning(s, c_dict):
    if check_replace(s):
        s = s.translate(c_dict)
    return s


def make_dict_cleaning(s, w_dict, skip_check=False):
    # Replaces a word using dict if it is mutable
    if skip_check or check_replace(s):
        s = w_dict.get(s, s)
    return s


# %%

## Get basic helper data

bert_uncased_vocabulary = load_helper_file('helper_bert_uncased_vocabulary')
bert_cased_vocabulary = load_helper_file('helper_bert_cased_vocabulary')
bert_char_list = list(set([c for line in bert_uncased_vocabulary + bert_cased_vocabulary for c in line]))

url_extensions = load_helper_file('helper_url_extensions')
html_tags = load_helper_file('helper_html_tags')
good_chars_dieter = load_helper_file('helper_good_chars_dieter')
bad_chars_dieter = load_helper_file('helper_bad_chars_dieter')
helper_contractions = load_helper_file('helper_contractions')
global_vocabulary = load_helper_file('helper_global_vocabulary')
global_vocabulary_chars = load_helper_file('helper_global_vocabulary_chars')
normalized_chars = load_helper_file('helper_normalized_chars')
white_list_chars = load_helper_file('helper_white_list_chars')
white_list_punct = " '*-.,?!/:;_()[]{}<>=" + '"'
pictograms_to_emoji = load_helper_file('helper_pictograms_to_emoji')
helper_custom_synonyms = load_helper_file('helper_custom_synonyms')
emoji_dict = set(e for lang in emoji.UNICODE_EMOJI.values() for e in lang)

# %%
## Load Data
files = pathlib.Path(os.path.join(ROOT_DIR, 'data/bitcoin_twitter_raw')).glob("part_*.parquet")
for chunk, file in enumerate(files):
    data = pd.read_parquet(file)

    ## Start preprocessing
    texts = data['text']
    local_vocab = bert_uncased_vocabulary
    global_lower = True
    texts = texts.astype(str)
    if verbose: print('#' * 20, 'Initial State:'); check_vocab(texts, local_vocab)


    # %%

    def lower(texts):
        texts = texts.apply(lambda x: x.lower())
        if verbose: print('#' * 10, 'Step - Lowering everything:'); check_vocab(texts, local_vocab)
        return texts


    if global_lower:
        texts = texts.pipe(lower)

    # %%

    # Normalize chars and dots - SEE HELPER FOR DETAILS
    # Global
    texts = texts.apply(lambda x: ' '.join([make_cleaning(i, normalized_chars) for i in x.split()]))
    texts = texts.apply(lambda x: re.sub('\(dot\)', '.', x))
    texts = texts.apply(lambda x: deaccent(x))
    if verbose: print('#' * 10, 'Step - Normalize chars and dots:'); check_vocab(texts, local_vocab)

    # %%

    # Remove 'control' chars
    global_chars_list = list(set([c for line in texts for c in line]))
    chars_dict = {c: '' for c in global_chars_list if unicodedata.category(c)[0] == 'C'}
    texts = texts.apply(lambda x: ' '.join([make_cleaning(i, chars_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Control Chars:'); check_vocab(texts, local_vocab)

    # %%

    # Remove hrefs
    texts = texts.apply(
        lambda x: re.sub(re.findall(r'\<a(.*?)\>', x)[0], '', x) if (len(re.findall(r'\<a (.*?)\>', x)) > 0) and (
                    'href' in re.findall(r'\<a (.*?)\>', x)[0]) else x)
    if verbose: print('#' * 10, 'Step - Remove hrefs:'); check_vocab(texts, local_vocab)

    # %%

    # Convert or remove Bad Symbols
    global_chars_list = list(set([c for line in texts for c in line]))
    chars = ''.join([c for c in global_chars_list if
                     (c not in bert_char_list) and (c not in emoji_dict) and (c not in white_list_chars)])
    chars_dict = {}
    for char in chars:
        try:
            new_char = unicodedata.name(char).split()[-1:][0].lower()
            if len(new_char) == 1:
                chars_dict[ord(char)] = new_char
            else:
                chars_dict[ord(char)] = ''
        except:
            chars_dict[ord(char)] = ''
    texts = texts.apply(lambda x: ' '.join([make_cleaning(i, chars_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Remove Bad Symbols:'); check_vocab(texts, local_vocab)
    if verbose: print(chars)
    if verbose: print_dict(chars_dict)

    # %%

    # Remove Bad Symbols PART 2
    # Global
    global_chars_list = list(set([c for line in texts for c in line]))
    chars = '·' + ''.join([c for c in global_chars_list if
                           (c not in white_list_chars) and (c not in emoji_dict) and (c not in white_list_punct) and (
                                       ord(c) > 256)])
    chars_dict = {}
    for char in chars:
        try:
            new_char = unicodedata.name(char).split()[-1:][0].lower()
            if len(new_char) == 1:
                chars_dict[ord(char)] = new_char
            else:
                chars_dict[ord(char)] = ''
        except:
            chars_dict[ord(char)] = ''
    texts = texts.apply(lambda x: ' '.join([make_cleaning(i, chars_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Remove Bad Symbols PART 2:'); check_vocab(texts, local_vocab)
    if verbose: print(chars)
    if verbose: print_dict(chars_dict)

    # %%

    # Remove html tags
    # Global
    temp_vocab = list(set([c for line in texts for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        if ('<' in word) and ('>' in word):
            for tag in html_tags:
                if ('<' + tag + '>' in word) or ('</' + tag + '>' in word):
                    temp_dict[word] = BeautifulSoup(word, 'html5lib').text
    texts = texts.apply(lambda x: ' '.join([temp_dict.get(i, i) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - HTML tags:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%

    # Remove links (There is valuable information in links (probably you will find a way to use it))
    # Global
    temp_vocab = list(set([c for line in texts for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    url_rule = r'(?P<url>https?://[^\s]+)'
    temp_dict = {k: domain_search(k) for k in temp_vocab if k != re.compile(url_rule).sub('url', k)}

    for word in temp_dict:
        new_value = temp_dict[word]
        if word.find('http') > 2:
            temp_dict[word] = word[:word.find('http')] + ' ' + place_hold(new_value, URL_TAG)
        else:
            temp_dict[word] = place_hold(new_value, URL_TAG)

    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Convert urls part 1:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%

    # Remove twitter links
    temp_dict = {
        f'{URL_TAG}[t.co]': ''
    }
    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict, skip_check=True) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Convert urls part 1.5:'); check_vocab(texts, local_vocab);

    # %%

    # Remove escaped html
    temp_vocab = list(set([c for line in texts for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    symbols = {
        '&quot;': '',
        '&amp;': ' and ',
        '&lt;': '',
        '&gt;': '',
    }
    temp_dict = {}
    for word in temp_vocab:
        if any([rep in word for rep in symbols.keys()]):
            new_word = word
            for rep, to in symbols.items():
                new_word = new_word.replace(rep, to)
            temp_dict[word] = new_word

    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict, skip_check=True) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Remove escaped html:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%

    # Convert urls part 2
    # Global
    temp_vocab = list(set([c for line in texts for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}

    for word in temp_vocab:
        url_check = False
        if 'file:' in word:
            url_check = True
        elif ('http' in word) or ('ww.' in word) or ('.htm' in word) or ('ftp' in word) or ('.php' in word) or (
                '.aspx' in word):
            if 'Aww' not in word:
                for d_zone in url_extensions:
                    if '.' + d_zone in word:
                        url_check = True
                        break
        elif ('/' in word) and ('.' in word):
            for d_zone in url_extensions:
                if '.' + d_zone + '/' in word:
                    url_check = True
                    break

        if url_check:
            temp_dict[word] = place_hold(domain_search(word), URL_TAG)

    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Convert urls part 2:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%

    # Normalize pictograms
    # Local (only unknown words)
    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        if len(re.compile('[a-zA-Z0-9]').sub('', word)) > 2:
            for pict in pictograms_to_emoji:
                if (pict in word) and (len(pict) > 2):
                    temp_dict[word] = word.replace(pict, pictograms_to_emoji[pict])
                elif pict == word:
                    temp_dict[word] = pictograms_to_emoji[pict]

    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Normalize pictograms:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%

    # Isolate emoji
    # Global
    global_chars_list = list(set([c for line in texts for c in line]))
    chars = ''.join([c for c in global_chars_list if c in emoji_dict])
    chars_dict = {ord(c): f' {c} ' for c in chars}
    texts = texts.apply(lambda x: ' '.join([make_cleaning(i, chars_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Isolate emoji:'); check_vocab(texts, local_vocab)
    if verbose: print(chars)

    # %%

    # Duplicated dots, question marks and exclamations
    # Local
    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        new_word = word
        if (Counter(word)['.'] > 1) or (Counter(word)['!'] > 1) or (Counter(word)['?'] > 1) or (Counter(word)[','] > 1):
            if (Counter(word)['.'] > 1):
                new_word = re.sub('\.\.+', ' . . . ', new_word)
            if (Counter(word)['!'] > 1):
                new_word = re.sub('\!\!+', ' ! ! ! ', new_word)
            if (Counter(word)['?'] > 1):
                new_word = re.sub('\?\?+', ' ? ? ? ', new_word)
            if (Counter(word)[','] > 1):
                new_word = re.sub('\,\,+', ' , , , ', new_word)
            temp_dict[word] = new_word
    temp_dict = {k: v for k, v in temp_dict.items() if k != v}
    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Duplicated Chars:'); check_vocab(texts, local_vocab);

    # %%

    # Remove underscore for spam words
    # Local
    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        if (len(re.compile('[a-zA-Z0-9\-\.\,\/\']').sub('', word)) / len(word) > 0.6) and ('_' in word):
            temp_dict[word] = re.sub('_', '', word)
    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Remove underscore:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%

    # Isolate spam chars repetition
    # Local
    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        if (len(re.compile('[a-zA-Z0-9\-\.\,\/\']').sub('', word)) / len(word) > 0.6) and (
                len(Counter(word)) == 1) and (len(word) > 2):
            temp_dict[word] = ' '.join([' ' + next(iter(Counter(word).keys())) + ' ' for i in range(1)])
    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Spam chars repetition:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%

    # Normalize pictograms part 2
    # Local (only unknown words)
    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        if len(re.compile('[a-zA-Z0-9]').sub('', word)) > 1:
            for pict in pictograms_to_emoji:
                if pict == word:
                    temp_dict[word] = pictograms_to_emoji[pict]
    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Normalize pictograms part 2:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%

    # Isolate brakets and quotes
    # Global
    chars = '()[]{}<>"'
    chars_dict = {ord(c): f' {c} ' for c in chars}
    texts = texts.apply(lambda x: ' '.join([make_cleaning(i, chars_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Brackets and quotes:'); check_vocab(texts, local_vocab)
    if verbose: print_dict(chars_dict)

    # %%

    # Break short words
    # Global
    temp_vocab = list(set([c for line in texts for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_vocab = [k for k in temp_vocab if len(k) <= 20]

    temp_dict = {}
    for word in temp_vocab:
        if '/' in word and not word.startswith('u/') and not word.startswith('r/'):
            temp_dict[word] = re.sub('/', ' / ', word)

    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Break long words:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)


    # %%

    # Break long words
    def break_long_words(texts):
        temp_vocab = list(set([c for line in texts for c in line.split()]))
        temp_vocab = [k for k in temp_vocab if check_replace(k)]
        temp_vocab = [k for k in temp_vocab if len(k) > 20]

        temp_dict = {}
        for word in temp_vocab:
            if '_' in word:
                temp_dict[word] = re.sub('_', ' ', word)
            elif '/' in word and not word.startswith('u/') and not word.startswith('r/'):
                temp_dict[word] = re.sub('/', ' / ', word)
            elif len(' '.join(word.split('-')).split()) > 2:
                temp_dict[word] = re.sub('-', ' ', word)
            for s in ',.:;':
                if s in word and not re.compile('[+#@$/,.:;-]').sub('', word).isnumeric():
                    temp_dict[word] = word.replace(s, f' {s} ')

        texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
        if verbose: print('#' * 10, 'Step - Break long words:'); check_vocab(texts, local_vocab);
        if verbose: print_dict(temp_dict)
        return texts


    for i in range(3):
        texts = texts.pipe(break_long_words)

    # %%

    # TODO: add number parsing before
    # Diambiguate entities
    # Split words on @,# and $ to clear up ambiguities between entitites
    symbols = '@#$'
    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if (check_replace(k)) and ('@' in k or '#' in k or '$' in k)]

    temp_dict = {}
    for word in temp_vocab:
        for symbol in symbols:
            if symbol not in word: continue
            left, *right = word.split(symbol)
            rightz = symbol.join(right)
            if len(left) > 0 and len(right[0]) > 0 and right[0].isalnum():
                temp_dict[word] = f'{left} {symbol}{rightz}'
            break

    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Disambiguate entities:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)


    # %%

    def custom_synonyms(texts):
        temp_dict = {}
        for wfrom, wto in helper_custom_synonyms.items():
            temp_dict[wfrom] = wto
        texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
        if verbose: print('#' * 10, 'Step - Custom word synonyms:'); check_vocab(texts, local_vocab);
        if verbose: print_dict(temp_dict)
        return texts


    texts = texts.pipe(custom_synonyms)


    # %%

    # Remove/Convert usernames and hashtags
    def extract_entities(texts):
        temp_vocab = list(set([c for line in texts for c in line.split()]))
        temp_vocab = [k for k in temp_vocab if check_replace(k)]
        temp_dict = {}
        for word in temp_vocab:
            if (len(word) > 2) and (word[1:len(word) - 1].replace('\'s', '').replace('_', '').isalnum()):
                new_word = word.replace('\'s', '')
                if not re.compile('[#@$/,.:;]').sub('', new_word).isnumeric():
                    new_word = re.compile('[,.:;]').sub('', new_word)
                    if word.startswith('@'):
                        temp_dict[word] = place_hold(new_word[1:], USER_TAG)
                    elif word.startswith('#'):
                        temp_dict[word] = place_hold(new_word[1:], HASH_TAG)
                    elif word.startswith('u/'):
                        temp_dict[word] = place_hold(new_word[2:], USER_TAG)
                    elif word.startswith('r/'):
                        temp_dict[word] = place_hold(new_word[2:], HASH_TAG)
                    elif word.startswith('$') and word[1:].isalpha():
                        temp_dict[word] = place_hold(new_word[1:], CURRENCY_TAG)
        temp_dict = {k: v for k, v in temp_dict.items() if k != v}
        texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
        if verbose: print('#' * 10, 'Step - UserName and Hashtag:'); check_vocab(texts, local_vocab);
        if verbose: print_dict(temp_dict)
        return texts


    texts = texts.pipe(extract_entities)


    # %%

    # Hashtag and currency union
    def hashtag_currency_union(texts):
        temp_vocab = list(set([c for line in texts for c in line.split()]))
        temp_vocab = set([k for k in temp_vocab if not check_replace(k)])
        temp_dict = {}
        for w in temp_vocab:
            if w.startswith(CURRENCY_TAG):
                if w.replace(CURRENCY_TAG, HASH_TAG) in temp_vocab:
                    temp_dict[w.replace(CURRENCY_TAG, HASH_TAG)] = w
                if w.replace(CURRENCY_TAG, USER_TAG) in temp_vocab:
                    temp_dict[w.replace(CURRENCY_TAG, USER_TAG)] = w
        texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict, skip_check=True) for i in x.split()]))
        if verbose: print('#' * 10, 'Step - Hashtag and currency union:'); check_vocab(texts, local_vocab);
        if verbose: print_dict(temp_dict)
        return texts


    texts = texts.pipe(hashtag_currency_union)

    # %%

    # Remove ending underscore (or add quotation marks???)
    # Local
    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if (check_replace(k)) and ('_' in k)]
    temp_dict = {}
    for word in temp_vocab:
        new_word = word
        if word[len(word) - 1] == '_':
            for i in range(len(word), 0, -1):
                if word[i - 1] != '_':
                    new_word = word[:i]
                    temp_dict[word] = new_word
                    break
    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Remove ending underscore:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%

    # Remove starting underscore
    # Local
    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if (check_replace(k)) and ('_' in k)]
    temp_dict = {}
    for word in temp_vocab:
        new_word = word
        if word[0] == '_':
            for i in range(len(word)):
                if word[i] != '_':
                    new_word = word[i:]
                    temp_dict[word] = new_word
                    break
    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Remove starting underscore:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%

    # End word punctuations
    # Global
    temp_vocab = list(set([c for line in texts for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if (check_replace(k)) and (not k[len(k) - 1].isalnum())]
    temp_dict = {}
    for word in temp_vocab:
        new_word = word
        for i in range(len(word), 0, -1):
            if word[i - 1].isnumeric() and re.compile('[$£%€]').match(word[i]):
                break

            if word[i - 1].isalnum():
                new_word = word[:i] + ' ' + word[i:]
                break
        temp_dict[word] = new_word
    temp_dict = {k: v for k, v in temp_dict.items() if k != v}
    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - End word punctuations:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%

    scale_mapping = {
        'b': 1000000000,
        'bn': 1000000000,
        'bln': 1000000000,
        'billion': 1000000000,
        'm': 1000000,
        'mn': 1000000,
        'mln': 1000000,
        'million': 1000000,
        'k': 1000,
        'thousand': 1000,
        '-': -1,
    }

    translate = {
        '$': 'dollar', '£': 'pound', '%': 'percent', '€': 'euro'
    }

    translate_suffix = {
        'x': 'times'
    }

    translate_prefix = {
        '~': 'around',
        '+-': 'around',
        '±': 'around',
        '@': 'at',
        '=': 'equals',
        '*#': 'ranked',
        '#': 'ranked',
    }


    def serialize_numbers(texts):
        temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
        temp_vocab = [k for k in temp_vocab if check_replace(k)]
        temp_dict = {}
        re_inb = re.compile('[,\'"`]')
        re_num = re.compile('^(~|\+-|±|@|=|#|\*#)?[-@+*^#:]?[$£%€]?(([.:]?[0-9])+)[$£%€]?')
        re_fix = re.compile('^[$£%€][-+][0-9]')
        for word in temp_vocab:
            prefilter = re_inb.sub('', word).replace(',', '.')
            if re_fix.search(prefilter):
                prefilter = prefilter[1] + prefilter[0] + prefilter[2:]
            result = re_num.search(prefilter)

            if result and result.pos == 0:
                # Process combined numbers / ranges in next iteration
                if '-' in word and not word.startswith('-') and not word.startswith('+-'):
                    temp_dict[word] = ' '.join(word.split('-'))
                    continue

                main_part = prefilter[:result.end()]
                prefix = ''
                for prefix_key, prefix_name in translate_prefix.items():
                    if main_part.startswith(prefix_key):
                        prefix = prefix_name
                        main_part = main_part.replace(prefix_key, '', 1)
                        break

                main = re.compile('^[~@+*^#:]').sub('', main_part)
                currency = re.compile('[$£%€]').search(main)
                currency = main[currency.start():currency.end()] if currency else None
                main = re.compile('[$£%€]').sub('', main)
                suffix = prefilter[result.end():]

                multiplier = 1
                if re.compile('\.[0-9]{1,2}$').search(main):  # decimal
                    multiplier *= 0.01 if main[-1].isnumeric() else 0.1
                if '-' in main:  # Neg numbers
                    multiplier *= -1
                    main = main.replace('-', '')
                # Textual scale
                if suffix in scale_mapping:
                    multiplier *= scale_mapping[suffix]
                    suffix = ''
                if suffix in translate_suffix:
                    suffix = translate_suffix[suffix]

                number = round(float(main.replace('.', '').replace(':', '')) * multiplier, 2)
                # print(f'{number}  /  {currency}  /  {suffix}  /  {word}')
                # noinspection PyTypeChecker
                temp_dict[word] = ' '.join(filter(len, [
                    prefix,
                    place_hold(str(number), NUMBER_TAG),
                    translate[currency] if currency else '',
                    suffix
                ]))

        texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
        if verbose: print('#' * 10, 'Step - Serialize numbers:'); check_vocab(texts, local_vocab);
        if verbose: print_dict(temp_dict)
        return texts


    # Clean up numbers
    for i in range(4):
        texts.pipe(serialize_numbers)
        # texts = texts.pipe(serialize_numbers)

    # %%
    # Extract entities again
    texts = texts \
        .pipe(custom_synonyms) \
        .pipe(extract_entities) \
        .pipe(hashtag_currency_union)
    # %%

    # Start word punctuations
    # Global
    temp_vocab = list(set([c for line in texts for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if (check_replace(k)) and (not k[0].isalnum() and k[0] not in ['@', '#', '$'])]
    temp_dict = {}
    for word in temp_vocab:
        new_word = word
        for i in range(len(word)):
            if word[i].isalnum() or word[i] in ['#', '@', '$']:
                new_word = word[:i] + ' ' + word[i:]
                break
        temp_dict[word] = new_word
    temp_dict = {k: v for k, v in temp_dict.items() if k != v}
    # texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Start word punctuations:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%

    # Extract entities again and numbers
    texts = texts.pipe(serialize_numbers)
    texts = texts \
        .pipe(custom_synonyms) \
        .pipe(extract_entities) \
        .pipe(hashtag_currency_union)

    # %%

    # Find and replace acronims
    # Local (only unknown words)
    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        if (Counter(word)['.'] > 1) and (check_replace(word)):
            if (domain_search(word) != '') and (('www' in word) or (Counter(word)['/'] > 3)):
                temp_dict[word] = place_hold('url ' + domain_search(word))
            else:
                if (re.compile('[\.\,]').sub('', word) in local_vocab) and (
                        len(re.compile('[0-9\.\,\-\/\:]').sub('', word)) > 0):
                    temp_dict[word] = place_hold(re.compile('[\.\,]').sub('', word))
    temp_dict = {k: v for k, v in temp_dict.items() if k != v}
    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Find and replace acronims:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%

    # Apply spellchecker for contractions
    # Local (only unknown words)
    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if (check_replace(k)) and ("'" in k)]
    temp_dict = {}
    for word in temp_vocab:
        if word in helper_contractions:
            temp_dict[word] = helper_contractions[word]  # place_hold(helper_contractions[word])
    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Contractions:'); check_vocab(texts, local_vocab)
    if verbose: print_dict(temp_dict)

    # %%

    # Remove 's (DO WE NEED TO REMOVE IT???)
    # Local
    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {k: k[:-2] for k in temp_vocab if (check_replace(k)) and (k.lower()[-2:] == "'s")}
    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Remove "s:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%

    # Convert backslash
    # Global
    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if (check_replace(k)) and ('\\' in k)]
    temp_dict = {k: re.sub('\\\\+', ' / ', k) for k in temp_vocab}
    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Convert backslash:'); check_vocab(texts, local_vocab)
    if verbose: print_dict(temp_dict)

    # %%

    # Extract entities again and numbers
    texts = texts.pipe(serialize_numbers)
    texts = texts \
        .pipe(custom_synonyms) \
        .pipe(extract_entities) \
        .pipe(hashtag_currency_union)

    # %%

    # Try remove duplicated chars (not sure about this!!!!!). TODO check fist against vocab?
    # Local (only unknown words)
    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]

    temp_dict = {}
    temp_vocab_dup = []

    for word in temp_vocab:
        if not word.isalpha():
            continue
        temp_vocab_dup.append(''.join(ch for ch, _ in itertools.groupby(word)))
    temp_vocab_dup = set(temp_vocab_dup)
    temp_vocab_dup = temp_vocab_dup.difference(temp_vocab_dup.difference(set(local_vocab)))

    for word in temp_vocab:
        new_word = ''.join(ch for ch, _ in itertools.groupby(word))
        if new_word in temp_vocab_dup:
            temp_dict[word] = new_word
    temp_dict = {k: v for k, v in temp_dict.items() if (k != v) and (v in local_vocab)}

    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Dup chars (with vocab check):'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%
    # Extract entities again and numbers
    texts = texts.pipe(serialize_numbers)
    texts = texts \
        .pipe(custom_synonyms) \
        .pipe(extract_entities) \
        .pipe(hashtag_currency_union)

    # %%

    # Isolate numbers
    # Local (only unknown words)
    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        if re.compile('[a-zA-Z]').sub('', word) == word:
            if re.compile('[0-9]').sub('', word) != word:
                temp_dict[word] = word

    global_chars_list = list(set([c for line in temp_dict for c in line]))
    chars = ''.join([c for c in global_chars_list if not c.isdigit()])
    chars_dict = {ord(c): f' {c} ' for c in chars}
    temp_dict = {k: place_hold(k) for k in temp_dict}

    # texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Isolate numbers:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%

    # Join dashes
    # Local (only unknown words)
    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]

    temp_dict = {}
    for word in temp_vocab:
        temp_dict[word] = re.sub('\-\-+', '-', word)
    temp_dict = {k: v for k, v in temp_dict.items() if k != v}

    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Join dashes:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%

    # Try join word (Sloooow)
    # Local (only unknown words)
    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if (check_replace(k)) and (Counter(k)['-'] > 1)]

    temp_dict = {}
    for word in temp_vocab:
        new_word = ''.join(['' if c in '-' else c for c in word])
        if (new_word in local_vocab) and (len(new_word) > 3):
            temp_dict[word] = new_word

    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Try Split word:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%

    # Try Split word
    # Local (only unknown words)
    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]

    temp_dict = {}
    for word in temp_vocab:
        if len(re.compile('[a-zA-Z0-9\*]').sub('', word)) > 0:
            chars = re.compile('[a-zA-Z0-9\*]').sub('', word)
            temp_dict[word] = ''.join([' ' + c + ' ' if c in chars else c for c in word])

    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Try Split word:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)


    # %%

    # L33T vocabulary (SLOW)
    # https://simple.wikipedia.org/wiki/Leet
    # Local (only unknown words)
    def convert_leet(word):
        # basic conversion
        word = re.sub('0', 'o', word)
        word = re.sub('1', 'i', word)
        word = re.sub('3', 'e', word)
        word = re.sub('\$', 's', word)
        word = re.sub('\@', 'a', word)
        return word


    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]

    temp_dict = {}
    for word in temp_vocab:
        new_word = convert_leet(word)
        if (new_word != word):
            if (len(word) > 2) and (new_word in local_vocab):
                temp_dict[word] = new_word

    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - L33T (with vocab check):'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%
    # Extract entities again and numbers
    texts = texts.pipe(serialize_numbers)
    texts = texts \
        .pipe(custom_synonyms) \
        .pipe(extract_entities) \
        .pipe(hashtag_currency_union)

    # %%

    # Remove placeholders
    # Global
    temp_vocab = list(set([c for line in texts for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if (not check_replace(k) and k.startswith(WPLACEHOLDER))]
    temp_dict = {}
    for word in temp_vocab:
        temp_dict[word] = re.sub('___', ' ', word[17:-1])
    texts = texts.apply(lambda x: ' '.join([temp_dict.get(i, i) for i in x.split()]))
    texts = texts.apply(lambda x: ' '.join([i for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Open Holded words:'); check_vocab(texts, local_vocab)

    # %%

    # Search multiple form
    # Local | example -> flashlights / flashlight -> False / True
    temp_vocab = check_vocab(texts, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if (k[-1:] == 's') and (len(k) > 4)]
    temp_dict = {k: k[:-1] for k in temp_vocab if (k[:-1] in local_vocab)}
    texts = texts.apply(lambda x: ' '.join([make_dict_cleaning(i, temp_dict) for i in x.split()]))
    if verbose: print('#' * 10, 'Step - Multiple form:'); check_vocab(texts, local_vocab);
    if verbose: print_dict(temp_dict)

    # %%
    # Extract entities again and numbers
    texts = texts.pipe(serialize_numbers)
    texts = texts \
        .pipe(custom_synonyms) \
        .pipe(extract_entities) \
        .pipe(hashtag_currency_union)

    # %%
    # Cut away non english tweets
    model = fasttext.load_model('../../data/kaggle/lid.176.ftz')


    def langcheck(item, min_confidence=0.2):
        text = ' '.join([w for w in item.split() if not w.startswith('@')])
        if len(text) < 3:
            return True
        results = dict(zip(*model.predict(text, k=2)))
        return results.get('__label__en', 0) > min_confidence


    mask = texts.parallel_map(langcheck)
    if verbose: print(f'Deleted: {1 - sum(mask) / len(texts)}')
    texts = texts[mask]
    data = data[mask]
    if verbose: print('#' * 10, 'Step - Language datection:'); check_vocab(texts, local_vocab);

    # %%
    data['text'] = texts
    data.to_parquet(os.path.join(OUTPUT_PATH, f'part_{chunk}.parquet'))
