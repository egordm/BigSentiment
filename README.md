# BigSentiment
Sentiment data extraction and collection pipeline targeted but not limited to cryptocurrency markets. 
The toolset provides various scripts to pull real data from sources such as twitter, coindesk, cointelegraph and other news sites. 
Addionally it provides a pipeline for cleaning, processing and sentiment extraction from the pulled documents.

## Data Collection Usage
Requirements:
* docker

Setup:
* Start mongodb with: `docker-compose up`
* Create virtual env: `python -m venv venv`
* Install requirements `pip install -r requirements.txt`
* Copy `.env.example` to `.env` and fill in data (requires a discord bot and webhook to a logging channel)

Example commands:
```shell
# Twitter scraping based on official api
scraper twitter-old --query='#bitcoin' 
# Twitter scraping based on unofficial api
scraper twitter --query='#bitcoin' 
# Collects articles from sites such as cointelegraph, coindesk and newsbtc
scraper news 

# Add --help to see more args
```

## Sentiment Extraction Usage
Once that data is collected in the mongodb database, one can start preprocessing it and feeding it to the [sentiment extraction model](https://github.com/EgorDm/electra-sentiment)

Data cleaning/preprocessing scripts:
```shell
python build_twitter_labels.py
python normalize_twitter_labels.py
python build_sentiment_dataset.py
```

Once dataset is built `data/sentiment_dataset`, run it through pretrained sentiment extraction model:
```shell
git clone https://github.com/EgorDm/electra-sentiment.git

# Write hparams.json
{
  "do_train": "true",
  "do_eval": "false",
  "model_size": "small",
  "do_lower_case": "true",
  "vocab_size": 16537
}

# Build the tfrecords dataset
python build_pretraining_dataset.py --corpus-dir data/sentiment_dataset \
   --vocab-file data/vocab.txt --output-dir data/pretrain_tfrecords \
   --max-seq-length 128 --blanks-separate-docs True --do-lower-case --num-processes 12
   
# Pretrain (optional) otherwise use pretrained model (see below)
run_pretraining.py --data-dir ./data --model-name sentiment_dataset --hparams hparams.json

# Finetune (optional) otherwise use pretrained model (see below
run_finetuning.py --data-dir ./data --model-name sentiment_dataset --hparams hparams.json

# Evaluate and generate model outputs
run_eval.py --data-dir ./data --model-name sentiment_dataset --hparams hparams.json
```

Download model from the [releases](https://github.com/EgorDm/electra-sentiment/releases) and extract it in the data folder of the cloned repo (after building tfrecords)


### Sentiment extraction
Copy electra finetuned checkpoints to `data/models`

Convert the checkpoints with `notebooks/sentiment/convert_checkpoints.ipynb`

Preview trained embeddings with `notebooks/sentiment/test_electra.ipynb`

Preview sentemnt extraction results with `notebooks/sentiment/convert_checkpoints.ipynb`

### Extras
```shell
kaggle datasets download --unzip -p data/kaggle supertaz/pretrained-bert-including-scripts
kaggle datasets download --unzip -p data/kaggle takuok/glove840b300dtxt
kaggle datasets download --unzip -p data/kaggle yekenot/fasttext-crawl-300d-2m
kaggle datasets download --unzip -p data/kaggle alaix14/bitcoin-tweets-20160101-to-20190329
```

Backup mongodb
```shell
mongodump -u root -p bigbrain --authenticationDatabase admin --db scrapper --gzip --archive > ./bighead-premerge.gz
```

## Roadmap
* [x] Scrape twitter
* [x] Scrape twitter (historical posts)
* [x] Scrape https://www.newsbtc.com
* [x] Scrape https://www.coindesk.com
* [x] Scrape https://cointelegraph.com
* [x] Scrape https://coinmarketcap.com
* [ ] Scrape Reddit (r/cryptomoonshots, r/bitcoin, r/satoshistreetbets, r/cryptocurrency)
* [ ] Scrape 4chan(archives) (note 4chan has api)
* [ ] Scrape warosu
* [x] Create News Dataset
* [x] Import Twitter Dataset
* [x] Create Twitter Dataset
* [x] Configure persistent mongodb
* [x] Import Binance data
* [x] Scrape Binance data
* [x] Build sentiment analysis pipeline
* [ ] Document arichitecture for sentiment analysis
* [ ] Integrate data collection pipeline in continuous workflow (connect the scripts)

## Instpiration / References
* https://github.com/Drabble/TwitterSentimentAndCryptocurrencies
* https://www.kaggle.com/kyakovlev/preprocessing-bert-public/data
* see docs folder