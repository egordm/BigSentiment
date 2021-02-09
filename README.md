# BigHead
## Usage
Requirements:
* docker

Setup:
* Start mongodb with: `docker-compose up`
* Create virtual env: `python -m venv venv`
* Install requirements `pip install -r requirements.txt`
* Copy `.env.example` to `.env` and fill in data (requires a discord bot and webhook to a logging channel)

Example commands:
```shell
scraper twitter-old --query='#bitcoin'
scraper twitter --query='#bitcoin'
scraper news

# Add --help to see more args
```

### Extras
```shell
kaggle datasets download --unzip -p kaggle supertaz/pretrained-bert-including-scripts
kaggle datasets download --unzip -p kaggle takuok/glove840b300dtxt
kaggle datasets download --unzip -p kaggle yekenot/fasttext-crawl-300d-2m
kaggle datasets download --unzip -p kaggle alaix14/bitcoin-tweets-20160101-to-20190329
```



## TODO
* [x] Scrape twitter
* [x] Scrape twitter (historical posts)
* [x] Scrape https://www.newsbtc.com
* [x] Scrape https://www.coindesk.com
* [x] Scrape https://cointelegraph.com
* [ ] Scrape Reddit (r/cryptomoonshots, r/bitcoin, r/satoshistreetbets, r/cryptocurrency)
* [ ] Scrape 4chan(archives) (note 4chan has api)
* [ ] Scrape warosu
* [ ] Create News Dataset
* [x] Import Twitter Dataset
* [x] Create Twitter Dataset
* [x] Configure persistent mongodb
* [x] Import Binance data
* [x] Scrape Binance data
* [ ] Devise architecture for sentiment analysis
* [ ] Create Label generation for sentiment analysis
* [ ] Create Batch generation for sentiment analysis

## Instpiration / References
* https://github.com/Drabble/TwitterSentimentAndCryptocurrencies
* https://www.kaggle.com/kyakovlev/preprocessing-bert-public/data
