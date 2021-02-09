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
* [ ] Import Twitter Dataset
* [ ] Create Twitter Dataset
* [ ] Configure persistent mongodb
* [ ] Import Binance data
* [ ] Scrape Binance data
* [ ] Fix 429 too many requests for twitter
