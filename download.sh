#!/bin/bash
[ ! -d "data/kaggle/pretrained-bert-including-scripts" ] && kaggle datasets download --unzip -p data/kaggle supertaz/pretrained-bert-including-scripts
[ ! -d "data/kaggle/glove840b300dtxt" ] && kaggle datasets download --unzip -p data/kaggle takuok/glove840b300dtxt
[ ! -d "data/kaggle/fasttext-crawl-300d-2m" ] && kaggle datasets download --unzip -p data/kaggle yekenot/fasttext-crawl-300d-2m
[ ! -d "data/kaggle/bitcoin-tweets-20160101-to-20190329" ] && kaggle datasets download --unzip -p data/kaggle alaix14/bitcoin-tweets-20160101-to-20190329

if [ ! -d "data/cryptodata" ]; then
  mkdir -p data/cryptodata
  wget -P data/cryptodata --no-check-certificate https://www.cryptodatadownload.com/cdd/Bitstamp_BTCUSD_minute.csv
  wget -P data/cryptodata --no-check-certificate https://www.cryptodatadownload.com/cdd/Bitstamp_ETHUSD_minute.csv
  wget -P data/cryptodata --no-check-certificate https://www.cryptodatadownload.com/cdd/Bitstamp_LTCUSD_minute.csv
  wget -P data/cryptodata --no-check-certificate https://www.cryptodatadownload.com/cdd/Bitstamp_XRPUSD_minute.csv
  wget -P data/cryptodata --no-check-certificate https://www.cryptodatadownload.com/cdd/Bitstamp_BCHUSD_minute.csv
fi