import logging
from typing import Optional

import requests
import scrapy
from datetime import datetime

from scrapy import signals
from scrapy.http import HtmlResponse

SEARCH_URL = 'https://cointelegraph.com/api/v1/content/search/result'
SEARCH_URL2 = 'https://cointelegraph.com/search'
HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:48.0) Gecko/20100101 Firefox/48.0'}


class CointelegraphSpider(scrapy.Spider):
    name = 'CointelegraphSpider'
    query = ''
    from_date: Optional[datetime] = None
    SOURCE = 'cointelegraph'

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(CointelegraphSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def retrieve_csrf(self):
        preimage = requests.get(SEARCH_URL2, {
            'query': self.query
        }, headers=HEADERS)
        preimage_respose = HtmlResponse(url=SEARCH_URL2, body=preimage.content, encoding='utf-8')
        csrf_token = preimage_respose.xpath("//meta[@name='csrf-token']/@content")[0].extract()
        return csrf_token

    def start_requests(self):
        for page in range(100000):
            csrf_token = self.retrieve_csrf()

            data = requests.post(SEARCH_URL, json={
                'query': self.query,
                'page': page,
                'token': csrf_token,
            }, headers=HEADERS).json()

            if len(data) == 0:
                break

            for item in data['posts']:
                if not item:
                    continue

                date = datetime.fromisoformat(item['publishedW3']).replace(tzinfo=None)
                if self.from_date and self.from_date >= date:
                    return

                yield scrapy.Request(
                    item['url'],
                    cb_kwargs={'item': item},
                    dont_filter=True,
                    headers=HEADERS
                )

    def parse(self, response, **kwargs):
        meta = kwargs['item']
        text = '\n\n'.join([
            item.get()
            for item in response.css('.post-content p::text')
        ])
        tags = [
            item.get()
            for item in response.css('.tags-list__link::text')
        ]

        return {
            'source': self.SOURCE,
            'slug': meta['id'],
            'title': meta['title'],
            'summary': meta['lead'],
            'text': text,
            'author': meta['author_title'],
            'topics': tags,
            'date': datetime.fromisoformat(meta['publishedW3']).replace(tzinfo=None),
        }

    def spider_closed(self, spider):
        stats = spider.crawler.stats.get_stats()
        numcount = str(stats.get('item_scraped_count', 0))
        logging.info(f'Finished scraping CoinTelegraph. Results: {numcount} Query: {self.query}')
