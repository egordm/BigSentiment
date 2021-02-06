import logging
from typing import Optional

import requests
import scrapy
from datetime import datetime
from scrapy import signals

SEARCH_URL = 'https://www.coindesk.com/wp-json/v1/search'
HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:48.0) Gecko/20100101 Firefox/48.0'}


class CoindeskSpider(scrapy.Spider):
    name = 'CoindeskSpider'
    query = ''
    from_date: Optional[datetime] = None
    SOURCE = 'coindesk'

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(CoindeskSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def start_requests(self):
        for page in range(100000):
            data = requests.get(SEARCH_URL, {
                'keyword': self.query,
                'page': page
            }).json()

            if len(data) == 0:
                break

            for item in data['results']:
                date = datetime.fromisoformat(item['date'])
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
            for item in response.css('.article-pharagraph p.text::text')
        ])
        tags = [
            item.get()
            for item in response.css('.tags .tag a::text')
        ]

        return {
            'source': self.SOURCE,
            'slug': meta['slug'],
            'title': meta['title'],
            'summary': meta['text'],
            'text': text,
            'author': meta['author'][0]['name'],
            'topics': tags,
            'date': datetime.fromisoformat(meta['date']),
        }

    def spider_closed(self, spider):
        stats = spider.crawler.stats.get_stats()
        numcount = str(stats.get('item_scraped_count', 0))
        logging.info(f'Finished scraping Coindesk. Results: {numcount} Query: {self.query}')
