import logging
from datetime import datetime
from typing import Optional

import requests
import scrapy
from scrapy import signals
from scrapy.http import HtmlResponse

SEARCH_URL = 'https://www.newsbtc.com/'
HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:48.0) Gecko/20100101 Firefox/48.0'}


class NewsbtcSpider(scrapy.Spider):
    name = 'NewsbtcSpider'
    query = 0
    from_date: Optional[datetime] = None
    SOURCE = 'newsbtc'

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(NewsbtcSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def start_requests(self):
        for page in range(100000):
            data = requests.post(SEARCH_URL, params={
                'ajax-request': 'jnews'
            }, data={
                'lang': 'en',
                'action': 'jnews_module_ajax_jnews_block_3',
                'data[attribute][number_post]': 100,
                'data[current_page]': page,
                'data[attribute][post_type]': 'post',
                'data[attribute][sort_by]': 'latest',
                'data[attribute][include_category]': self.query,
            }, headers=HEADERS).json()

            response = HtmlResponse(url=SEARCH_URL, body=data['content'], encoding='utf-8')
            articles = list(response.css('article'))

            if len(articles) == 0:
                break

            for item in articles:
                url = item.css('.jeg_post_title a').attrib['href']
                date = datetime.strptime(item.css('.jeg_meta_date a::text').get().strip(), '%B %d, %Y')

                if self.from_date and self.from_date >= date:
                    return

                yield scrapy.Request(
                    url,
                    cb_kwargs={'item': item},
                    dont_filter=True,
                    headers=HEADERS
                )

    def parse(self, response, **kwargs):
        item = kwargs['item']
        text = '\n\n'.join([
            item.get()
            for item in response.css('.content-inner p::text')
        ])
        tags = [
            item.get()
            for item in response.css('.jeg_post_tags a::text')
        ]
        url = item.css('.jeg_post_title a').attrib['href']
        slug = url.rstrip('/').rsplit('/', 1)[-1]
        date = datetime.strptime(item.css('.jeg_meta_date a::text').get().strip(), '%B %d, %Y')

        return {
            'source': self.SOURCE,
            'slug': slug,
            'title': item.css('.jeg_post_title a::text').get(),
            'summary': item.css('.jeg_post_excerpt p::text').get(),
            'text': text,
            'author': item.css('.jeg_meta_author a::text').get(),
            'topics': tags,
            'date': date,
        }

    def spider_closed(self, spider):
        stats = spider.crawler.stats.get_stats()
        numcount = str(stats.get('item_scraped_count', 0))
        logging.info(f'Finished scraping NewsBtc. Results: {numcount} Query: {self.query}')
