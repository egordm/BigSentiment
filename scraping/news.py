import logging
from datetime import datetime

import click
import pymongo
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from config import mongodb
from scraping.spiders.CoindeskSpider import CoindeskSpider
from scraping.spiders.NewsbtcScrapper import NewsbtcSpider


class MongoDBPipeline(object):
    def __init__(self):
        client = mongodb()
        scrapperdb = client['scrapper']

        self.collection = scrapperdb['news']
        self.collection.ensure_index([
            ('date', pymongo.DESCENDING),
            ('topics', pymongo.ASCENDING),
            ('source', pymongo.ASCENDING),

        ])

    def process_item(self, item, spider):
        self.collection.update({'_id': item['slug']}, {
            '_id': item['slug'],
            **item,
        }, upsert=True)
        return item


@click.command()
@click.option('--query_coindesk', help='Query to fetch data from', required=True)
@click.option('--category_newsbtc', default=0, type=int, help='Query to fetch data from', required=True)
def news(query_coindesk, category_newsbtc):
    settings = get_project_settings()
    settings.set('CONCURRENT_REQUESTS', 1)
    settings.set('ITEM_PIPELINES', {".".join([MongoDBPipeline.__module__, MongoDBPipeline.__name__]): 300})
    process = CrawlerProcess(settings)
    collection = mongodb()['scrapper']['news']

    longago = datetime.fromisoformat('2015-01-01T00:00:00')

    def last_date(source):
        return collection.find({'source': source}).sort({"date": -1}).limit(1)['date']

    latest = last_date(CoindeskSpider.SOURCE)
    process.crawl(
        CoindeskSpider,
        query=query_coindesk,
        from_date=latest.date if latest else longago
    )

    latest = last_date(NewsbtcSpider.SOURCE)
    process.crawl(
        NewsbtcSpider,
        query=category_newsbtc,
        from_date=latest.date if latest else longago
    )

    process.start()
