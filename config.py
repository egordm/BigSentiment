import logging
import sys

import pymongo
import os
import discord as dsc
from dotenv import load_dotenv
import asyncio
from discord_handler import DiscordHandler

load_dotenv()


def mongodb():
    return pymongo.MongoClient(
        os.getenv('MONGODB_HOST'),
        username=os.getenv('MONGODB_USER'),
        password=os.getenv('MONGODB_PASS'),
    )


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logging.getLogger('scrapy').propagate = True
logging.getLogger('scrapy').setLevel(logging.DEBUG)

FORMAT = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M")
discord_handler = DiscordHandler(os.getenv('DISCORD_LOG_WEBHOOK'), 'Basement')
stream_handler = logging.StreamHandler(sys.stdout)

discord_handler.setLevel(logging.WARNING)
stream_handler.setLevel(logging.DEBUG)

discord_handler.setFormatter(FORMAT)
stream_handler.setFormatter(FORMAT)

logger.addHandler(discord_handler)
logger.addHandler(stream_handler)

