from discord.ext import commands


class Bot(commands.Cog):
    def __init__(self, bot):
        self.bot = bot