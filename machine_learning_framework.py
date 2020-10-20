#!/usr/bin/python

import articles
import tensorflow as tf

class Paraphrase:
    """
    Generate a paraphrase based off of learned phrases
    from the MarketMate corpus.
    """
    def __init__(self, title):
        self.Articles = articles.Dataset()
        self.dataset = self.Articles.get_dataset()