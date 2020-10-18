#!/usr/bin/python

import articles
import googletrans
from googletrans import Translator

class Translate:

    def __init__(self, title):
        print("initialized")
        translator = Translator()
        result = translator.translate(title, dest='fr')
        print(result.text)

Article = articles.Dataset()
title = Article.get_literal_random_title()
translated_title = Translate(title)