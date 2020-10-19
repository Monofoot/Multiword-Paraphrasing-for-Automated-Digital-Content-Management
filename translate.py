#!/usr/bin/python

import articles
import googletrans
from googletrans import Translator

class Translate:

    def __init__(self, title):
        self.french = self.translate_to(title, "fr")
        self.french_to_english = self.translate_to(self.french, "en")
        print("Initial, in English:\n ", title)
        print("Translated to French:\n ", self.french)
        print("Translated from French to English:\n ", self.french_to_english)


        #for language in googletrans.LANGUAGES:
            #print(language)
            # Here we can run the translation for each known language

    def translate_to(self, title, language):
        translator = Translator()
        result = translator.translate(title, dest=language)
        return result.text

    def get_translation(self):
        return self.french
    
    def get_backtranslation(self):
        return self.french_to_english

if __name__ == "__main__":
    Article = articles.Dataset()
    title = Article.get_literal_random_title()
    translated_title = Translate(title)
    title_test = translated_title.get_translation()
    french_to_english = translated_title.get_backtranslation()
    Article.create_digraph_from_untagged_data_french(title_test)
    Article.create_digraph_from_untagged_data_backtranslate(french_to_english)
