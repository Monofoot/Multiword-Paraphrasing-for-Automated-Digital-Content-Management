#!/usr/bin/python

import titles
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
    corpus = titles.Titles()
    title = corpus.get_literal_random_title()
    Translated = Translate(title)
    title_test = Translated.get_translation()
    french_to_english = Translated.get_backtranslation()
    corpus.create_digraph_from_untagged_translated_data(title_test, "de")
    corpus.create_digraph_from_untagged_data_backtranslate(french_to_english)
