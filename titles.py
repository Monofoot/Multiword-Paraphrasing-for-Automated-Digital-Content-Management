#!/usr/bin/python

import collections
import os
import random as rand
import string
from ast import literal_eval
from pathlib import Path

import en_core_web_sm
import matplotlib as mpl

if os.environ.get('DISPLAY','') == '':
    print('No display for matplotlib found.')
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import pandas as pd
import spacy
import visualise_spacy_tree
from networkx.drawing.nx_agraph import graphviz_layout
from nltk.tag import map_tag, pos_tag
from spacy import displacy
from spacy.lang.en import English

from subject_object_extraction import findSVOs

'''
Each title is stored as:
[["What",0,"PRON","dobj",2,"What "],
["coronavirus",1,"PROPN","nsubj",2,"coronavirus "],
["means",2,"VERB","ROOT",2,"means "],
["for",3,"ADP","prep",2,"for "],
["pensions",4,"NOUN","pobj",3,"pensions "],
["and",5,"CCONJ","cc",4,"and "],
["investments",6,"NOUN","conj",4,"investments "],
["-",7,"PUNCT","punct",2,"- "],
["Which",8,"DET","ROOT",8,"Which"],
["?",9,"PUNCT","punct",8,"? "],
["News",10,"NOUN","ROOT",10,"News"]]
'''

class Titles:

    def __init__(self):
        """
        Constructor for Dataset.

        Does several things:
        Reads from the csv file and stores it as corpus.
        Select a random title for analysis.
        Convert the title from a long string to readable object.
        Preprocess the dataset after converting,
        so that we can remove the entire object we don't want.
        """

        self.corpus = pd.read_csv('mscarticles.csv')
        self.list_of_svo_titles = []

        self.random_title = self.corpus.parsed_title.iloc[rand.randrange(0, 9999)]
        self.random_title = self.convert_from_string_to_objects(self.random_title)
    

        self.corpus.drop_duplicates(subset="title", keep="first", inplace=True)
        for index, row in self.corpus.iterrows():
            self.corpus.loc[index, 'parsed_title'] = self.convert_from_string_to_objects(self.corpus.loc[index, 'parsed_title'])
            if self.is_less_than_two_words(self.corpus.loc[index, 'parsed_title']) == True:
                self.corpus.drop(index, inplace=True)

    def get_dataset(self):
        """
        Return the dataset.
        """

        return self.corpus
    
    def get_title_count(self):
        """
        Return the length of the dataset.
        """

        return len(self.corpus)

    def get_tokenized_random_title(self):
        """
        Return the random title in token form.
        """

        return self.random_title
    
    def get_literal_random_title(self):
        """
        Return the random title in literal form.
        """

        title = []
        for token in self.random_title:
            title.append(token[0])
        literal_title = self.convert_to_string(title)
        return literal_title

    def convert_from_string_to_objects(self, data):
        """
        Convert from string to Python objects.

        Because the data in the corpus is a long string
        which needs to be represented as objects, ast's 
        literal_eval is used to convert it. This also
        essentially acts as our tokenizer.
        """
        
        return literal_eval(data)

    def convert_to_string(self, tokens):
        """
        Convert from a list to a readable string.
        """

        for token in tokens: converted = " ".join(str(token) for token in tokens)
        return converted

    def is_less_than_two_words(self, data):
        """
        Remove entries which are less than two in length.
        """

        if len(data) <= 2: return True
        else: return False

    def extract_subject_verb_object(self):
        """
        Extract subject verb object relationships.

        This takes a very long time, and it most likely
        due to the fact that we run each word through a parser.
        In the future this could be redone, but this is the best
        method for finding SVO relationships I've found as of yet.
        """
        
        nlp = en_core_web_sm.load()
        patterns = []
        for index, row in self.corpus.iterrows():
            indexed_title = self.corpus.loc[index, 'parsed_title']

            words = []
            title = []
            for token in indexed_title: words.append(token[0])
            title = self.convert_to_string(words)

            if type(title) is str: parse = nlp(title)
            elif type(title) is list: None # Just... do nothing...
            svo = findSVOs(parse)
            if svo:
                patterns.append(svo)
                self.list_of_svo_titles.append(indexed_title)
        return patterns

    def get_list_of_tokenized_svos(self):
        """
        Return the qualified SVO titles.

        Make sure extract_svo has been run, 
        otherwise list is empty.
        """

        if len(self.list_of_svo_titles) == 0: self.extract_subject_verb_object()
        return self.list_of_svo_titles

    def total_subject_verb_object(self, list_of_svo):
        """
        Return the total number of subject verb object relationships.
        """

        return len(list_of_svo)

    def most_frequent_subject_verb_object(self, list_of_svo, max):
        """
        Find the most frequent subject verb object relationships.

        Note: using collections in this scenario requires the list to be
        mapped to a tuple.
        """

        counter = collections.Counter(map(tuple, list_of_svo))

        return counter.most_common(max)
    
    def create_digraph(self, data):
        """
        """

        G = nx.DiGraph()
        labels = {}
        plt.figure(figsize=(10, 10))
        for index, token in enumerate(data):
            G.add_node(token[4])
            G.add_edge(token[4], token[1])
            labels[index]=token[0] + " " + token[3]

        pos = graphviz_layout(G, prog='dot')
        nx.draw(G, pos=pos, with_labels=False, font_weight='bold')
        nx.draw_networkx_labels(G, pos, labels)
        plt.savefig("graphs/syntactic_parse_tree_en.png")
        plt.clf()
        #plt.show()
    
    def create_digraph_from_untagged_data_backtranslate(self, data):
        """
        Same as create_digraph, but with untagged data.
        """

        nlp = spacy.load("en_core_web_sm")
        parse = nlp(data)
        G_BACKTRANSLATED = nx.DiGraph()
        labels = {}
        
        for index, word in enumerate(parse):
            G_BACKTRANSLATED.add_node(word.i)
            G_BACKTRANSLATED.add_edge(word.head.i, word.i)
            labels[index]=str(word.text) + " " + str(word.dep_)
        plt.figure(figsize=(10, 10))
        pos_backtranslated = graphviz_layout(G_BACKTRANSLATED, prog='dot')
        nx.draw(G_BACKTRANSLATED, pos=pos_backtranslated, with_labels=False, font_weight='bold')
        nx.draw_networkx_labels(G_BACKTRANSLATED, pos_backtranslated, labels)
        plt.savefig("graphs/syntactic_parse_tree_backtranslated.png")
        plt.clf()
        #plt.show()

    def create_digraph_from_untagged_translated_data(self, data, language):
        """
        Same as create_digraph, but with untagged, translate data.

        Takes the language code as a second parameter, for example:
        french = fr, germany = de
        Refer to https://spacy.io/usage/models for a full list
        """

        # Wrap this in a try, else do the other language model
        load_language = language + "_core_news_sm"
        nlp = spacy.load(load_language)
        parse = nlp(data)
        G_FRENCH = nx.DiGraph()
        labels = {}
        plt.figure(figsize=(10, 10))
        for index, word in enumerate(parse):
            G_FRENCH.add_node(word.i)
            G_FRENCH.add_edge(word.head.i, word.i)
            labels[index]=str(word.text) + " " + str(word.dep_)
        
        pos_french = graphviz_layout(G_FRENCH, prog='dot')
        nx.draw(G_FRENCH, pos=pos_french, with_labels=False, font_weight='bold')
        nx.draw_networkx_labels(G_FRENCH, pos_french, labels)
        plt.savefig("graphs/syntactic_parse_tree_fr.png")
        plt.clf()
        #plt.show()
        
    def draw_dependency_graph(self):
        """
        Draw a dependency graph, save as svg file in cwd.
        """

        nlp = en_core_web_sm.load()
        doc = nlp(self.get_literal_random_title())
        
        dependency_graph = displacy.render(doc, style="dep", jupyter=False)

        file_name = "graphs/dependency_graph.svg"
        output_path = Path.cwd() / file_name
        output_path.open("w", encoding="utf-8").write(dependency_graph)
    
    def draw_entity_recogniser(self):
        """
        Draw an entity recogniser, save as html file in cwd.
        """

        nlp = en_core_web_sm.load()
        doc = nlp(self.get_literal_random_title())
        
        entity_recogniser = displacy.render(doc, style="ent", page=True)

        file_name = "graphs/entity_recogniser.html"
        output_path = Path.cwd() / file_name
        output_path.open("w", encoding="utf-8").write(entity_recogniser)

if __name__ == "__main__":
    Articles = Titles()
    tokenized_random_title = Articles.get_tokenized_random_title()
    literal_random_title = Articles.get_literal_random_title()

    """
    print("Tokenized: \n", tokenized_random_title)
    print("Literal: \n", literal_random_title)
    """

    """   
    list_of_svo = Articles.extract_subject_verb_object()
    total_subject_verb_object = Articles.total_subject_verb_object(list_of_svo)
    average_svo_score = round(total_subject_verb_object/Articles.get_title_count(), 2)
    list_of_tokenized_svos = Articles.get_list_of_tokenized_svos()
    most_frequent_svo = Articles.most_frequent_subject_verb_object(list_of_svo, 5)
    print("Average score: \n", average_svo_score)
    print(most_frequent_svo)
    """

    """
    Articles.draw_dependency_graph()
    Articles.draw_entity_recogniser()
    #Articles.find_shortest_path_between_subject_and_object(list_of_svos)
    """

    # Remove duplicate titles
    # namedentities