#!/usr/bin/python

import collections
import os
import random as rand
import string
from ast import literal_eval

import en_core_web_sm
import matplotlib as mpl

if os.environ.get('DISPLAY','') == '':
    print('No display for matplotlib found.')
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import pandas as pd
import spacy
from nltk.tag import map_tag, pos_tag
from spacy.lang.en import English

from subject_object_extraction import findSVOs

'''
Each title is stored as:
[ WORD, INDEX, PART OF SPEECH, LABEL OF EDGE BETWEEN THIS NODE AND PARENT, INDEX OF PARENT NODE, PARENT ] 
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

class Dataset:

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

        self.random_title = self.corpus.parsed_title.iloc[rand.randrange(0, 9999)]
        self.random_title = self.convert_from_string_to_objects(self.random_title)

        self.random_title = self.preprocess(self.random_title)
        self.corpus.drop_duplicates(subset="title", keep="first", inplace=True)
        for index, row in self.corpus.iterrows():
            self.corpus.loc[index, 'parsed_title'] = self.convert_from_string_to_objects(self.corpus.loc[index, 'parsed_title'])
            self.corpus.loc[index, 'parsed_title'] = self.preprocess(self.corpus.loc[index, 'parsed_title'])
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

    def convert_from_string_to_objects(self, data):
        """
        Convert from string to Python objects.

        Because the data in the corpus is a long string
        which needs to be represented as objects, ast's 
        literal_eval is used to convert it. This also
        essentially acts as our tokenizer.
        """
        
        return literal_eval(data)
        
        
    
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
    
    def preprocess(self, data):
        """
        Call a series of functions to preprocess data.
        """

        preprocessed_data = self.remove_stopwords(data)

        return preprocessed_data

    def remove_stopwords(self, data):
        """
        Remove stopwords.

        Thankfully these are tagged and easy to find.
        """

        for token in data:
            if token[2] == "PUNCT": data.remove(token)
            if token[0] in string.punctuation and token[2] != "PUNCT": data.remove(token)
        return data

    def is_less_than_two_words(self, data):
        """
        Remove entries which are less than two in length.
        """

        if len(data) <= 2: return True
        else: return False
    
    def convert_to_string(self, tokens):
        """
        Convert from a list to a readable string.
        """
        for token in tokens: converted = " ".join(str(token) for token in tokens)
        return converted


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
            if svo: patterns.append(svo)
        return patterns

    def total_subject_verb_object(self, list_of_svo):
        """
        Return the total number of subject verb object relationships.
        """
        return len(list_of_svo)

    def most_frequent_subject_verb_object(self, list_of_svo):
        """
        Find the most frequent subject verb object relationships.

        Note: using collections is this scenario requires the list to be
        mapped to a tuple.
        """
        counter = collections.Counter(map(tuple, list_of_svo))

        return counter.most_common(3)

    def draw_syntactic_parse_tree(self):
        """
        One line doc string.
        """

        entities = []
        for token in self.random_title:
            entities.append((token[0], token[2]))
        draw_entities = nltk.chunk.ne_chunk(entities)
        # Need to sort the tokens so that it looks like this:
        #[('word', 'NOUN'), ('nextword', 'PRON')]

        # To draw the tree:
        draw_entities.draw()
        

"""
# Check frequent patterns between subject and direct object.
raw_edges = []

for token in random_title:
    raw_edges.append((token[4], token[1])) # Store relationships between edges.

graph_title = nx.DiGraph(raw_edges)

nsubj_nodes = []
dobj_nodes = []

# This might be really inefficient but it's 1am and I'm falling asleep
labels = {}
for node in graph_title.nodes:
    for token in random_title:
        if token[1] == node:
            labels[node] =  str(node) + " " + token[3] + " " + token[0]
            # Also check for nsubj and dobj here.
            if token[3] == "nsubj":
                nsubj_nodes.append(node)
            if token[3] == "dobj":
                dobj_nodes.append(node)

print("Edges: ", graph_title.edges)
print("Nodes: ", graph_title.nodes)


############
# subject object shortest paths
############
#print("The nsubj node(s): ", nsubj_nodes)
#print("The dobj node(s): ", dobj_nodes)
#nx.all_pairs_shortest_path_length(graph_title)
#if len(nsubj_nodes) == len(dobj_nodes) and len(nsubj_nodes) > 0:
#    print("Printing the nodes as they are the same length: ", nsubj_nodes, dobj_nodes)
#    print("The shortest path between each nsubj and dobj: ", nx.shortest_path(graph_title, nsubj_nodes[0], dobj_nodes[0])) # Work from here, can't find a path

# Find node whose label is a subject and then find nx.shortest_path
# So you're simply finding the shortest path from the parent to the child
# "A court has lifted the restriction on Mary Trump's tell-all book" becomes:
# "A court lifted the restriction on Mary Trump's tell-all book"

# So if you get a pattern of like (0, 1) (1, 0) etc, then see how many times you get that pattern
# use nx, don't do it all by hand, research nx - shortest path

pos = nx.planar_layout(graph_title)

nx.draw_networkx_labels(graph_title, pos, labels)
nodes = nx.draw_networkx_nodes(graph_title, pos)
edges = nx.draw_networkx_edges(graph_title, pos)

#plt.show()

##############################

print("\n")

import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

# Use the random title which has been converted to it's raw string sentence.
doc = nlp(eligible_title)
displacy.render(doc, style="dep", jupyter=True)



print("Before stripping it down, the title was: {}".format(random_title))

"""

if __name__ == "__main__":
    Articles = Dataset()
    tokenized_random_title = Articles.get_tokenized_random_title()
    literal_random_title = Articles.get_literal_random_title()
    
    list_of_svo = Articles.extract_subject_verb_object()
    total_subject_verb_object = Articles.total_subject_verb_object(list_of_svo)
    #average_svo_score = round(total_subject_verb_object/Articles.get_title_count(), 2)
    #print("Average: ", average_svo_score)
    most_frequent_svo = Articles.most_frequent_subject_verb_object(list_of_svo)
    print(most_frequent_svo)

    #Articles.draw_syntactic_parse_tree()


