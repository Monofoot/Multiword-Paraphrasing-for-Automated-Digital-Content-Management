#!/usr/bin/python

import random as rand
import string
from ast import literal_eval
import os

import en_core_web_sm
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('No display for matplotlib found.')
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import pandas as pd
import spacy
from spacy.lang.en import English
from nltk.tag import map_tag, pos_tag
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
        print("Selecting a random title.")
        self.random_title = self.corpus.parsed_title.iloc[rand.randrange(0, 9999)]
        
        print("Converting the random title from string to object.")
        self.random_title = self.convert_from_string_to_objects(self.random_title)
        print("Preprocessing the title.")
        self.random_title = self.preprocess(self.random_title)

        print("Doing the rest of it now")
        for index, row in self.corpus.iterrows():
            self.corpus.parsed_title.iloc[index] = self.convert_from_string_to_objects(self.corpus.parsed_title.iloc[index])
            self.corpus.parsed_title.iloc[index] = self.preprocess(self.corpus.parsed_title.iloc[index])

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
        literal_title = " ".join(str(word) for word in title)
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
    
    # make this a bit for loop to do what test_svo does, 
    # make a new list like string[] or something and store the entries there
    #def convert_to_string(self)

    def extract_subject_verb_object(self):
        """
        Extract subject verb object relationships.
        """
        
        nlp = en_core_web_sm.load()
        for index, row in self.corpus.iterrows():
            indexed_title = self.corpus.parsed_title.iloc[index]

            words = []
            title = []
            for token in indexed_title:
                words.append(token[0])
            for word in words:
                title = " ".join(str(word) for word in words)

            if type(title) is str:
                parse = nlp(title)
            elif type(title) is list:
                None # Just... do nothing...
            if findSVOs(parse):
                total_svo_count += 1
        return total_svo_count

    def total_subject_verb_object(self, list_of_svo):
        """
        Return the total number of subject verb object relationships.
        """
        total_svo_count = 0
                    if type(title) is str:
                parse = nlp(title)
            elif type(title) is list:
                None # Just... do nothing...
            if findSVOs(parse):
                total_svo_count += 1

    def most_frequent_subject_verb_object(self):
        """
        Find the most frequent subject verb object relationships.
        """

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


################
# SVO
################
# Checking for SVO, SVVO or SVOO.

        list_of_subjects = []
        list_of_objects = []
        list_of_verbs = []

'''for token in random_title:
    if token[3] in SUBJECTS:
        list_of_subjects.append(token)
    if token[3] in OBJECTS:
        list_of_objects.append(token)
    if token[2] == "VERB":
        list_of_verbs.append(token)
'''



"""
"""try:
    print("SVO found: ", list_of_subjects[0], list_of_verbs[0], list_of_objects[0])
except:
    print("No SVO pattern found.")
"""
"""
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
    #tokenized_random_title = Articles.get_tokenized_random_title()
    #literal_random_title = Articles.get_literal_random_title()
    #total_subject_verb_object = Articles.total_subject_verb_object()
    #average_svo_score = round(Articles.total_subject_verb_object()/Articles.get_title_count(), 2)
    #Articles.draw_syntactic_parse_tree()
    #print("Average: ", average_svo_score)

