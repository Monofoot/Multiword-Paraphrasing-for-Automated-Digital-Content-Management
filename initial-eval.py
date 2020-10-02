import pandas as pd
import random as rand
import string
import networkx as nx # Graph library
import spacy,en_core_web_sm
import matplotlib as mpl
import matplotlib.pyplot as plt
from ast import literal_eval # Interprets strings as Python objects.

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
["–",7,"PUNCT","punct",2,"– "],
["Which",8,"DET","ROOT",8,"Which"],
["?",9,"PUNCT","punct",8,"? "],
["News",10,"NOUN","ROOT",10,"News"]]
'''

class Dataset:

    corpus = None
    random_title = None

    def __init__(self):
        self.corpus = pd.read_csv('mscarticles.csv')
        self.random_title = self.corpus.parsed_title.iloc[rand.randrange(0, 9999)]

    def get_dataset(self):
        return self.corpus
    
    def get_title_count(self):
        return len(self.corpus)

    
    def get_tokenized_random_title(self):
        return self.random_title
    
    def get_literal_random_title(self):
        title_word = []
        for token in literal_eval(self.random_title):
            title_word.append(token[0])
        literal_title = " ".join(str(word) for word in title_word)
        return literal_title
    
    def total_subject_verb_object(self):
        SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
        OBJECTS = ["dobj", "dative", "attr", "oprd"]

        total_svo_count = 0

        for index, row in self.corpus.iterrows():
            indexed_title = self.corpus.parsed_title.iloc[index]
            indexed_title = literal_eval(indexed_title)
            
            subjects = []
            objects = []
            verbs = []

            for token in indexed_title:
                if token[3] in SUBJECTS:
                    subjects.append(token)
                if token[3] in OBJECTS:
                    objects.append(token)
                if token[2] == "VERB":
                    verbs.append(token)
            
            if len(subjects) > 0 and len(objects) > 0 and len(verbs) > 0:
                total_svo_count += 1
        return total_svo_count

    def draw_syntactic_parse_tree(self):
        



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

import nltk
from nltk.tag import pos_tag, map_tag

print("Before stripping it down, the title was: {}".format(random_title))
# Positional tagging
####################
# For each word in the sentence,
# if POS = PROPN, POS = NNP
# Might need a fancy regex for this to make it wildly simpler

print("Testing untokenized: {}".format(eligible_title))
tokens = nltk.word_tokenize(eligible_title)
print("Tokenized: {}".format(tokens))

testag = ()
for token in random_title:
    print("Word: ", token[0], "pos: ", token[2])
    testag.append(token[0], token[3])
print("Testing testtag: ", testag) #JUST FIX THIS IT SUCKS

tagged = nltk.pos_tag(tokens)
tagged = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in tagged]
print("Tagged: {}".format(tagged))

entities = nltk.chunk.ne_chunk(tagged)
print("Entities: {}".format(entities))

# To draw the tree:
entities.draw()
"""

if __name__ == "__main__":
    Articles = Dataset()
    tokenized_random_title = Articles.get_tokenized_random_title()
    literal_random_title = Articles.get_literal_random_title()
    total_subject_verb_object = Articles.total_subject_verb_object()
    average_svo_score = round(Articles.total_subject_verb_object()/Articles.get_title_count(), 2)
