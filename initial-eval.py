import pandas as pd
import random as rand
import string
import networkx as nx # Graph library
import spacy,en_core_web_sm
import matplotlib as mpl
import matplotlib.pyplot as plt
from ast import literal_eval # Interprets strings as Python objects.

print("\n")

df = pd.read_csv('mscarticles.csv')

# Grab a random title.
random_title = df.parsed_title.iloc[rand.randrange(0, 9999)]
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

# By using the ast library we can represent the title as a list of lists.
random_title = literal_eval(random_title) # may not always work, try catch this for escape characters
# Now deconstruct the list into a readable title.
title_word = []
for token in random_title:
    title_word.append(token[0])
eligible_title = " ".join(str(word) for word in title_word)
print(eligible_title)

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

print("Nodes: ", graph_title.edges)
print("Edges: ", graph_title.nodes)


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
SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]

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

total_svo_count = 0

for index, row in df.iterrows():
    indexed_title = df.parsed_title.iloc[index]
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

print("The total SVO count is: ", total_svo_count)
average_svo = total_svo_count/len(df)
print("Average SVO score: ", average_svo, "%")



"""try:
    print("SVO found: ", list_of_subjects[0], list_of_verbs[0], list_of_objects[0])
except:
    print("No SVO pattern found.")
"""
# Find node whose label is a subject and then find nx.shortest_path
# So you're simply finding the shortest path from the parent to the child
# "A court has lifted the restriction on Mary Trump's tell-all book" becomes:
# "A court lifted the restriction on Mary Trump's tell-all book"

# So if you get a pattern of like (0, 1) (1, 0) etc, then see how many times you get that pattern
# use nx, don't do it all by hand, research nx - shortest path

"""pos = nx.planar_layout(graph_title)

nx.draw_networkx_labels(graph_title, pos, labels)
nodes = nx.draw_networkx_nodes(graph_title, pos)
edges = nx.draw_networkx_edges(graph_title, pos)

plt.show()
"""
print("\n")