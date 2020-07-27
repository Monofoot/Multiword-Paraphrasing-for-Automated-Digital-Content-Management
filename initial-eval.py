import pandas as pd
import random as rand
import networkx as nx # Graph library
from ast import literal_eval # Interprets stirngs as Python objects.
df = pd.read_csv('mscarticles.csv')

# Print a title.
random_title = df.parsed_title.iloc[rand.randrange(0, 9999)]
'''
Each title is stored as:
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

[ WORD, INDEX, PART OF SPEECH, LABEL OF EDGE BETWEEN THIS NODE AND PARENT, INDEX OF PARENT NODE, PARENT ] 
'''

# By using the ast library we can represent the title as a list of lists.
random_title = literal_eval(random_title) # may not always work, try catch this for escape characters
# Now deconstruct the list into a readable title.
title_word = []
for token in random_title:
    title_word.append(token[0])
eligible_title = " ".join(str(word) for word in title_word)
print(eligible_title)



# Store all of the sentence types in one array - not sure how useful this will be, but let's try it
"""
sentence_root = []
sentence_dobj = []
sentence_pobj = []
sentence_nsubj = []
sentence_conj = []
sentence_prep = []
sentence_punct = []
sentence_cc = []

for token in random_title:
    print("Testing: ", token[3])
    if token[3] == "ROOT":
        sentence_root.append(token)
    if token[3] == "dobj":
        sentence_dobj.append(token)
    if token[3] == "pobj":
        sentence_pobj.append(token)
    if token[3] == "nsubj":
        sentence_nsubj.append(token)
    if token[3] == "conj":
        sentence_conj.append(token)
    if token[3] == "prep":
        sentence_prep.append(token)
    if token[3] == "punct":
        sentence_punct.append(token)
    if token[3] == "cc":
        sentence_cc.append(token)

print("The root is: ", sentence_root)
print("The dobj is: ", sentence_dobj)
print("The pobj is: ", sentence_pobj)
"""

# Check frequent patterns between subject and direct object.
edges = []

for token in random_title:
    edges.append((token[4], token[1])) # Store relationships between edges.

graph_title = nx.DiGraph(edges)

print(graph_title.edges)
print(graph_title.nodes)

title_subjects = []
for token in random_title:
    if token[3] == "nsubj":
        title_subjects.append(token)

print("The subjects: ", title_subjects)


# Find node whose label is a subject and then find nx.shortest_path
# So you're simply finding the shortest path from the parent to the child
# "A court has lifted the restriction on Mary Trump's tell-all book" becomes:
# "A court lifted the restriction on Mary Trump's tell-all book"

# So if you get a pattern of like (0, 1) (1, 0) etc, then see how many times you get that pattern

# FIND THE SVO PATTERNS "SUBJECT, VERB, OBJECT"
# num of SVO patterns / len(df), the higher the more linguistically standard the newspaper headlines are
# use nx, don't do it all by hand, research nx - shortest path
