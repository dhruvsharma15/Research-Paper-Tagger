import gensim
import os
import pprint
import json
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
pd.options.mode.chained_assignment = None 
import re
from sklearn.manifold import TSNE
from collections import Counter


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels_all = []
    tokens = []

    for word in model.wv.vocab:
        labels_all.append(word)
        #print word
    
    c = Counter(labels_all)
    labels = c.most_common(len(labels_all))
    print(labels)


    for w in labels:
        tokens.append(model[w[0]])

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i][0],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


stopWords = set(stopwords.words('english'))

path="txt_files/"

files=os.listdir(path)

doc = []
i=0
for f in files:
	i=i+1
	with open(path+f,'rb') as file:
		for line in file:
			words = line.split()
			resultwords  = [word for word in words if word.lower() not in stopWords and word.lower() not in string.punctuation]
			result = ' '.join(resultwords)
			print(result)
			doc.append(gensim.utils.simple_preprocess (result))
	file.close()
	print("\n\n DOC ENDING "+str(i))

document = list(doc)

pp = pprint.PrettyPrinter()

#pp.pprint(document)

print("\nBegin Training")

model = gensim.models.Word2Vec(document, size=150, window=10, min_count=2, workers=10)
print("\nTraining Start")

model.train(document,total_examples=len(document),epochs=10)
print("\nTraining End")
model.wv.save("trained_model.kv")
model.save("model.bin")

tsne_plot(model)