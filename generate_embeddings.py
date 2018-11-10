# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:43:50 2018

@author: dhruv
"""

from glove import Corpus, Glove
import os
from collections import Counter
from nltk.tokenize import word_tokenize

def read_files(data_path):
    print('print papers')
    papers = os.listdir(data_path)
    data = []
    
    for article in papers:
        article_path = data_path+'/'+article
        with open(article_path,'r') as f:
            data.append(f.read())
            
    return data

def get_embeddings(papers, embed_size, window_size, n_epochs):
    print('generating embeddings')
    tokenized_data = []
    for paper in papers:
        tokenized_data.append(word_tokenize(paper))
    
    corpus = Corpus()
    corpus.fit(tokenized_data, window=window_size)
    glove = Glove(no_components=embed_size, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=n_epochs, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save('glove.model')
    
    return glove.word_vectors, corpus.dictionary, tokenized_data

def vocab_count(tokenized_data):
    print('generating vocab')
    words = []
    for data_point in tokenized_data:
        for word in data_point:
            words.append(word)
        
    words_count = Counter(words)
    
    return words_count
    
def write_vocab_to_file(filename, count_words):
    with open(filename, 'w') as f:
        for key, val in count_words.items():
            f.write(str(key)+" "+str(val)+"\n")

embedding_size = 1
window_size = 1
epochs = 1
data_path = 'text_Files/'
papers = read_files(data_path)
glove_vectors, dictionary, tokenized_data = get_embeddings(papers, embedding_size, window_size, epochs)
count_words = vocab_count(tokenized_data)
#write_vocab_to_file("vocab.txt",count_words)