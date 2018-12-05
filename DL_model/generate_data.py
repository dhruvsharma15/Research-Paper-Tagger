# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 17:39:07 2018

@author: dhruv
"""

from glove import  Glove
from nltk.tokenize import word_tokenize
import os
import pickle
import numpy as np

glove_path = '../glove_Arxiv/glove2.model'
data_path = '../../text_Files/'
empty_vector = np.random.rand(300)
doc_len = 500

def tokenize_data(data_path):
    print("tokenizing data")
    papers = os.listdir(data_path)
    data = []
    
    for article in papers:
        article_path = data_path+'/'+article
        with open(article_path,'r') as f:
            data.append(f.read())
        
    tokenized_data = []
    count = 1
    data = data[:2]
    for paper in data:
        tokenized_data.append(word_tokenize(paper))
        if(count%1==0):
            print(count/len(data)*100,"% tokenization done!")
        count+=1
        
    return tokenized_data


def text_to_embeddings(embeddings_path, data_path):
#    tokenized_data = tokenize_data(data_path)
    with open('../glove_Arxiv/tokenized.pkl', 'rb') as f:
        tokenized_data = pickle.load(f)
#    tokenized_data = tokenized_data[:2]
    print("loading glove")
    glove_model = Glove.load(glove_path)
    word_to_int = glove_model.dictionary
    embeddings = glove_model.word_vectors
    print("extracting word vectors")
    file_no = 1
    for i in range(0, len(tokenized_data), 1000):
        batch = tokenized_data[i:i+1000]
        word_vectors = []
        count = 1
        print("vectorizing batch no. ",file_no)
        for paper in batch:
            word_vec = []
            for word in paper:
                try:
                    word_vec.append(embeddings[word_to_int[word]])
                except:
                    print('word:',word)
    #                print('index:', word_to_int[word])
    #                print(embeddings[word_to_int[word]])
            if len(word_vec)<doc_len:
                l = len(word_vec)
                for i in range(doc_len+10 - l):
                    word_vec.extend([empty_vector])
            word_vec = word_vec[0:doc_len]
            word_vectors.append(word_vec)
            if(count%100==0):
                print(count/len(batch)*100,"% vectorization done!")
            count+=1
        
        filename = 'vector_data/data'+str(file_no)+'.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(word_vectors, f)
        file_no+=1
#    return word_vectors
    
text_to_embeddings(glove_path, data_path)

#with open('../../data.pkl', 'wb') as f:
#    pickle.dump(vectorized_data, f)
