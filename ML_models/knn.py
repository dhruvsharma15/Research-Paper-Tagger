# coding: utf-8

# In[65]:


import os
import json
import re
path = './dataset/papers'
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

from sklearn.model_selection import train_test_split
import numpy as np
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
# In[66]:


with open("./arxivdataset/arxivData.json", "r") as read_file:
    papers = json.load(read_file)

print("Dataset Loaded")
# In[67]:


invalidTags = []
with open("invalidTags.txt", 'r') as f:
    for line in f:
        invalidTags.append(line.rstrip())
sampledPapers = []
with open("sampledPapers.txt", 'r') as f:
    for line in f:
        sampledPapers.append(line.rstrip())


# In[68]:


sampledFiles = []
with open("sampledPapers.txt", 'r') as f:
    for line in f:
        sampledFiles.append("file" + str(int(line.rstrip()) + 1) + ".txt")

for filename in os.listdir(path):
    if(filename not in sampledFiles):
        os.remove(path + '/' + filename)


# In[69]:


def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext


def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned


def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


# In[70]:



import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
allLabels = set()

for paperIndex in sampledPapers:
    paper = papers[int(paperIndex)]
    tags = paper["tag"].replace("'",'"')
    tags = tags.replace('None', '"None"')
    tags_json = json.loads(tags)
    for it in tags_json:
        if (it['term'] not in invalidTags and re.match(r"""[A-Z]""", it['term']) == None and re.match(r"""\d+""", it['term']) == None):
            allLabels.add(it['term'])

print("Created the set allLabels")
# In[71]:


dataset = []

for paperIndex in sampledPapers:
    currentDict = {}
    labels = []
    paper = papers[int(paperIndex)]
    fileIndex = int(paperIndex) + 1
    try:
        with open(path + '/' + "file" + str(fileIndex) + ".txt", "r+",encoding="utf8") as f:
            content = f.read()
        tags = paper["tag"].replace("'",'"')
        tags = tags.replace('None', '"None"')
        tags_json = json.loads(tags)
        for it in tags_json:
            if (it['term'] not in invalidTags and re.match(r"""[A-Z]""", it['term']) == None and re.match(r"""\d+""", it['term']) == None):
                labels.append(it['term'])
        currentDict["content"] = content
        for l in allLabels:
            if(l in labels):
                currentDict[l] = 1
            else:
                currentDict[l] = 0
        dataset.append(currentDict)
    except:
        pass


# In[72]:



data = pd.DataFrame(dataset)
data['content'] = data['content'].str.lower()
data['content'] = data['content'].apply(cleanHtml)
data['content'] = data['content'].apply(cleanPunc)
data['content'] = data['content'].apply(keepAlpha)
data.head()
data = data[:4000]
print("Data Processed")
# In[73]:


stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

data['content'] = data['content'].apply(removeStopWords)


# In[74]:


stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

data['content'] = data['content'].apply(stemming)
data.head()


# In[75]:


train, test = train_test_split(data, random_state=42, test_size=0.30, shuffle=True)

print(train.shape)
print(test.shape)


# In[76]:


train_text = train['content']
test_text = test['content']


# In[77]:


vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
vectorizer.fit(train_text)
vectorizer.fit(test_text)

print("TfIdf done")
# In[78]:


x_train = vectorizer.transform(train_text)
y_train = train.drop(labels = ['content'], axis=1)

x_test = vectorizer.transform(test_text)
y_test = test.drop(labels = ['content'], axis=1)

from sklearn.linear_model import LogisticRegression
# using Label Powerset
from skmultilearn.problem_transform import LabelPowerset
# initialize label powerset multi-label classifier

# using classifier chains
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
# initialize classifier chains multi-label classifier

from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from datetime import timedelta
import time
start = time.time()

from scipy.sparse import csr_matrix, lil_matrix
from skmultilearn.adapt import MLkNN
x_train = lil_matrix(x_train).toarray()
y_train = lil_matrix(y_train).toarray()
x_test = lil_matrix(x_test).toarray()
classifier = MLkNN(k=4)

# train
classifier.fit(x_train, y_train)

# predict
predictions = classifier.predict(x_test)

# accuracy
print("Accuracy = ",accuracy_score(y_test,predictions))
print("\n")
print("F1 = ",f1_score(y_test,predictions,average='micro'))
print("\n")

print("Jaccard = ",jaccard_similarity_score(y_test,predictions))
print("\n")

print("Precision = ",precision_score(y_test, predictions,average='micro'))
print("\n")

print("Recall = ",recall_score(y_test,predictions,average='micro'))
print("\n")

print(timedelta(seconds=time.time()-start))
