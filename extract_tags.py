# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 20:08:21 2018

@author: dhruv
"""

import json
import re
import matplotlib.pylab as plt
import numpy as np

filename='../arxivData.json'
data_size = 25000
threshold = 0.4

def generate_tags(filename, datasize):
    labels = []
    with open(filename) as f:
        data = json.load(f)
    
    data = data[0:data_size]
    all_tags = []
    
    for paper in data:
        tags = paper["tag"].replace("'",'"')
        tags = tags.replace('None', '"None"')
        tags_json = json.loads(tags)
        for tag in tags_json:
            if(re.match(r"""[A-Z]""", tag['term'])!=None or re.match(r"""\d+""", tag['term'])!=None):
                continue
            else:
                all_tags.append(tag['term'])
        
    tag_vocab=list(set(all_tags))
    tag_vocab.sort()
    tag_ind = dict()

    
    count = 0
    for ele in tag_vocab:
        tag_ind[ele] = count
        count=count + 1
    
    for paper in data:
        tags = paper["tag"].replace("'",'"')
        tags = tags.replace('None', '"None"')
        tags_json = json.loads(tags)
        label = [0]*len(tag_vocab)
        for tag in tags_json:
            if(re.match(r"""[A-Z]""", tag['term'])!=None or re.match(r"""\d+""", tag['term'])!=None):               
                continue
            else:
                label[tag_ind[tag['term']]] = 1
        labels.append(label)
    return labels, all_tags, tag_ind

def tag_distribution(all_tags):
    tag_count = dict()
    for tag in all_tags:
        if tag in tag_count:
            tag_count[tag]+=1
        else:
            tag_count[tag]=1
    
    lists = sorted(tag_count.items())
    tag_label, y = zip(*lists) # unpack a list of pairs into two tuples
    x=[]
    
    for i in range(len(y)):
        x.append(i+1)
    
    #plt.plot(x, y)
    #plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    line, = ax.plot(x, y, lw=2)
    
    for i in range(len(y)):
        if(y[i]>4000):
            ax.annotate(tag_label[i], xy=(x[i], y[i]), xytext=(x[i], y[i]+1000),
                arrowprops=dict(facecolor='black', shrink=0.05),
                )
    
    plt.show()
    return lists
    
def count_more_than_threshold(threshold, tag_count):
    tag_label, y = zip(*tag_count)
    count_threshold = data_size*threshold/100
    valid_tags = 0
    for c in y:
        if(c > count_threshold):
            valid_tags+=1
    return valid_tags
    
def find_threshold(tag_count):
    threshold = 0.1
    th_count = []
    while(threshold <= 1):
        this_th = []
        valid_tags = count_more_than_threshold(threshold, tag_count)
        this_th.append(threshold)
        this_th.append(valid_tags)
        th_count.append(this_th)
        threshold+=0.1
    return th_count
    
def invalid_tags(tag_count, threshold):
    count_threshold = data_size*threshold/100
    inval_tags = []
    inval_tag_ind = []
    count = 0
    for tag_label, y in tag_count:
        if y < count_threshold:
            inval_tags.append(tag_label)
            inval_tag_ind.append(count)
        count+=1
    return inval_tags, inval_tag_ind
        
def valid_papers(labels, inval_tags, tag_ind):
    subsampled_labels = []
    subsampled_papers_ind = []
    count = 0
    for label in labels:
        isVal = True
        for tag in inval_tags:
            if label[tag_ind[tag]]==1:
                isVal = False
                break
        if(isVal):
            subsampled_labels.append(label)
            subsampled_papers_ind.append(count)
        count+=1
    return subsampled_labels, subsampled_papers_ind

def clean_labels(subsampled_labels, inval_tag_ind):
    subsampled_labels = np.array(subsampled_labels)
    count = 0
    for ind in inval_tag_ind:
        subsampled_labels = np.delete(subsampled_labels, ind-count, axis = 1)
        count+=1
    return subsampled_labels

def subsampled_data_distribution(subsampled_labels):
    y = np.sum(subsampled_labels, axis=0)
    x = []
    
    for i in range(len(subsampled_labels[0])):
        x.append(i+1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    line, = ax.plot(x, y, lw=2)
    
    for i in range(len(x)):
        if(y[i]>4000):
            ax.annotate(x[i], xy=(x[i], y[i]), xytext=(x[i], y[i]+1000),
                arrowprops=dict(facecolor='black', shrink=0.05),
                )
    
    plt.show()          
        
labels, all_tags, tag_ind = generate_tags(filename, data_size)
tag_count = tag_distribution(all_tags)
th_count = find_threshold(tag_count)

inval_tags_list, inval_tag_ind = invalid_tags(tag_count, threshold)

subsampled_labels, subsampled_papers_ind = valid_papers(labels, inval_tags_list, tag_ind)
subsampled_labels = clean_labels(subsampled_labels, inval_tag_ind)

subsampled_data_distribution(subsampled_labels)








