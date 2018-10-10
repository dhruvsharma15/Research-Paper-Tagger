# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 20:08:21 2018

@author: dhruv
"""

import json
import re
import matplotlib.pylab as plt

valid_files = 'accesibleFiles.txt'
ind = []
labels = []

filename='../arxivData.json'
data_size = 25000
count = 0

#with open(valid_files) as f:
#    for line in f:
#        ind.append(int(line)-1)
#
#ind.sort()
#ind = set(ind)

for i in range(data_size):
    ind.append(i)

with open(filename) as f:
    data = json.load(f)

data = data[0:data_size]
all_tags = []

for paper in data:
    if count in ind:
        tags = paper["tag"].replace("'",'"')
        tags = tags.replace('None', '"None"')
        tags_json = json.loads(tags)
        for tag in tags_json:
            if(re.match(r"""[A-Z]""", tag['term'])!=None or re.match(r"""\d+""", tag['term'])!=None):
                continue
            else:
                all_tags.append(tag['term'])
    count = count + 1
    
tag_vocab=list(set(all_tags))
tag_vocab.sort()
tag_ind = dict()
tag_count = dict()

for tag in all_tags:
    if tag in tag_count:
        tag_count[tag]+=1
    else:
        tag_count[tag]=1

lists = sorted(tag_count.items())
x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)
plt.show()

##################### need to do stratified subsampling #######################

#count = 0
#for ele in tag_vocab:
#    tag_ind[ele] = count
#    count=count + 1
#
#count = 0 
#for paper in data:
#    if count in ind:
#        tags = paper["tag"].replace("'",'"')
#        tags = tags.replace('None', '"None"')
#        tags_json = json.loads(tags)
#        label = [0]*len(ind)
#        for tag in tags_json:
#            if(re.match(r"""[A-Z]""", tag['term'])!=None or re.match(r"""\d+""", tag['term'])!=None):               
#                continue
#            else:
#                label[tag_ind[tag['term']]] = 1
#        labels.append(label)
#    count = count + 1        