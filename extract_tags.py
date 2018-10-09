# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 20:08:21 2018

@author: dhruv
"""

import json
import re

filename='arxivData.json'
data_size = 1000

with open(filename) as f:
    data = json.load(f)

data = data[0:data_size]
tag_vocab = []

for paper in data:
    tags = paper["tag"].replace("'",'"')
    tags = tags.replace('None', '"None"')
    tags_json = json.loads(tags)
    for tag in tags_json:
        if(re.match(r"""[A-Z]""", tag['term'])!=None or 
        re.match(r"""\d+""", tag['term'])!=None):
            continue
        else:
            tag_vocab.append(tag['term'])
    
tag_vocab=set(tag_vocab)
print(tag_vocab)
    