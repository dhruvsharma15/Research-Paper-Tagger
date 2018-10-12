# -*- coding: utf-8 -*-



import json
import requests 

with open('arxivData.json', 'r') as f:
    papers = json.load(f)
 
i = 0
papers = papers[0:1000]

for paper in papers:
    i = i + 1
    print("Downloading Paper " + str(i))
    link = paper['link']
    link = link.replace("'", '"')
    link = json.loads(link)
    file_url = link[-1]['href']
    print(file_url)
    r = requests.get(file_url, stream = True) 
    with open("file" + str(i) + ".pdf","wb") as pdf: 
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: 
                pdf.write(chunk) 



