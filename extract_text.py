# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 12:45:22 2018

@author: dhruv
"""
#### imports ##################
import xml.etree.ElementTree as ET
import re
import os
import shutil

#### to remove tags such as <ref> or <b> and all from the text ######
def removeTags(text):
    cleanr = re.compile('<.*?>|\n')
    cleantext = re.sub(cleanr, '', text)
    return cleantext

#### to extract tet from an XML file as a dictionary #################
def XMLtoText(path):
    tree = ET.parse(path)
    root = tree.getroot()
        
    text_tag = root.find("{http://www.tei-c.org/ns/1.0}text")
    body_tag = text_tag.find("{http://www.tei-c.org/ns/1.0}body")
    
    text=""
    for div in body_tag.findall("{http://www.tei-c.org/ns/1.0}div"):
#        head = div.find("{http://www.tei-c.org/ns/1.0}head").text
        for ele in div.findall("{http://www.tei-c.org/ns/1.0}p"):
            xmlstr = ET.tostring(ele, encoding='utf8', method='xml')
            cleantext = removeTags(xmlstr)
            text=text+cleantext
    
    return text

#### fetching all XML files and extracting text from them to store at destination ###
def fetchXMLfiles(source, destination):
    filenames = os.listdir(source)
    for f in filenames:
        pdf_text = XMLtoText(source+f)
        dest = destination#+os.path.splitext(f)[0]
#        if os.path.exists(dest):
#            shutil.rmtree(dest)
#        os.makedirs(dest)
        file_name=dest+os.path.splitext(f)[0]+".txt"
        with open(file_name, 'w') as f:
            f.write('{}\n'.format(pdf_text))

##### path to the folder where all the XML files are stored #####
source = "D:\\Study Material\\M.S\\FALL 18\\Data Analytics\\Project\\XML_Files\\"

#### path where you want the text files to be stored ############
destination = "D:\\Study Material\\M.S\\FALL 18\\Data Analytics\\Project\\text_Files\\"

#### calling function for extraction process ####################
fetchXMLfiles(source, destination)
