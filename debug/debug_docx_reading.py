# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:28:58 2020

@author: yaniv
"""

import sys
sys.path.append('..')

from src.data.docx  import read_docx,process_docx,find_content
import os 
import glob
from statistics import mode
import re

'''Defining CONSTS'''
docx_path = '../data/raw/docx'
groundbase_dir = '../data/raw/groundbase'
transcripts_dir = os.path.join(groundbase_dir,'transcripts')
topic_dataset_path = os.path.join(groundbase_dir,'dataset.csv')
transcript_filespath = glob.glob(groundbase_dir + '/transcripts/*.json')

videos_ids = list(map(lambda fl: fl.split('\\')[-1].split('.')[0],glob.glob(docx_path + '/*')))

desired_videos = ['zWg7U0OEAoE']#['7kLHJ-F33GI','RIawrYLVdIw','7snJ1mx1EMQ'] #
videos_ids = list(filter(lambda x: x in desired_videos,videos_ids))


for vid in videos_ids:
    doc_path = glob.glob(os.path.join(docx_path,vid + '/*.docx'))[0]
    doc_name = doc_path.split('\\')[-1].split('.')[0]
    full_text,font_sizes = read_docx(doc_path)
    #vika = find_content('statbook',full_text,font_sizes,main_chapter_keyword='Topic')
    vika = find_content('Dsa',full_text,font_sizes,main_chapter_keyword='Chapter')