# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 21:03:06 2020

@author: yaniv
"""

import sys
sys.path.append('..')
from src.models.pipeline import pipeline
import glob
import json
import os
import pandas as pd
import ast
from datetime import datetime
from src.models.train_model import functionsBuilder
from src.models.audio import downloadAudioFromYoutube
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data import pdf
from functools import reduce
import heapq
from gensim.corpora import Dictionary
from gensim import models
from src.visualization.visualize import MyPlotting
from gensim import similarities
import re
import statistics
import numpy as np





groundbase_dir = '../data/raw/groundbase'
transcripts_dir = os.path.join(groundbase_dir,'transcripts')
topic_dataset_path = os.path.join(groundbase_dir,'dataset.csv')
transcript_filespath = glob.glob(groundbase_dir + '/transcripts/*.json')

'''Read the transcript'''
transcripts_jsons = {}
for fl in transcript_filespath:
    with open(fl,encoding="utf8") as f:
        transcript =ast.literal_eval(f.read()) #json.load(f)
        vid = fl.split('\\')[-1].split('.')[0]
        #print(vid)
        transcripts_jsons[vid] = transcript
#print(transcripts_jsons)

'''Read the videos metadata to perform on them the segmentation'''
df_videos = pd.read_csv(topic_dataset_path)

''' Transfer topic shifts time to seconds units instead HH:MM:SS'''

def topic_shifts_seconds(topic_shifts):
    tp_shift_sec=[]
    for tp in topic_shifts:
        intervals = tp.split(':')
        seconds = int(intervals[2])
        minutes = int(intervals[1]) * 60
        hours = int(intervals[0]) * 60 *60
        tp_shift_sec.append(seconds + minutes + hours)
    return tp_shift_sec


for video_id in transcripts_jsons.keys():    
    df_videos.at[df_videos['video id'] == video_id,'topic shifts(ends)'] =\
    topic_shifts_seconds(\
                         df_videos[df_videos['video id'] == \
                                   video_id]['topic shifts(ends)'])
    
    
    
df_results = pd.read_csv('../data/processed/bayesian_opt/results.csv')
#best_results = df_results.groupby('video')[['video','workflow','params','max_target']].max().values.tolist()
n_largest_res = 1
'''This is the videos that have a pdf'''
#filtered_video = ['x5zLaWT5KPs','dkAr9ThdSUU','2mC1uqwEmWQ',
#                  'MkiUBJcgdUY','Q-HugPvA7GQ','tORLeHHtazM','zWg7U0OEAoE']

filtered_video =['7kLHJ-F33GI'] #['tORLeHHtazM','zWg7U0OEAoE',,'7snJ1mx1EMQ','RIawrYLVdIw']

best_results = df_results[df_results['video'].isin(filtered_video)].groupby('video')[['video','workflow','params','max_target']].apply(lambda grp: grp.nlargest(n_largest_res,'max_target')).values.tolist()

filtered_video = [bs[0] for bs in  best_results]
best_results



vid_words= []
vids_shift_times = []
vids_id = []
#best_results = best_results[1:]
for vid_results in best_results: #range(0,len(best_results),n_largest_res):
    '''From the get optimized by bayesian we get that for the video '''
    # the precision is about 66% 
    vid = vid_results[0]
    params = ast.literal_eval(vid_results[2]) #{'n_clusters': 18, 'sim_thresh': 0.6, 'step_size': 49, 'window_size': 150}
    #print(params)
    for key in ['n_clusters','step_size','window_size']:
        params[key] = int(params[key])# - 1
    workflow = vid_results[1] #'sliding_window-tfidf-cosine-median_(3,3)-spectral_clustering'

    groundbase = df_videos.loc[df_videos['video id'] == vid,'topic shifts(ends)'].values.tolist()[:-1]
    transcripts = transcripts_jsons[vid]
    #print(grounbase)
    _pipeline = workflow.split('-')
    filter_type = None
    mask_shape = None
    filtering = _pipeline[3]
    if filtering != 'None':
        filter_type = filtering.split('_')[0]
        mask_shape = ast.literal_eval(filtering.split('_')[1])
    '''This running may not work at first time do not give up and run it couple of times'''

    '''print('Running the following %s for video %s with params %s %s %s'
          %(workflow, vid,params,filter_type,mask_shape))'''
    shift_times,topic_words = (None,None)
    while shift_times is None and topic_words is None:
        shift_times,topic_words = pipeline.run_for_baye(groundbase,transcripts,slicing_method='sliding_window',
                              window_size=params['window_size'],step_size_sd=params['step_size'],
                              #silence_threshold=-30,slice_length=1000,step_size_audio=10,wav_file_path="../../data/raw/audio/Mod-01 Lec-01 Foundation of Scientific Computing-01.wav",                
                              vector_method='tfidf',vectorizing_params=None,
                              similarity_method='cosine',
                              accurrcy_shift=30,
                              filter_params={"filter_type":filter_type,
                                             "mask_shape":mask_shape,
                                             "sim_thresh":params['sim_thresh'],
                                             "is_min_thresh":True
                                             },
                             clustering_params={
                                     'algorithm':'spectral_clustering',
                                     'n_clusters':params['n_clusters']
                                     },return_value='division') or (None,None)
    print("For video %s, %s where found" %(vid,len(shift_times)))
    vid_words.append(topic_words)
    shift_times.append('end')
    vids_shift_times.append(shift_times)
    vids_id.append(vid)