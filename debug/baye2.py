# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 19:28:28 2020

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
from shutil import copyfile
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from src.models.train_model import functionsBuilder
from src.models.audio import downloadAudioFromYoutube
import warnings
warnings.filterwarnings("ignore")



groundbase_dir = '../data/raw/groundbase'
transcripts_dir = os.path.join(groundbase_dir,'transcripts')
topic_dataset_path = os.path.join(groundbase_dir,'dataset.csv')
results_csv = '../data/processed/bayesian_opt/phrases/lemmas_adv_propn.csv'


videos_ids = ['GJpt_3ie4WU']#['zWg7U0OEAoE','7kLHJ-F33GI','7snJ1mx1EMQ','RIawrYLVdIw','tORLeHHtazM']
transcript_filespath = [glob.glob(groundbase_dir + '/transcripts/{}.json'.format(vid))[0]
                                  for vid in videos_ids]


'''Read the transcript'''
transcripts_jsons = {}
for fl in transcript_filespath:
    with open(fl,encoding="utf8") as f:
        transcript =ast.literal_eval(f.read()) #json.load(f)
        vid = fl.split('/')[-1].split('.')[0]
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
    
    
models_workflow = {
    'sliding_window':[
        'sliding_window-tfidf-cosine-None-spectral_clustering',
        'sliding_window-tfidf-cosine-median_(2,2)-spectral_clustering',
        "sliding_window-tfidf-cosine-median_(3,3)-spectral_clustering"
    ]
}
models = list(models_workflow.keys())
'''Read the parameters bounds'''
with open(os.path.join('../models/bayesian_opt/experiment','parameters_bounds.json'),'r') as f:
    param_bounds = ast.literal_eval(f.read())
df_results =pd.read_csv(results_csv) #pd.DataFrame(columns=['video','max_target','workflow','params','param_bounds']) ##

for vid in videos_ids:
    print("running on video %s" %(vid))
    groundbase = df_videos.loc[df_videos['video id'] == vid,'topic shifts(ends)'].values.tolist()[:-1]
    tr = transcripts_jsons[vid]
    for model in models:
        #samples = [d["train"] for d in dataset if d[model] == model][0]
        for workflow_label in models_workflow[model]:
            print('Running workflow %s' %(workflow_label))
            #logger = JSONLogger(path=os.path.join('../data/processed/bayesian_opt/phrases',('%s.json' %(vid +"_"+ workflow_label))))
            #optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            
            function_to_optimized = functionsBuilder(groundbase=groundbase,transcripts=tr).build_f_to_optimize(workflow_label)
            
            optimizer = BayesianOptimization(
                f=function_to_optimized,
                pbounds=param_bounds[workflow_label],
                verbose=2,
                random_state=1
                )
            
            optimizer.maximize(
                init_points = 3,
                n_iter = 6
            )
            
            print(optimizer.max)
            for res in optimizer.res:
                df_results = df_results.append({'video':vid,
                                                'max_target':res['target'],
                                                'workflow':workflow_label,
                                                'params':res['params'],
                                                'param_bounds':param_bounds[workflow_label]
                                               },ignore_index=True)
        

