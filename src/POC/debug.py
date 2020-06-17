# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:06:01 2020

@author: yaniv
"""

'''import sys
sys.path.append('../../')

import json
from src.models.pipeline import pipeline
import ast
import os
from pytube import YouTube
import pandas as pd      
groundbase = [22,87,207,270,319,370,450,558,602,671,756,930,978,1011]

video_id = '8BeXwhljq2g' #2mC1uqwEmWQ'
with open('../../data/raw/groundbase/transcripts/' + video_id +'.json',encoding="utf8") as f:
    transcripts = ast.literal_eval(f.read())


df = pd.DataFrame() #pd.read_csv('../../reports/csv/TextTiling_Freq_Vectors.csv')

video_len = 1059'''








#print("yaniv")

#p = pipeline(df,groundbase,video_id,video_len,transcripts,vector_method='tfidf',\
#window_size=40,step_size=20,similarity_method='cosine',is_min_thresh=True)
#df = p.run(algorithm='spectral_clustering',n_clusters=13,sim_filter=['median',(2,2)],sim_thresh=0.4)


# working
#pipeline.run(df,groundbase,video_id,video_len,transcripts,vector_method='tfidf',\
#window_size=40,step_size=20,similarity_method='cosine',is_min_thresh=True,\
#algorithm='spectral_clustering',n_clusters=13,sim_filter=['median',(2,2)],sim_thresh=0.4)


#pipeline.run(df,groundbase,video_id,video_len,transcripts,vector_method='tfidf',\
#window_size=40,step_size=20,similarity_method='cosine',is_min_thresh=True,\
#algorithm='spectral_clustering',n_clusters=13,sim_filter=['median',(2,2)],sim_thresh=0.4)

#pipeline.run(df,groundbase,video_id,video_len,transcripts,vector_method='do_nothing',\
#window_size=40,step_size=20,similarity_method='wmdistance',is_min_thresh=True,\
#algorithm='spectral_clustering',n_clusters=13,sim_filter=None,sim_thresh=0.4)

#pipeline.run(df,groundbase,video_id,video_len,transcripts,vector_method='word2vec_naive_mean',\
#window_size=40,step_size=20,similarity_method='cosine',is_min_thresh=True,\
#algorithm='spectral_clustering',n_clusters=13,sim_thresh=0.4)


'''pipeline.run(df,groundbase,video_id,video_len,transcripts,vector_method='tfidf',\
window_size=40,step_size=20,similarity_method='cosine',is_min_thresh=True,sim_filter=['gaussian_laplace',1.4],\
algorithm='dbscan',clustering_params={"eps":0.1,"min_samples":4},n_clusters=13,sim_thresh=None)'''

'''pipeline.run(df,groundbase,video_id,video_len,transcripts,vector_method='lda',\
window_size=40,step_size=20,similarity_method='jensen_shannon',is_min_thresh=True,\
algorithm='spectral_clustering',n_clusters=13,sim_filter=['median',(2,2)],sim_thresh=0.4,\


'''
#from src.visualization.visualize import MyPlotting
#import numpy as np

#MyPlotting.similarity_matrix(np.ones((50,50)))
'''

print(pipeline.run_for_baye(groundbase,transcripts,window_size=60,step_size=20,
                      vector_method='tfidf',vectorizing_params=None,similarity_method='cosine',
                      filter_params={"filter_type":'median',
                                     "mask_shape":(2,2),
                                     "sim_thresh":0.4,
                                     "is_min_thresh":True
                                     },
                     clustering_params={
                             'algorithm':'spectral_clustering',
                             'n_clusters':13
                             }))

'''




'''from src.features.segment_transcript import CreateBlocks

block_handler =  CreateBlocks(transcripts)
timestamp_array = [25,400,1020]
bla = block_handler.partion_by_timestamp(timestamp_array)
print(bla)
'''

'''from src.models import audio
bla = audio.getSubjSilentRanges('Mod-01 Lec-01 Foundation of Scientific Computing-01.wav',0,1,1)
print(bla)'''


'''pipeline.run(df,groundbase,video_id,video_len,transcripts,vector_method='doc2vec',\
window_size=40,step_size=20,similarity_method='kendall_tau',is_min_thresh=True,\
clustering_params={"algorithm":'spectral_clustering',"n_clusters":13},sim_filter=None,sim_thresh=0.4)
'''

#"C:\Users\yaniv\Desktop\playgrounds2\DeepLearning\Pytorch\POC FINAL PROJECT\project_name\data\raw\audio\Mod-01 Lec-01 Foundation of Scientific Computing-01.wav"
'''pipeline.run_for_baye(groundbase,transcripts,slicing_method='audio',
                      silence_threshold=-30,slice_length=1000,step_size_audio=10,wav_file_path="../../data/raw/audio/Mod-01 Lec-01 Foundation of Scientific Computing-01.wav",                
                      vector_method='tfidf',vectorizing_params=None,similarity_method='cosine',
                      filter_params={"filter_type":'median',
                                     "mask_shape":(2,2),
                                     "sim_thresh":0.4,
                                     "is_min_thresh":True
                                     },
                     clustering_params={
                             'algorithm':'spectral_clustering',
                             'n_clusters':13
                             })'''


#from src.models import audio
#audio.downloadAudioFromYoutube('https://www.youtube.com/watch?v=' + video_id)

'''pipeline.run(df,groundbase,video_id,video_len,transcripts,vector_method='tfidf',
             figure_path='../../data/processed/mannul test',
             #slicing_method='sliding_window',
             #window_size=120,step_size_sd=20,
             slicing_method='audio',
             silence_threshold=-43,slice_length=1000,step_size_audio=100,wav_file_path="../../data/raw/audio/" + video_id + ".wav",
             similarity_method='cosine',is_min_thresh=True,
             #vectorizing_params= {'n_clusters':14},
             clustering_params={"algorithm":'spectral_clustering',"n_clusters":13},
             sim_filter=None,sim_thresh=0.4)'''

'''
video_url = "https://www.youtube.com/watch?v=" + video_id
youtube = YouTube(video_url)
os.mkdir('../../data/raw/videos/' + video_id)
youtube.streams.first().download('../../data/raw/videos',filename=video_id)

pipeline.run(df,groundbase,video_id,video_len,transcripts,vector_method='tfidf',
             figure_path='../../data/processed/handtuning/mannul test',
             #slicing_method='sliding_window',
             #window_size=120,step_size_sd=20,
             slicing_method='text_changing',
             #silence_threshold=-43,slice_length=1000,step_size_audio=100,wav_file_path="../../data/raw/audio/" + video_id + ".wav",
              wanted_frequency=30, change_threshold=75,
             similarity_method='cosine',is_min_thresh=True,
             #vectorizing_params= {'n_clusters':14},
             clustering_params={"algorithm":'spectral_clustering',"n_clusters":14},
             sim_filter=None,sim_thresh=0.4)

'''


'''print(pipeline.run_for_baye(groundbase,transcripts,slicing_method='sliding_window',
                      window_size=40,step_size_sd=20,
                      #silence_threshold=-30,slice_length=1000,step_size_audio=10,wav_file_path="../../data/raw/audio/Mod-01 Lec-01 Foundation of Scientific Computing-01.wav",                
                      vector_method='tfidf',vectorizing_params=None,
                      similarity_method='cosine',
                      filter_params={"filter_type":'median',
                                     "mask_shape":(2,2),
                                     "sim_thresh":0.4,
                                     "is_min_thresh":True
                                     },
                     clustering_params={
                             'algorithm':'spectral_clustering',
                             'n_clusters':13
                             },return_value='division'))
pass'''



#params = {'n_clusters': 18, 'sim_thresh': 0.6, 'step_size': 49, 'window_size': 150}#{'n_clusters': 18, 'sim_thresh': 0.9, 'window_size': 60, 'step_size': 20}
#params = {'n_clusters': 18, 'sim_thresh': 0.9, 'step_size': 60, 'window_size': 200}
#workflow = 'sliding_window-tfidf-cosine-median_(3,3)-spectral_clustering'




#groundbase = df_videos.loc[df_videos['video id'] == vid,'topic shifts(ends)'].values.tolist()[:-1]
#transcripts = transcripts_jsons[vid]


pipeline.run_for_baye(groundbase,transcripts,slicing_method='sliding_window',
                      window_size=params['window_size'],step_size_sd=params['step_size'],
                      #silence_threshold=-30,slice_length=1000,step_size_audio=10,wav_file_path="../../data/raw/audio/Mod-01 Lec-01 Foundation of Scientific Computing-01.wav",                
                      vector_method='tfidf',vectorizing_params=None,
                      similarity_method='cosine',
                      filter_params={"filter_type":'median',
                                     "mask_shape":(3,3),
                                     "sim_thresh":params['sim_thresh'],
                                     "is_min_thresh":True
                                     },
                     clustering_params={
                             'algorithm':'spectral_clustering',
                             'n_clusters':params['n_clusters']
                             },return_value='division')