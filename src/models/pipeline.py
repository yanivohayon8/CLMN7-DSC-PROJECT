# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:04:13 2020

@author: yaniv
"""

import sys
sys.path.append('../../')

from src.features.segment_transcript import CreateBlocks
from src.features.build_vectors import vectorizer
from src.features.similiarity_calc import similarity
from src.visualization.visualize import MyPlotting
from src.models.clustering import clustering

import numpy as np

import gensim 
from gensim.models import Word2Vec
import gensim.downloader as api

#word2vec_wiki_model = api.load('glove-wiki-gigaword-300')

#https://radimrehurek.com/gensim/scripts/glove2word2vec.html
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile

import pandas as pd

''' slow loading :
#glove_file = datapath('C:\\Users\\yaniv\\Desktop\\gensim\\glove.6B.50d.txt')
#tmp_file = get_tmpfile("C:\\Users\\yaniv\\Desktop\\gensim\\glove.6B.50d_word2vec.txt")
#_ = glove2word2vec(glove_file, tmp_file)
#word2vec_wiki_model  = KeyedVectors.load_word2vec_format(tmp_file)

##word2vec_wiki_model = KeyedVectors.load_word2vec_format('C:\\Users\\yaniv\\Desktop\\gensim\\glove.6B.50d_word2vec.txt')
'''

''' 
word2vec_wiki_model.save_word2vec_format('C:\\Users\\yaniv\\Desktop\\gensim\\glove.6B.50d_word2vec.bin',binary=True)
last answer : https://stackoverflow.com/questions/42986405/how-to-speed-up-gensim-word2vec-model-load-time 
'''
word2vec_wiki_model = KeyedVectors.load('C:\\Users\\yaniv\\Desktop\\gensim\\glove.6B.50d_word2vec_test.bin',mmap='r')

class pipeline():#,myvectorizer
    
    '''def __init__(self,df,groundbase,video_id,video_len,transcripts_jsons,\
                 window_size=40,step_size=20,\
                 vector_method='tfidf',similarity_method='cosine',is_min_thresh=True
                 ):
        transcripts = transcripts_jsons
        window_size = window_size
        step_size = step_size
        vector_method = vector_method
        similarity_method = similarity_method
        is_min_thresh=is_min_thresh
        groundbase = groundbase
        df = df
        video_id = video_id
        video_len = video_len
    '''
        
    '''
        This class runs the following pipeline:
            segmentation
            vectorization
            similiary calc
            processing similirity(not must, for example,thersholds)
            clustering
    '''
    
    
    #word2vec_wiki_model = api.load('glove-wiki-gigaword-300')
    
    @staticmethod
    def run(df,groundbase,video_id,video_len,transcripts,\
            figure_path=None,\
            window_size=40,step_size=20,\
            vector_method='tfidf',vectorizing_params=None, 
            similarity_method='cosine',is_min_thresh=True,\
            algorithm='spectral_clustering',n_clusters=4,clustering_params=None,\
            sim_thresh=None,sim_filter=None,accurrcy_shift=15):
        
        '''Initializing parameters '''
        w2v_model = None
        recall,precision,tp,fp,fn = 0,0,0,0,0
        is_failed = False
        failure_mess = None
        if vectorizing_params is not None:
            vectorizing_params['n_clusters'] = n_clusters
        
        try:
            ''' Segmenting transcripts'''
            block_handler =  CreateBlocks(transcripts,window_size=window_size)
            blocks = block_handler.partion_by_sliding_windows(window_size,step_size)
            gap_timestamp = block_handler.get_block_gap_timestamp()        
            method_label = 'chunking'        
            if window_size == step_size:
                method_label = 'slidingwindow'
                
            ''' vectorizing the segment '''        
            if 'word2vec' in vector_method:
                w2v_model = word2vec_wiki_model            
            vector_array = vectorizer.calc(blocks,vector_method,w2v_model,vectorizing_params)
            method_label = method_label + '_' + vector_method
            
            
            ''' Calculate similarity'''        
            if similarity_method == 'wmdistance':
                w2v_model = word2vec_wiki_model        
            similarity_matrix = similarity.calc_adjacent_matrix(vector_array,similarity_method,w2v_model)
            method_label = method_label + '_' + similarity_method
            
            ''' Ploting similarity'''
            df_stats_similarity = pd.DataFrame(similarity_matrix.reshape(-1,1))
            MyPlotting.print_stats(similarity_matrix,df_stats_similarity,\
                                   'raw similarity matrix',figure_path=figure_path)
            #MyPlotting.similarity_matrix(similarity_matrix,min_=_min,max_=_max,median_=_median,\
            #                             title='raw similarity matrix',figure_path=figure_path)
            
            '''Applying threshold to the similarity matrix'''
            if sim_thresh is not None:
                if is_min_thresh:
                    # make shure if 0 is o'right!
                    similarity_matrix = np.asarray([0 if sim < sim_thresh else sim \
                                        for sim in np.concatenate(similarity_matrix.reshape(-1,1))]).reshape(similarity_matrix.shape)
                else: 
                    similarity_matrix = np.asarray([100 if sim > sim_thresh else sim \
                                        for sim in np.concatenate(similarity_matrix.reshape(-1,1))]).reshape(similarity_matrix.shape)
                method_label = method_label + '_threshold'

            ''' Plotting similarity'''                
            df_stats_similarity = pd.DataFrame(similarity_matrix.reshape(-1,1))
            MyPlotting.print_stats(similarity_matrix,df_stats_similarity,\
                                   'similarity matrix after threshold',figure_path=figure_path)
            #MyPlotting.similarity_matrix(similarity_matrix,'similarity matrix after threshold',figure_path)
            
            '''Appplying filter on the similiry matrix'''
            if sim_filter is not None:
                similarity_matrix = similarity.apply_filter(similarity_matrix,\
                                                            filter_type=sim_filter[0],params=sim_filter[1:])
                df_stats_similarity = pd.DataFrame(similarity_matrix.reshape(-1,1))
                MyPlotting.print_stats(similarity_matrix,df_stats_similarity,\
                                   'similarity matrix after %s %s filter' %(sim_filter[0],sim_filter[1]),\
                                   figure_path=figure_path)

                #MyPlotting.similarity_matrix(similarity_matrix,'similarity matrix after %s %s filter' %(sim_filter[0],sim_filter[1]),figure_path)    
                method_label = method_label + '_filter'
            
            '''Execute clustering'''
            recall,precision,tp,fp,fn  = clustering.run(similarity_matrix,n_clusters,gap_timestamp,\
                                                            groundbase,algorithm,accurrcy_shift,clustering_params)
        except Exception as inst:
            print(inst)
            failure_mess = inst
            is_failed = True
            
        method_label = method_label + '_' + algorithm
        
        df = df.append({'METHOD':\
                                method_label,\
                                'RECALL':recall,\
                                'PRECISION':precision,\
                                'BLOCKSIZE':window_size,\
                                'STEPSIZE':step_size\
                                ,'NUMOFCLUSTERSFORSC':n_clusters,\
                                'THERSHOLD':sim_thresh,\
                                'FILTER': str(sim_filter),\
                               'VIDEO':video_id,\
                               'VIDEOLENGTH':video_len,\
                               'NUMTOPICS':len(groundbase),\
                               'TP':tp,
                               'FP':fp,
                               'FN':fn,
                               'ACCURECYSECONDS': accurrcy_shift,
                               'ISFAILED':is_failed,
                               'FAILUREMESSAGE': failure_mess,
                               'CLUSTERINGPARAMS':clustering_params,
                               'VECTORIZATIONPARAMS':vectorizing_params
                               }
                ,ignore_index=True
                
                )
        
        return df


