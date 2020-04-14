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
glove_file = datapath('C:\\Users\\yaniv\\Desktop\\gensim\\glove.6B.50d.txt')
tmp_file = get_tmpfile("C:\\Users\\yaniv\\Desktop\\gensim\\glove.6B.50d_word2vec.txt")
_ = glove2word2vec(glove_file, tmp_file)
word2vec_wiki_model  = KeyedVectors.load_word2vec_format(tmp_file)
#word2vec_wiki_model = KeyedVectors.load_word2vec_format('C:\\Users\\yaniv\\Desktop\\gensim\\glove.6B.50d_word2vec.txt')


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
            vector_method='tfidf',similarity_method='cosine',is_min_thresh=True,\
            algorithm='spectral_clustering',n_clusters=4,sim_thresh=None,sim_filter=None,accurrcy_shift=15):
        
        '''Initializing parameters '''
        w2v_model = None
        
        ''' Segmenting transcripts'''
        block_handler =  CreateBlocks(transcripts,window_size=window_size)
        blocks = block_handler.partion_by_sliding_windows(window_size,step_size)
        gap_timestamp = block_handler.get_block_gap_timestamp()        
        method_label = 'chunking'        
        if window_size == step_size:
            method_label = 'slidingwindow'
            
        ''' vectorizing the segment '''        
        if vector_method == 'word2vec_wiki':
            w2v_model = word2vec_wiki_model            
        vector_array = vectorizer.calc(blocks,vector_method,w2v_model)
        method_label = method_label + '_' + vector_method
        
        
        ''' Calculate similarity'''        
        if similarity_method == 'wmdistance':
            w2v_model = word2vec_wiki_model        
        similarity_matrix = similarity.calc_adjacent_matrix(vector_array,similarity_method,w2v_model)
        method_label = method_label + '_' + similarity_method
        
        ''' Ploting similarity'''
        MyPlotting.similarity_matrix(similarity_matrix,'raw similarity matrix',figure_path)
        
        '''Applying threshold to the similarity matrix'''
        if sim_thresh is not None:
            if is_min_thresh:
                # make shure if 0 is o'right!
                similarity_matrix = np.asarray([0 if sim < sim_thresh else sim \
                                    for sim in np.concatenate(similarity_matrix.reshape(-1,1))]).reshape(similarity_matrix.shape)
                #debug
                '''mat = []
                for sim in np.concatenate(similarity_matrix.reshape(-1,1)):
                    print(sim)
                    val = sim
                    if sim < sim_thresh:
                        val = 0 
                    mat.append(val)
                similarity_matrix = np.asarray(mat).reshape(similarity_matrix.shape)'''
            else: 
                similarity_matrix = np.asarray([100 if sim > sim_thresh else sim \
                                    for sim in np.concatenate(similarity_matrix.reshape(-1,1))]).reshape(similarity_matrix.shape)
            method_label = method_label + '_threshold'
            MyPlotting.similarity_matrix(similarity_matrix,'similarity matrix after threshold',figure_path)
        
        '''Appplying filter on the similiry matrix'''
        if sim_filter is not None:
            similarity_matrix = similarity.apply_filter(similarity_matrix,filter_type=sim_filter[0],mask_size=sim_filter[1])
            MyPlotting.similarity_matrix(similarity_matrix,'similarity matrix after %s %s filter' %(sim_filter[0],sim_filter[1]),figure_path)    
            method_label = method_label + '_filter'
        
        '''Execute clustering'''
        recall,precision,tp,fp,fn = 0,0,0,0,0
        is_failed = False
        failure_mess = None
        
        try:
            recall,precision,tp,fp,fn  = clustering.run(similarity_matrix,n_clusters,gap_timestamp,\
                                                        groundbase,algorithm,accurrcy_shift)
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
                               'FAILUREMESSAGE': failure_mess
                               }
                ,ignore_index=True
                
                )
        
        return df


import pandas as pd      
groundbase = [22,87,207,270,319,370,450,558,602,671,756,930,978,1011]
transcripts = [{'text': 'hello my name is Tommy and in this video', 'start': 0.03, 'duration': 5.37}, {'text': "I'll be giving an introduction to kernel", 'start': 3.21, 'duration': 4.74}, {'text': 'density estimation this is going to be a', 'start': 5.4, 'duration': 4.619}, {'text': "graphical tutorial I'm going to show you", 'start': 7.95, 'duration': 3.96}, {'text': "a lot of plots I'm going to show you", 'start': 10.019, 'duration': 4.561}, {'text': "some equations but it's not meant to be", 'start': 11.91, 'duration': 5.25}, {'text': "rigorous I'll tell you at the end or you", 'start': 14.58, 'duration': 5.64}, {'text': 'can go to find literature if you really', 'start': 17.16, 'duration': 5.279}, {'text': "want to read up on this let's get", 'start': 20.22, 'duration': 3.53}, {'text': 'straight to it', 'start': 22.439, 'duration': 4.441}, {'text': 'kernel density estimation is a way of', 'start': 23.75, 'duration': 5.289}, {'text': 'estimating an unknown probability', 'start': 26.88, 'duration': 6.69}, {'text': 'density function given some data now in', 'start': 29.039, 'duration': 6.331}, {'text': 'middle school you probably learned about', 'start': 33.57, 'duration': 4.47}, {'text': 'the histogram and this is a bit like the', 'start': 35.37, 'duration': 5.7}, {'text': 'histogram the idea is that you define a', 'start': 38.04, 'duration': 5.91}, {'text': 'kernel function and you Center a kernel', 'start': 41.07, 'duration': 6.3}, {'text': 'function on each data point so and on', 'start': 43.95, 'duration': 5.4}, {'text': 'every data point in your sample you', 'start': 47.37, 'duration': 6.84}, {'text': 'place a kernel function after doing this', 'start': 49.35, 'duration': 7.47}, {'text': 'you sum these functions together so you', 'start': 54.21, 'duration': 5.06}, {'text': 'add the first kernel the second one', 'start': 56.82, 'duration': 7.079}, {'text': 'third one fourth and fifth and then you', 'start': 59.27, 'duration': 7.72}, {'text': 'have a kernel density estimate the one', 'start': 63.899, 'duration': 5.911}, {'text': 'over N in the equation normalizes the', 'start': 66.99, 'duration': 4.83}, {'text': 'estimate since every kernel function', 'start': 69.81, 'duration': 4.41}, {'text': 'must have an integral evaluating to one', 'start': 71.82, 'duration': 5.31}, {'text': 'and we added five the integral will be', 'start': 74.22, 'duration': 6.03}, {'text': 'five then we divide through by five to', 'start': 77.13, 'duration': 6.36}, {'text': 'to have a an integral of one again so', 'start': 80.25, 'duration': 9.329}, {'text': "that's the purpose now the kernel", 'start': 83.49, 'duration': 11.82}, {'text': 'function can vary typically you require', 'start': 89.579, 'duration': 8.22}, {'text': 'these three things of a kernel function', 'start': 95.31, 'duration': 4.98}, {'text': 'it should be non-negative because a', 'start': 97.799, 'duration': 5.941}, {'text': 'probability is always non-negative it', 'start': 100.29, 'duration': 6.09}, {'text': 'should be symmetric meaning that if you', 'start': 103.74, 'duration': 4.559}, {'text': 'go to the left or you go to the right of', 'start': 106.38, 'duration': 4.23}, {'text': 'your data point the kernel should have', 'start': 108.299, 'duration': 4.11}, {'text': 'the same value and it should be', 'start': 110.61, 'duration': 3.93}, {'text': 'decreasing so that when you go away from', 'start': 112.409, 'duration': 6.6}, {'text': 'the data point the kernel goes closer to', 'start': 114.54, 'duration': 8.46}, {'text': "zero it doesn't have to have these three", 'start': 119.009, 'duration': 8.011}, {'text': 'properties but typically in the simple', 'start': 123.0, 'duration': 6.3}, {'text': 'straightforward cases this is what you', 'start': 127.02, 'duration': 3.709}, {'text': 'demand from a kernel', 'start': 129.3, 'duration': 4.7}, {'text': 'function now a kernel function can be of', 'start': 130.729, 'duration': 5.971}, {'text': 'bounded support or not in this example', 'start': 134.0, 'duration': 5.129}, {'text': 'the Gaussian to the left does not have', 'start': 136.7, 'duration': 4.319}, {'text': 'bounded support because it never really', 'start': 139.129, 'duration': 5.881}, {'text': 'goes down to zero it goes infinitely', 'start': 141.019, 'duration': 7.291}, {'text': 'close to zero as X gets infinitely', 'start': 145.01, 'duration': 5.88}, {'text': 'larger or smaller but it never really', 'start': 148.31, 'duration': 5.91}, {'text': 'reaches zero this is in contrast to the', 'start': 150.89, 'duration': 6.03}, {'text': 'Box kernel the triangular kernel and the', 'start': 154.22, 'duration': 6.329}, {'text': 'try weight kernel which is zero outside', 'start': 156.92, 'duration': 11.19}, {'text': "of a domain let's see one more example", 'start': 160.549, 'duration': 10.401}, {'text': 'now with the triangular kernel we add', 'start': 168.11, 'duration': 6.3}, {'text': 'the kernel onto every single data point', 'start': 170.95, 'duration': 7.99}, {'text': 'and then we sum these together for a', 'start': 174.41, 'duration': 8.31}, {'text': 'density estimate notice that this', 'start': 178.94, 'duration': 6.0}, {'text': 'estimate is not nearly as smooth as the', 'start': 182.72, 'duration': 4.73}, {'text': 'Gaussian one that we saw earlier the', 'start': 184.94, 'duration': 5.31}, {'text': 'choice of kernel is actually not that', 'start': 187.45, 'duration': 6.25}, {'text': 'important because once you start getting', 'start': 190.25, 'duration': 5.94}, {'text': 'data the estimates are going to look', 'start': 193.7, 'duration': 4.379}, {'text': 'very similar no matter what kernel you', 'start': 196.19, 'duration': 4.139}, {'text': "choose so it's not really crucial to", 'start': 198.079, 'duration': 6.151}, {'text': 'choose a perfect kernel or anything like', 'start': 200.329, 'duration': 5.731}, {'text': 'that usually the Gaussian will do just', 'start': 204.23, 'duration': 5.939}, {'text': 'fine however the choice of bandwidth is', 'start': 206.06, 'duration': 8.69}, {'text': 'very important we use a bandwidth H and', 'start': 210.169, 'duration': 8.491}, {'text': 'we divide through through in the kernel', 'start': 214.75, 'duration': 7.72}, {'text': 'function if H is large then it will', 'start': 218.66, 'duration': 6.21}, {'text': 'spread the kernel function out we have', 'start': 222.47, 'duration': 5.19}, {'text': 'to divide by H on the outside of the', 'start': 224.87, 'duration': 5.519}, {'text': 'kernel function again to ensure that the', 'start': 227.66, 'duration': 5.969}, {'text': "integral equals 1 so that's why I have", 'start': 230.389, 'duration': 6.061}, {'text': 'an H outside and inside of the kernel', 'start': 233.629, 'duration': 8.311}, {'text': 'function capital K now the plot that you', 'start': 236.45, 'duration': 7.98}, {'text': 'see here has a very small bandwidth and', 'start': 241.94, 'duration': 4.439}, {'text': 'if I increase the bandwidth it looks', 'start': 244.43, 'duration': 4.369}, {'text': 'like this', 'start': 246.379, 'duration': 2.42}, {'text': "so it's not really clear what the", 'start': 252.9, 'duration': 6.01}, {'text': 'optimal value is clearly the first', 'start': 255.37, 'duration': 5.61}, {'text': 'estimate is too narrow', 'start': 258.91, 'duration': 6.51}, {'text': 'this might be to spread too large so', 'start': 260.98, 'duration': 7.08}, {'text': 'there are a few methods to automatically', 'start': 265.42, 'duration': 5.19}, {'text': 'choose a bandwidth and the simple one is', 'start': 268.06, 'duration': 6.12}, {'text': "called Silverman's rule of thumb now it", 'start': 270.61, 'duration': 6.33}, {'text': 'computes an optimal age by assuming that', 'start': 274.18, 'duration': 5.13}, {'text': 'the data is normally distributed and', 'start': 276.94, 'duration': 5.13}, {'text': 'this is somewhat paradoxical because if', 'start': 279.31, 'duration': 4.32}, {'text': 'you really knew that the data was', 'start': 282.07, 'duration': 3.15}, {'text': "normally distributed you wouldn't use", 'start': 283.63, 'duration': 3.54}, {'text': 'kernel density estimation you would use', 'start': 285.22, 'duration': 6.48}, {'text': 'maximum likelihood to just estimate mu', 'start': 287.17, 'duration': 6.26}, {'text': 'and Sigma', 'start': 291.7, 'duration': 6.21}, {'text': 'but typically your data might be close', 'start': 293.43, 'duration': 5.23}, {'text': 'to normal', 'start': 297.91, 'duration': 3.63}, {'text': "so then Silverman's rule of thumb is a", 'start': 298.66, 'duration': 5.16}, {'text': 'good starting point for instance', 'start': 301.54, 'duration': 4.86}, {'text': 'consider this example we have a standard', 'start': 303.82, 'duration': 4.62}, {'text': 'normal distribution we generate some', 'start': 306.4, 'duration': 4.74}, {'text': "data and then we use Silverman's rule of", 'start': 308.44, 'duration': 6.24}, {'text': "thumb this is our estimate so it's", 'start': 311.14, 'duration': 10.02}, {'text': "fairly good there's a different", 'start': 314.68, 'duration': 8.49}, {'text': "algorithm which is better if you're if", 'start': 321.16, 'duration': 4.2}, {'text': 'you have a lot of data or if your data', 'start': 323.17, 'duration': 4.56}, {'text': 'is multimodal meaning there are several', 'start': 325.36, 'duration': 5.58}, {'text': 'modes so consider this data you have two', 'start': 327.73, 'duration': 5.37}, {'text': "normal distributions they're spread", 'start': 330.94, 'duration': 4.47}, {'text': 'apart and we generate some data from', 'start': 333.1, 'duration': 5.88}, {'text': 'this probability density function we get', 'start': 335.41, 'duration': 6.63}, {'text': "this data if we use Silverman's rule of", 'start': 338.98, 'duration': 4.98}, {'text': 'thumb we end up with a poor estimate', 'start': 342.04, 'duration': 5.79}, {'text': 'where the bandwidth is too large if we', 'start': 343.96, 'duration': 6.3}, {'text': "use the improved eurozone's algorithm we", 'start': 347.83, 'duration': 6.03}, {'text': 'get a far better estimate we do need', 'start': 350.26, 'duration': 6.0}, {'text': 'quite a bit of data to make them improve', 'start': 353.86, 'duration': 4.44}, {'text': 'jitter Jones algorithm do a good job so', 'start': 356.26, 'duration': 4.32}, {'text': "that's one disadvantage but if you", 'start': 358.3, 'duration': 4.86}, {'text': "suspect something that's far from normal", 'start': 360.58, 'duration': 5.73}, {'text': "or something that's bimodal then this", 'start': 363.16, 'duration': 6.89}, {'text': 'algorithm does a good job', 'start': 366.31, 'duration': 3.74}, {'text': 'the next thing to look at is weighing', 'start': 371.55, 'duration': 6.089}, {'text': 'the data in the previous example every', 'start': 373.71, 'duration': 7.079}, {'text': 'data point was way weight weighted', 'start': 377.639, 'duration': 5.49}, {'text': 'uniformly meaning it had the same weight', 'start': 380.789, 'duration': 6.511}, {'text': 'but you might have cases were each data', 'start': 383.129, 'duration': 7.38}, {'text': 'point has a weight for instance if these', 'start': 387.3, 'duration': 6.81}, {'text': 'data points are the age of people and', 'start': 390.509, 'duration': 6.451}, {'text': 'you want to know the distribution of net', 'start': 394.11, 'duration': 7.679}, {'text': 'worth over H you might put H on the', 'start': 396.96, 'duration': 7.44}, {'text': 'x-axis and weight your data points by', 'start': 401.789, 'duration': 6.711}, {'text': 'the net worth of each individual and', 'start': 404.4, 'duration': 7.59}, {'text': 'what you do is you replace the one over', 'start': 408.5, 'duration': 6.069}, {'text': 'n with weights and you assign a weight', 'start': 411.99, 'duration': 4.47}, {'text': 'to every data point and you should', 'start': 414.569, 'duration': 3.84}, {'text': "ensure that the sum of the way it's", 'start': 416.46, 'duration': 5.28}, {'text': 'equal one so that the integral of the', 'start': 418.409, 'duration': 6.51}, {'text': 'your estimate equals one in the end and', 'start': 421.74, 'duration': 5.64}, {'text': "apart from that it's the same thing you", 'start': 424.919, 'duration': 8.641}, {'text': 'add kernels and you weigh them so they', 'start': 427.38, 'duration': 9.089}, {'text': 'can look a bit different and then you', 'start': 433.56, 'duration': 9.449}, {'text': 'sum these together and this is your', 'start': 436.469, 'duration': 9.35}, {'text': 'final estimate so pretty straightforward', 'start': 443.009, 'duration': 5.22}, {'text': "there's one more thing that I want to", 'start': 445.819, 'duration': 4.871}, {'text': "discuss in one dimension and that's", 'start': 448.229, 'duration': 5.101}, {'text': 'bound the domains it happens quite often', 'start': 450.69, 'duration': 4.979}, {'text': "that you know that you're working on a", 'start': 453.33, 'duration': 4.47}, {'text': 'bounded domain for instance if your data', 'start': 455.669, 'duration': 4.68}, {'text': 'is the age of people or the net worth of', 'start': 457.8, 'duration': 7.47}, {'text': 'individuals then you know that your', 'start': 460.349, 'duration': 12.031}, {'text': 'density is 0 when access is smaller than', 'start': 465.27, 'duration': 10.98}, {'text': '0 so you know that your data supposed to', 'start': 472.38, 'duration': 7.5}, {'text': 'be for instance to the right side of', 'start': 476.25, 'duration': 7.38}, {'text': 'some some boundary now if you just', 'start': 479.88, 'duration': 5.759}, {'text': 'compute the kernel density estimate on', 'start': 483.63, 'duration': 5.159}, {'text': "this data you get this and it's a bit", 'start': 485.639, 'duration': 5.881}, {'text': 'unfortunate because it places density to', 'start': 488.789, 'duration': 5.49}, {'text': 'the left of the boundary so think of', 'start': 491.52, 'duration': 4.47}, {'text': 'this being H and you just said that', 'start': 494.279, 'duration': 3.51}, {'text': "there's a probability for people being", 'start': 495.99, 'duration': 5.28}, {'text': "less than 0 of H which doesn't really", 'start': 497.789, 'duration': 5.731}, {'text': 'make a lot of sense there are many ways', 'start': 501.27, 'duration': 3.93}, {'text': 'to deal with this but I want to enter', 'start': 503.52, 'duration': 4.08}, {'text': "a simple way to deal with it and it's", 'start': 505.2, 'duration': 5.7}, {'text': 'called mirroring the data what you do is', 'start': 507.6, 'duration': 5.19}, {'text': 'you mirror the data about the boundary', 'start': 510.9, 'duration': 4.8}, {'text': 'like so then you compute the kernel', 'start': 512.79, 'duration': 5.58}, {'text': 'density estimate on this new mirror the', 'start': 515.7, 'duration': 6.09}, {'text': 'data then you sum your original and', 'start': 518.37, 'duration': 7.82}, {'text': 'mirrored the kernel density estimate and', 'start': 521.79, 'duration': 8.64}, {'text': "then you chop this so that it's zero to", 'start': 526.19, 'duration': 9.04}, {'text': 'the left of the boundary now compare the', 'start': 530.43, 'duration': 7.68}, {'text': 'blue final kernel density estimate with', 'start': 535.23, 'duration': 8.61}, {'text': "the red one you see that it's it's moved", 'start': 538.11, 'duration': 7.86}, {'text': 'some of the density from the left of the', 'start': 543.84, 'duration': 3.9}, {'text': 'boundary to the right of the boundary so', 'start': 545.97, 'duration': 6.0}, {'text': 'this is a simple trick to ensure to', 'start': 547.74, 'duration': 6.93}, {'text': "ensure that you don't get this bias at", 'start': 551.97, 'duration': 6.65}, {'text': 'the boundary if you have bounded domains', 'start': 554.67, 'duration': 7.2}, {'text': "let's go to D dimensions or more", 'start': 558.62, 'duration': 6.6}, {'text': "specifically let's go to two dimensions", 'start': 561.87, 'duration': 6.96}, {'text': 'now if you want to extend to higher', 'start': 565.22, 'duration': 6.16}, {'text': 'dimensions one way to do it is to', 'start': 568.83, 'duration': 6.51}, {'text': 'introduce a normal because you need some', 'start': 571.38, 'duration': 6.09}, {'text': 'measure of distance in higher dimensions', 'start': 575.34, 'duration': 5.16}, {'text': 'and there are several to choose so', 'start': 577.47, 'duration': 5.28}, {'text': "you're gonna choose a P norm for", 'start': 580.5, 'duration': 5.04}, {'text': "instance and you're gonna replace the", 'start': 582.75, 'duration': 5.01}, {'text': 'one over H with one over H to the power', 'start': 585.54, 'duration': 5.76}, {'text': 'of D to normalize and apart from that', 'start': 587.76, 'duration': 6.24}, {'text': "it's the same thing so here you see four", 'start': 591.3, 'duration': 6.3}, {'text': 'kernels in two dimensions the box kernel', 'start': 594.0, 'duration': 5.37}, {'text': 'the triangular kernel the by weight', 'start': 597.6, 'duration': 5.52}, {'text': 'kernel and the Gaussian kernel the', 'start': 599.37, 'duration': 6.06}, {'text': 'choice of norm really matters in higher', 'start': 603.12, 'duration': 5.52}, {'text': "dimensions because you're allowed to", 'start': 605.43, 'duration': 8.58}, {'text': "pick a norm and it's not obvious in", 'start': 608.64, 'duration': 8.04}, {'text': 'every case which norm is the correct one', 'start': 614.01, 'duration': 5.46}, {'text': 'typically P equals two in the penal', 'start': 616.68, 'duration': 5.88}, {'text': 'which corresponds to standard Euclidean', 'start': 619.47, 'duration': 5.04}, {'text': "distance it's a good choice because it's", 'start': 622.56, 'duration': 4.2}, {'text': 'invariant under rotation but there are', 'start': 624.51, 'duration': 5.36}, {'text': 'other choices as well', 'start': 626.76, 'duration': 3.11}, {'text': 'the common choices are P equals one P', 'start': 631.9, 'duration': 6.06}, {'text': 'equals two and P equals infinity P', 'start': 635.11, 'duration': 4.71}, {'text': 'equals one is often called the Manhattan', 'start': 637.96, 'duration': 4.319}, {'text': "distance because it's the distance that", 'start': 639.82, 'duration': 4.8}, {'text': 'you have to travel in a grid so imagine', 'start': 642.279, 'duration': 4.381}, {'text': 'the city of Manhattan with a streets', 'start': 644.62, 'duration': 5.25}, {'text': 'looking looking like a grid P equals two', 'start': 646.66, 'duration': 4.89}, {'text': 'is the Euclidean norm and P equals', 'start': 649.87, 'duration': 6.9}, {'text': "infinity is the maximum normal let's", 'start': 651.55, 'duration': 7.979}, {'text': 'look at these kernel functions as we', 'start': 656.77, 'duration': 7.62}, {'text': 'change the norm the Box kernel looks', 'start': 659.529, 'duration': 8.401}, {'text': 'different in different norms and so does', 'start': 664.39, 'duration': 6.18}, {'text': 'the triangular kernel and the Gaussian', 'start': 667.93, 'duration': 6.12}, {'text': "kernel let's look at what happens when", 'start': 670.57, 'duration': 7.26}, {'text': "we have some data now you're gonna see", 'start': 674.05, 'duration': 6.68}, {'text': 'that as the number of data points', 'start': 677.83, 'duration': 5.58}, {'text': 'increases the choice of kernel and the', 'start': 680.73, 'duration': 6.01}, {'text': "choice of of P in the P norm doesn't", 'start': 683.41, 'duration': 4.77}, {'text': "really matter it's going to look more", 'start': 686.74, 'duration': 6.56}, {'text': "and more similar so let's add some data", 'start': 688.18, 'duration': 8.909}, {'text': 'it still looks pretty different but as', 'start': 693.3, 'duration': 5.77}, {'text': 'we increase the data and work our way', 'start': 697.089, 'duration': 4.5}, {'text': "towards a thousand data points you'll", 'start': 699.07, 'duration': 5.19}, {'text': 'see that the estimates grow closer and', 'start': 701.589, 'duration': 4.801}, {'text': 'closer and in the end it looks very much', 'start': 704.26, 'duration': 5.19}, {'text': 'the same so just like in one dimension', 'start': 706.39, 'duration': 4.86}, {'text': 'the choice of kernel is not really that', 'start': 709.45, 'duration': 2.52}, {'text': 'important', 'start': 711.25, 'duration': 2.85}, {'text': 'the choice of kernel and the choice of P', 'start': 711.97, 'duration': 4.35}, {'text': 'norm is not really that important in', 'start': 714.1, 'duration': 5.52}, {'text': "higher dimensions so you don't really", 'start': 716.32, 'duration': 5.25}, {'text': 'have to worry too much the two norm is', 'start': 719.62, 'duration': 3.75}, {'text': "typically fine but it's still an", 'start': 721.57, 'duration': 3.87}, {'text': 'interesting point to bring out the', 'start': 723.37, 'duration': 4.17}, {'text': 'bandwidth is still important and in', 'start': 725.44, 'duration': 4.11}, {'text': 'higher dimensions the bandwidth is not', 'start': 727.54, 'duration': 4.2}, {'text': 'necessarily a number anymore it could be', 'start': 729.55, 'duration': 4.65}, {'text': 'a matrix because you could have', 'start': 731.74, 'duration': 4.14}, {'text': 'different bandwidths in different', 'start': 734.2, 'duration': 5.46}, {'text': "directions and it doesn't really have to", 'start': 735.88, 'duration': 7.649}, {'text': 'be aligned aligned with the principal', 'start': 739.66, 'duration': 7.08}, {'text': 'axis so we could have a matrix of', 'start': 743.529, 'duration': 5.491}, {'text': 'bandwidths and in the higher dimensional', 'start': 746.74, 'duration': 8.599}, {'text': 'case a D times D matrix in D dimensions', 'start': 749.02, 'duration': 6.319}, {'text': "let's look at a fast algorithm for", 'start': 756.779, 'duration': 4.571}, {'text': 'actually computing a kernel density', 'start': 759.73, 'duration': 3.57}, {'text': "estimate this is just if you're", 'start': 761.35, 'duration': 4.409}, {'text': "interested it's a very quick explanation", 'start': 763.3, 'duration': 7.979}, {'text': "and it's interesting to see so the fast", 'start': 765.759, 'duration': 8.851}, {'text': 'computation in one dimension is', 'start': 771.279, 'duration': 5.55}, {'text': 'performed using linear binning and then', 'start': 774.61, 'duration': 4.44}, {'text': "convolution it's not the only fast", 'start': 776.829, 'duration': 4.711}, {'text': 'algorithm but in practice this is really', 'start': 779.05, 'duration': 4.649}, {'text': "fast and it's quite a simple algorithm", 'start': 781.54, 'duration': 4.039}, {'text': "so I'd really like to show it to you", 'start': 783.699, 'duration': 4.411}, {'text': 'imagine that you have data and you have', 'start': 785.579, 'duration': 5.62}, {'text': 'a grid your grid is equidistant meaning', 'start': 788.11, 'duration': 6.0}, {'text': 'the distance between every two grid', 'start': 791.199, 'duration': 5.791}, {'text': "points is the same so it's for instance", 'start': 794.11, 'duration': 5.099}, {'text': 'one two three four and so forth and then', 'start': 796.99, 'duration': 3.93}, {'text': 'we have some data and the data is not', 'start': 799.209, 'duration': 4.62}, {'text': 'like with this need necessarily what you', 'start': 800.92, 'duration': 5.31}, {'text': 'do is you go through every data point', 'start': 803.829, 'duration': 5.7}, {'text': 'and then you assign assign weight to the', 'start': 806.23, 'duration': 6.18}, {'text': 'grid points that are close to every data', 'start': 809.529, 'duration': 6.18}, {'text': 'point so go to the first data point and', 'start': 812.41, 'duration': 6.21}, {'text': 'then you assign weights to grid points', 'start': 815.709, 'duration': 5.911}, {'text': 'one and two this assigns slightly more', 'start': 818.62, 'duration': 5.25}, {'text': 'weight to grid point two because the', 'start': 821.62, 'duration': 4.23}, {'text': 'data point is slightly closer to grid', 'start': 823.87, 'duration': 5.009}, {'text': 'point two into grid point one same with', 'start': 825.85, 'duration': 5.489}, {'text': 'the next one and then this third data', 'start': 828.879, 'duration': 4.291}, {'text': 'point is very close to the third grid', 'start': 831.339, 'duration': 3.961}, {'text': 'point so it will almost exclusively', 'start': 833.17, 'duration': 6.149}, {'text': 'assign weight to this grid point then', 'start': 835.3, 'duration': 5.87}, {'text': 'you go through the data like this', 'start': 839.319, 'duration': 4.77}, {'text': 'assigning weights this one is just to', 'start': 841.17, 'duration': 4.389}, {'text': 'the left of the grid point to it it will', 'start': 844.089, 'duration': 4.651}, {'text': 'assign almost everything to this fifth', 'start': 845.559, 'duration': 8.101}, {'text': 'grid point and so now this algorithm', 'start': 848.74, 'duration': 7.05}, {'text': 'clearly has to go through every data', 'start': 853.66, 'duration': 5.399}, {'text': 'point and then has to go to the two', 'start': 855.79, 'duration': 6.51}, {'text': 'closest grid points so the complexity is', 'start': 859.059, 'duration': 6.481}, {'text': 'o of capital n the number of data points', 'start': 862.3, 'duration': 6.24}, {'text': 'times two to the power of D because in', 'start': 865.54, 'duration': 4.649}, {'text': 'higher dimensions there will be two to', 'start': 868.54, 'duration': 4.409}, {'text': 'the power of the d grid points that are', 'start': 870.189, 'duration': 4.62}, {'text': 'adjacent which you have to visit and', 'start': 872.949, 'duration': 7.651}, {'text': "assign weights to once you've finished", 'start': 874.809, 'duration': 8.58}, {'text': 'this part you sample your kernel so you', 'start': 880.6, 'duration': 5.64}, {'text': 'take your kernel looks like this and you', 'start': 883.389, 'duration': 5.671}, {'text': 'also sample it at equidistant points', 'start': 886.24, 'duration': 5.43}, {'text': 'now you have two sequences or two', 'start': 889.06, 'duration': 5.4}, {'text': 'vectors of samples on equidistant points', 'start': 891.67, 'duration': 6.42}, {'text': 'and you can use the convolution discrete', 'start': 894.46, 'duration': 7.11}, {'text': 'convolution to actually compute your', 'start': 898.09, 'duration': 7.89}, {'text': 'kernel density estimate now this can be', 'start': 901.57, 'duration': 6.81}, {'text': 'done using the fast Fourier transform', 'start': 905.98, 'duration': 4.2}, {'text': 'and this is called the convolution', 'start': 908.38, 'duration': 5.64}, {'text': 'theorem and runs an n log n time where n', 'start': 910.18, 'duration': 7.52}, {'text': 'as small n is the number of grid points', 'start': 914.02, 'duration': 6.15}, {'text': 'so the total running time of this', 'start': 917.7, 'duration': 6.01}, {'text': 'algorithm is given by capital n times 2', 'start': 920.17, 'duration': 7.77}, {'text': "to the power of D plus n log n and it's", 'start': 923.71, 'duration': 7.98}, {'text': "very fast for many data points let's", 'start': 927.94, 'duration': 7.1}, {'text': 'look at two dimensional linear binning', 'start': 931.69, 'duration': 7.32}, {'text': 'this red data point a science wait or', 'start': 935.04, 'duration': 8.14}, {'text': 'the dark blue means more weight will add', 'start': 939.01, 'duration': 6.9}, {'text': 'more data points and see that the grid', 'start': 943.18, 'duration': 6.38}, {'text': 'kind of just lights up where we add data', 'start': 945.91, 'duration': 8.45}, {'text': 'and in the end you end up with this', 'start': 949.56, 'duration': 8.56}, {'text': 'beautiful grid equidistant grid and then', 'start': 954.36, 'duration': 6.19}, {'text': 'you can compute the 2d convolution you', 'start': 958.12, 'duration': 4.56}, {'text': 'sample the kernel in two dimensions and', 'start': 960.55, 'duration': 4.5}, {'text': 'then you convolve these two matrices and', 'start': 962.68, 'duration': 4.71}, {'text': "it's really fast and it works in higher", 'start': 965.05, 'duration': 5.61}, {'text': "dimensions but the speed-up is it's best", 'start': 967.39, 'duration': 5.28}, {'text': "in low dimensions obviously because it's", 'start': 970.66, 'duration': 4.11}, {'text': '2 to the power of D and the algorithm', 'start': 972.67, 'duration': 6.87}, {'text': "complexity okay I'd like to take one", 'start': 974.77, 'duration': 7.14}, {'text': 'minute to talk about my software', 'start': 979.54, 'duration': 5.49}, {'text': "implementation I've written a library in", 'start': 981.91, 'duration': 5.94}, {'text': 'Python called KDE Pi which is a very', 'start': 985.03, 'duration': 6.12}, {'text': "it's not a very inspired not named it's", 'start': 987.85, 'duration': 5.97}, {'text': "just a quick name but it's it's pretty", 'start': 991.15, 'duration': 6.75}, {'text': "fast and it's starting to become pretty", 'start': 993.82, 'duration': 6.03}, {'text': "good so if you'd like to experiment with", 'start': 997.9, 'duration': 3.66}, {'text': "it every graph and everything that I've", 'start': 999.85, 'duration': 3.63}, {'text': 'done in this presentation is made using', 'start': 1001.56, 'duration': 4.77}, {'text': "this library it's on github so please", 'start': 1003.48, 'duration': 4.83}, {'text': "have a look if you're interested in", 'start': 1006.33, 'duration': 4.26}, {'text': 'working with kernel density estimation', 'start': 1008.31, 'duration': 6.06}, {'text': "in Python if you'd like to read more", 'start': 1010.59, 'duration': 6.08}, {'text': 'about kernel density estimation you can', 'start': 1014.37, 'duration': 5.42}, {'text': 'look at the book by silverman and', 'start': 1016.67, 'duration': 6.15}, {'text': "there's another book by wand which is", 'start': 1019.79, 'duration': 5.1}, {'text': 'perhaps slightly more difficult to read', 'start': 1022.82, 'duration': 4.92}, {'text': 'a bit more recent and Drake Rondo plus', 'start': 1024.89, 'duration': 4.92}, {'text': 'has a blog post where he talks about', 'start': 1027.74, 'duration': 4.2}, {'text': 'kernel density estimation and Python', 'start': 1029.81, 'duration': 4.92}, {'text': "it's five years old but it's really good", 'start': 1031.94, 'duration': 5.61}, {'text': 'so actually just you take a look at that', 'start': 1034.73, 'duration': 4.8}, {'text': 'if you want to there are a few more', 'start': 1037.55, 'duration': 5.73}, {'text': 'references if you go to KDE PI and you', 'start': 1039.53, 'duration': 5.22}, {'text': "go to the documentation there's a", 'start': 1043.28, 'duration': 3.51}, {'text': 'literature overview or you can find more', 'start': 1044.75, 'duration': 3.3}, {'text': 'information about kernel density', 'start': 1046.79, 'duration': 4.26}, {'text': 'estimation so thanks a lot for watching', 'start': 1048.05, 'duration': 5.01}, {'text': 'this and I hope you learned something', 'start': 1051.05, 'duration': 4.29}, {'text': 'and I hope you have fun playing around', 'start': 1053.06, 'duration': 4.73}, {'text': 'with this', 'start': 1055.34, 'duration': 2.45}]
df = pd.DataFrame() #pd.read_csv('../../reports/csv/TextTiling_Freq_Vectors.csv')
video_id = 'x5zLaWT5KPs'
video_len = 1059

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

