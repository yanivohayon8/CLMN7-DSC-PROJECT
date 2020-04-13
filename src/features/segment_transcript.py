# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 09:36:56 2020

@author: yaniv
"""

import functools
import numpy as np


import re
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import spacy
from gensim.models.phrases import Phrases, Phraser

from awlify import awlify
import json

class CreateBlocks:
    
    def preprocessing_text(self,raw_data):
        stop_words = stopwords.words('english')
        nlp = spacy.load('en',disable=['parser','ner'])
        allowed_postags=['NOUN', 'ADJ', 'VERB']
    
        processed_brth_grp = []
        for brth_grp in raw_data:
            # set lower case
            #brth_group_tokenized_lower = [w.lower() for w in brth_grp]
            #brth_group_text_no_punc = re.sub(r"[-()\"#/@;:<>{}`+=~|!?,.\n]", "", ' '.join(brth_group_lower)[:-1])
            brth_group_text_no_punc = simple_preprocess(' '.join(brth_grp),deacc=True) 
            tokenized_text_non_stop_words = [ word for word in brth_group_text_no_punc \
                                             if word not in stop_words]
            text_non_stop_words = ' '.join(tokenized_text_non_stop_words)
            tokenized_lemmas = nlp(text_non_stop_words)
            tokenized_lemmas = [token.lemma_ for token in tokenized_lemmas \
                                if token.pos_ in allowed_postags]
            processed_brth_grp.append(tokenized_lemmas)
        
        return processed_brth_grp
        
        
    
    '''
        Divide the text - if window size is not specified, then the breath group remain as they are
    ''' 
    def __init__(self,transcripts_jsons,window_size=20,n_breath_group_union=-1):

        '''Find processed corpus'''
        #raw_data_tokenized = np.concatenate([brth['text'].split(' ') for brth in transcripts_jsons])
        #raw_text = ' '.join(raw_data_tokenized)[:-1]
        #self.tokenized_corpus = self.preprocessing_text(raw_text)
        
        raw_data_tokenized = [brth['text'].split(' ') for brth in transcripts_jsons]
        self.tokenized_trgrp_corpus = self.preprocessing_text(raw_data_tokenized)
        self.tokenized_corpus = np.concatenate(self.tokenized_trgrp_corpus)
        
        '''
            Find the words timestamp
        '''
        self.word_timestamp = []
        for brth_index,brth in enumerate(transcripts_jsons):
            words = brth['text'].split(' ')
            avg_delay = round(brth['duration'] / len(words),4)
            brth_grp_timestamp = []
            for w in range(len(self.tokenized_trgrp_corpus[brth_index])):
                self.word_timestamp.append(round(brth['start'] + w * avg_delay,2))
                
        '''
            Find the block sizes
            
            Need to fix the if section of the n_breath_group_union to work on the processed data
        '''
        '''if n_breath_group_union != -1:
            self.block_sizes = [len(grp['text'].split(' ')) for grp in transcripts_jsons]
            blk_sizes = []
            if n_breath_group_union > 0:
                for blk_index in range(1,len(self.block_sizes) - n_breath_group_union,\
                                       n_breath_group_union):
                      blk_sizes.append(sum(self.block_sizes[blk_index - 1:blk_index + n_breath_group_union]))
                
                self.block_sizes = blk_sizes
        else:'''        
        
        self.window_size = window_size
        self.block_sizes = [window_size] * (int(len(self.tokenized_corpus) / window_size))
        self.block_sizes.append(len(self.tokenized_corpus) % window_size)
            
        if sum(self.block_sizes) != len(self.tokenized_corpus):
            print("sum of block_sizes : %s , number of words %s" % (sum(self.block_sizes),len(self.tokenized_corpus)))
            raise Exception("sum of block size is not as the number of words in the corpus")
            
        #print(self.block_sizes)
        
        self.corpus_by_blocks = None
        self.word_ind_blocks = None
    '''
        partion to foregion groups
    '''
    def partion(self):
        return self.partion_by_sliding_windows(self.window_size,self.window_size)
        '''corpus_divided = []
        indexes_diversion = []
        start_index = 1
        
        for w_index in range(1,len(self.tokenized_corpus) - self.window_size,self.window_size):
            start = start_index - 1
            end = start + self.window_size
            next_blk = self.tokenized_corpus[start:end]
            indexes_diversion.append((start,end))
            corpus_divided.append(next_blk)
            start_index += self.window_size + 1
        
        # adding the remaining (modulu)
        if start_index < len(self.tokenized_corpus) - 1:
            corpus_divided.append(self.tokenized_corpus[start_index:])
            indexes_diversion.append((start_index,len(self.tokenized_corpus) - 1))
        
        self.word_ind_blocks = indexes_diversion    
        return corpus_divided#,indexes_diversion'''
        
    
    '''This used to divide the text by a sliding window (unforegion groups)'''
    def partion_by_sliding_windows(self,window_size,step_size):
        corpus_divided = []
        indexes_diversion = []
        start_flag = 0
        for word_index in range(1,len(self.tokenized_corpus) - window_size,step_size):
            start = word_index - 1
            end = word_index+window_size
            start_flag = end + 1
            corpus_divided.append(self.tokenized_corpus[start:end])
            indexes_diversion.append((start,end))
            
        # adding the remaining (modulu)
        if start_flag < len(self.tokenized_corpus) - 1:
            corpus_divided.append(self.tokenized_corpus[start_flag:])
            indexes_diversion.append((start_flag,len(self.tokenized_corpus) - 1))
        
        self.word_ind_blocks = indexes_diversion    
        return corpus_divided#,indexes_diversion
    
    # notice fix that you will work on the processed data
    def get_block_timestamp(self):
        return [(self.word_timestamp[blk[0]],self.word_timestamp[blk[1]])\
                for blk in self.word_ind_blocks]
    
    def get_block_gap_timestamp(self):
        blk_timestamp = self.get_block_timestamp()
        time_stamp = []
        start_index = 0
        for blk_index in range(len(blk_timestamp) -1):
            prev_end = blk_timestamp[blk_index][1]
            succ_start = blk_timestamp[blk_index + 1][0]
            time_stamp.append((prev_end + succ_start)/2)
        return time_stamp