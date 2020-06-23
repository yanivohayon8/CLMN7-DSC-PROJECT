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
import os

import sys
#sys.path.append('..')

from src.models.audio import getSubjSilentRanges
#from data.collect_text_changes_time import get_time_titles_changed

from functools import reduce
from nltk.stem import PorterStemmer
porter = PorterStemmer()

class CreateBlocks:
    
    def preprocessing_text(self,raw_data,lemmatizing = "lemma"):
        stop_words = stopwords.words('english')
        nlp = spacy.load('en',disable=['parser','ner'])
        allowed_postags=['NOUN', 'ADJ', 'VERB','PROPN','ADV']#['NOUN', 'ADJ', 'VERB']#['NOUN', 'ADJ', 'VERB','ADV']#['NOUN', 'ADJ', 'VERB','PROPN']#['NOUN', 'ADJ', 'VERB']#
    
        processed_brth_grp = []
        for brth_grp in raw_data:
            # set lower case
            #brth_group_tokenized_lower = [w.lower() for w in brth_grp]
            #brth_group_text_no_punc = re.sub(r"[-()\"#/@;:<>{}`+=~|!?,.\n]", "", ' '.join(brth_group_lower)[:-1])
            brth_group_text_no_punc = simple_preprocess(' '.join(brth_grp),deacc=True) 
            tokenized_text_non_stop_words = [ word for word in brth_group_text_no_punc \
                                             if word not in stop_words]
            
            
            if lemmatizing == "stemm":
                tokenized_lemmas = [porter.stem(w) for w in tokenized_text_non_stop_words]
            else:
                text_non_stop_words = ' '.join(tokenized_text_non_stop_words)
                tokenized_lemmas = nlp(text_non_stop_words)
                tokenized_lemmas = [token.lemma_ for token in tokenized_lemmas \
                                    if token.pos_ in allowed_postags]
            '''text_non_stop_words = ' '.join(tokenized_text_non_stop_words)
            tokenized_lemmas = nlp(text_non_stop_words)
            tokenized_lemmas = [token.lemma_ for token in tokenized_lemmas \
                                if token.pos_ in allowed_postags]'''
            processed_brth_grp.append(tokenized_lemmas)
        
        all_tokenized = reduce(lambda acc,x: acc+x,processed_brth_grp)
        size =100
        tokenized_fixed_size = [all_tokenized[i:i+size] for i in range(0,len(all_tokenized) - size,size)]
        tokenized_fixed_size.append(all_tokenized[-(len(all_tokenized)%size):])
        
        #bigram = Phrases(tokenized_fixed_size, min_count=5, threshold=30) # higher threshold fewer phrases.
        bigram = Phrases(tokenized_fixed_size, min_count=10,threshold=0,scoring='npmi') # higher threshold fewer phrases.
        #bigram_mod = Phraser(bigram)
        #documents_bigrams = [bigram_mod[doc] for doc in tokenized_text_non_stop_words] 
        trigram = Phrases(bigram[tokenized_fixed_size],min_count=7,threshold=0,scoring='npmi')
        phrases_brth_grp = [trigram[bigram[doc]] for doc in tokenized_fixed_size]
        

        '''Return the corpus to the original breath group structure'''
        phrases = list(set(reduce(lambda acc,x: acc+x,
                         [[ph for ph in grp if '_' in ph] for grp in phrases_brth_grp])))
        brth_grp_prc_sizes = [len(brth_grp) for brth_grp in processed_brth_grp]        
        flat_phrases_brth_grp = list(reduce(lambda acc,x: acc+x,phrases_brth_grp))
        org_size_grp = []
        group_index = 0
        word_in_group_index = 0
        curr_group = []
        for flat_index in range(len(flat_phrases_brth_grp)):
            word = flat_phrases_brth_grp[flat_index]
            curr_group.append(word)
            phrase_size = 1 if word not in phrases else (word.count('_') + 1)
            word_in_group_index+=phrase_size
            
            # if we reached to the end of the breath group
            if word_in_group_index >= brth_grp_prc_sizes[group_index]:
                org_size_grp.append(curr_group)
                curr_group = []
                word_in_group_index = word_in_group_index - brth_grp_prc_sizes[group_index]
                group_index+=1
        
        
        return org_size_grp
        
        
    
    '''
        Divide the text - if window size is not specified, then the breath group remain as they are
    ''' 
    def __init__(self,transcripts_jsons,video_id="XXXXXX",lemmatizing = "lemma"):

        '''Find processed corpus'''
        #raw_data_tokenized = np.concatenate([brth['text'].split(' ') for brth in transcripts_jsons])
        #raw_text = ' '.join(raw_data_tokenized)[:-1]
        #self.tokenized_corpus = self.preprocessing_text(raw_text)
        
        raw_data_tokenized = [brth['text'].split(' ') for brth in transcripts_jsons]
        self.tokenized_trgrp_corpus = self.preprocessing_text(raw_data_tokenized,lemmatizing)
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
                        
            
        self.corpus_by_blocks = None
        self.word_ind_blocks = None
    '''
        partion to foregion groups
    '''
    '''def partion(self):
        return self.partion_by_sliding_windows(self.window_size,self.window_size)
    '''
    def partion(self,method="sliding_window",
                window_size=40,step_size_sd=20,
                silence_threshold=-30,slice_length=1000,step_size_audio=10,wav_file_path=None,
                video_path=None,wanted_frequency=15,wanted_similarity_percent = 75):
        if method == 'sliding_window':
            return self.partion_by_sliding_windows(window_size,step_size_sd)
        
        if method == 'audio':
            return self.partion_by_audio(silence_threshold,slice_length,step_size_audio,wav_file_path)
        
        if method == 'text_changing':
            return self.partion_by_slides_text_changes(video_path, wanted_frequency, wanted_similarity_percent)
        
        return None
    
    '''This function devides the video to chunks according to its silent parts'''
    def partion_by_audio(self,silence_threshold,slice_length,step_size_audio,wav_file_path):
        if os.path.isfile(wav_file_path) is False:
            raise Exception('path %s was not found. (For more details +972-54378975893)' % (wav_file_path))
        slices_seconds = getSubjSilentRanges(wav_file_path,silence_threshold,slice_length,step_size_audio)
        
        # removing silence parts after transcripts ends
        while slices_seconds[-1] > self.word_timestamp[-1]:
            slices_seconds.pop(-1)
            
        return self.partion_by_timestamp(slices_seconds)
    
    ''' This function devides the video to chunks according to text changes in the lecture slides of the video'''
    def partion_by_slides_text_changes(self, video_path, wanted_frequency, wanted_similarity_percent):
        if os.path.isfile(video_path) is False:
            raise Exception('path %s was not found. (For more details +972-54378975893)' % (video_path))
        slices_seconds = get_time_titles_changed(video_path = video_path, wanted_frequency = wanted_frequency ,
                                                 change_threshold = wanted_similarity_percent)
        
        # removing silence parts after transcripts ends
        while slices_seconds[-1] > self.word_timestamp[-1]:
            slices_seconds.pop(-1)
            
        return self.partion_by_timestamp(slices_seconds)
    
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
    
    def partion_by_timestamp(self,block_timestamps):
        corpus_divided = []
        indexes_diversion = []
        ts_index = 0
        word_index_pre = 0
        word_index_next = 0
        
        for ts_index in range(len(block_timestamps)):
            # get the next word index for the next timestamp
            while self.word_timestamp[word_index_next] - block_timestamps[ts_index] <= 0 :
                word_index_next+=1
            corpus_divided.append(self.tokenized_corpus[word_index_pre:word_index_next])
            indexes_diversion.append((word_index_pre,word_index_next))
            word_index_pre = word_index_next + 1
            pass
       
        # adding the rest
        corpus_divided.append(self.tokenized_corpus[word_index_next:])
        indexes_diversion.append((word_index_next,len(self.tokenized_corpus) - 1))
        
        self.word_ind_blocks = indexes_diversion    
        return corpus_divided#,indexes_diversion     