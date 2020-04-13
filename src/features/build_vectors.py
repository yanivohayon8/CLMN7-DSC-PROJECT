# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 09:39:02 2020

@author: yaniv
"""

from abc import ABC
#import gensim.corpora as corpora
#import gensim
import numpy as np



class CalculateVectors(ABC):
    '''
        2.	Calculate vector 
            a.	Tf-idf , TF, Word embedding 
            b.	Make the calculation across all text (like word embedding) or
                just within the block or adjacent block (what ever you would like)
                Choose sliding window – fixed size or documents
    '''
    
    blocks = NotImplemented
    
    def calc(self):
        pass


    
'''
    consider aggregate the 2 classes into single class
'''    
    
from sklearn.feature_extraction.text import CountVectorizer
class TFCalculator(CalculateVectors):
    def __init__(self,blocks): 
        self.blocks = blocks
        self.vectorizer = CountVectorizer()
        
    
    # each vector is n word in all of the vocabulary
    def calc(self):
        #print(self.blocks)
        con_blocks = [' '.join(blk) for blk in self.blocks]
        #print(con_blocks)
        return self.vectorizer.fit_transform(con_blocks).toarray()
        
from sklearn.feature_extraction.text import TfidfVectorizer
class TFIDFCalculator(CalculateVectors):
    def __init__(self,blocks):
        self.blocks = blocks
        self.vectorizer = TfidfVectorizer()
    
    def calc(self):
        con_blocks = [' '.join(blk) for blk in self.blocks]
        return self.vectorizer.fit_transform(con_blocks).toarray()




class vectorizer():
    @staticmethod
    def calc(blocks,method='tfidf'):
        con_blocks = [' '.join(blk) for blk in blocks]
        
        myvectorizer = None
        if method == 'tfidf':
            myvectorizer = TfidfVectorizer()
        
        if method == 'tf':
            myvectorizer = CountVectorizer()
        
        return myvectorizer.fit_transform(con_blocks).toarray()
