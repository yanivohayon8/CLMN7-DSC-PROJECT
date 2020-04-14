# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 09:40:00 2020

@author: yaniv
"""

from abc import ABC



class SimilarityCalc(ABC):
    
    vectors = NotImplemented
    
    def calc(self):
        pass



class DirectSimilarity(SimilarityCalc):
    '''    
        This class is used to calculate the direct similarity between 2 vectors
    '''
    
    def __init__(self,vectors):
        self.vectors = vectors
        self.similarity_function = cosine_similarity# consider using other similarity function than cosine 
        
    # notice it is more valid of vectors like the text tiling looking for minima example
    def calc(self):
        return [self.similarity_function(self.vectors[i].reshape(1,-1),\
                                         self.vectors[i+1].reshape(1,-1))[0][0]\
                for i in range(0,len(self.vectors) - 1)]
        #pass
    
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage.filters import maximum_filter,median_filter,convolve
import numpy as np


class similarity():
    @staticmethod
    def calc_adjacent_matrix(vectors,method='cosine',word2vec_model=None):
        adjacent_matrix = None
        
        if method == 'cosine':
            adjacent_matrix = cosine_similarity(vectors,vectors)
        
        # make shure that vector method called for it is do_nothing
        if method == 'wmdistance':
            adjacent_matrix = np.ndarray(shape=(len(vectors),len(vectors)))
            print(len(vectors))
            print(adjacent_matrix.shape)
            for i in range(adjacent_matrix.shape[0]):
                for j in range(adjacent_matrix.shape[1]):
                    adjacent_matrix[i][j] = word2vec_model.wmdistance(vectors[i],vectors[j])
        return adjacent_matrix
    
    @staticmethod
    def apply_filter(similarity_matrix,filter_type='median',mask_size=(2,2)):
        mask = np.ones(mask_size)
        
        if filter_type == 'median':
            return median_filter(similarity_matrix,footprint=mask)
        
        return None