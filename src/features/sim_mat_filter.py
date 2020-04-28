# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:00:55 2020

@author: yaniv
"""
from scipy.ndimage.filters import maximum_filter,median_filter,convolve,gaussian_laplace
import numpy as np

class similarityFilters():
    
    @staticmethod
    def similarity_filter(similarity_matrix,params):
        
        if params is None:
            return similarity_matrix
        
        filter_type = params['filter_type']
        mask_shape = params['mask_shape']
        sim_thresh = params['sim_thresh']
        is_min_thresh = params['is_min_thresh']
        
        '''Applying threshold filtering'''
        if sim_thresh is not None:
            if is_min_thresh:
                # make shure if 0 is o'right!
                similarity_matrix = np.asarray([0 if sim < sim_thresh else sim \
                                    for sim in np.concatenate(similarity_matrix.reshape(-1,1))]).reshape(similarity_matrix.shape)
            else: 
                similarity_matrix = np.asarray([100 if sim > sim_thresh else sim \
                                    for sim in np.concatenate(similarity_matrix.reshape(-1,1))]).reshape(similarity_matrix.shape)

        ''' Applying image filtering'''      
        if filter_type is not None:
            if filter_type == 'median':
                mask = np.ones(mask_shape)
                return median_filter(similarity_matrix,footprint=mask)
            
            if filter_type == 'gaussian_laplace':
                return gaussian_laplace(similarity_matrix,sigma=mask_shape)

        
        return similarity_matrix