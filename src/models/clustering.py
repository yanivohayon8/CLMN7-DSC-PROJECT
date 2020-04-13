# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 09:40:23 2020

@author: yaniv
"""

class Clustering():
    pass

from numpy import diff,sign
class TextTilingClustering():
    
    '''
        used to find clusters by local minima
    '''
    @staticmethod
    def find_cluster(similarities_array,gap_times):
        minimas = (diff(sign(diff(similarities_array))) > 0).nonzero()[0] + 1
        #print([gap_times[min] for min in minimas])
        results = []
        #print ("%s topics were found" % (len(minimas)))
        for min_index in minimas:
            results.append(int(gap_times[min_index]))
            #print("%s:%s or %s seconds" % (int(round(gap_times[min_index]/60)),gap_times[min_index]%60,gap_times[min_index]))
        return results
    
    
            
            
from sklearn.cluster import SpectralClustering            
class clustering():
    '''
        compare the shifts of topic from our results compared to the ground base
    '''
    @staticmethod
    def boundryevaluation(curresults,ground_base,accurrcy_shift = 15):
        if len(curresults) != len(ground_base):
            print("the number of the topic in the given results is not the same as the number of topics in the ground base:")
            print("results: %s , ground base: %s" %(len(curresults),len(ground_base)))
        true_positive = 0
        false_negative = 0
        false_positive = 0 
        
        print("The results:")
        print(curresults)
        print("The ground base:")
        print(ground_base)
        
        true_positive_list_debug = []
        #false_negative_list_debug = []
        false_positive_list_debug = []
        
        for grb in ground_base:
            is_false_positive = True
            for res in curresults:
                if abs(res - grb) < accurrcy_shift:
                    is_false_positive = False
                    true_positive_list_debug.append(res)
                    true_positive+=1
                    curresults.remove(res)
                    break   
            if is_false_positive:
                false_positive_list_debug.append(grb)
                false_positive+=1
                
        print("TP: " + str(true_positive_list_debug))
        print("FP: " + str(false_positive_list_debug))
        print("FN: " + str(curresults))
        
        false_negative = len(curresults) # make sure the true positive are removed from here 
        recall_rate = true_positive/(true_positive + false_negative )
        precision_rate = true_positive/(true_positive + false_positive)
        print("TP: %s , FP: %s, FN: %s" %(true_positive,false_positive,false_negative))
        print("precision rate : %s, recall rate : %s " % (precision_rate,recall_rate))
        return (recall_rate,precision_rate,true_positive,false_positive,false_negative)
        
    '''
        Search for change label after running spectral clustering
    '''
    @staticmethod
    def boundery_change(block_labeles,block_gap_times):
        results = []
        labeled_checked = []
        for blk_index in range(len(block_labeles) - 1):
            if block_labeles[blk_index] != block_labeles[blk_index + 1]:
                if block_labeles[blk_index] in labeled_checked:
                    raise Exception("label %s has done already cannot evaluate unsequentional segmentation, try other parameters" % (block_labeles[blk_index]))
                else:
                    labeled_checked.append(block_labeles[blk_index])
                    results.append(int(block_gap_times[blk_index]))
                    print("Change label on time %s:%s from %s to %s" % \
                          (int(block_gap_times[blk_index]/60),\
                           int(block_gap_times[blk_index]%60),block_labeles[blk_index],\
                          block_labeles[blk_index + 1]))
        #print(results)
        return results
    
    
    @staticmethod
    def label_block_timestamp(block_labeles,block_times):
        labels = np.unique(np.array(block_labeles))
        print(len(block_times))
        print(len(block_labeles))
        #start_times = [_time[0] for _time in block_times]
        #end_times = [_time[1] for _time in block_times]
        times = {}
        block_labeles = list(block_labeles)
        for lb in labels:
            first_index = block_labeles.index(lb)
            last_index = len(block_labeles) - 1 - block_labeles[::-1].index(lb)
            start_time = block_times[first_index][0]
            end_time = block_times[last_index][1]
            print("label :%s , starts : %s , ends: %s" %(str(lb),str(start_time),str(end_time)))
            times[lb] = (start_time,end_time)
            
    @staticmethod
    def run(adjacent_matrix,n_clusters,gap_timestamp,groundbase,algorithm='spectral_clustering',accurrcy_shift=15):
        clustering_results = None
        if algorithm == 'spectral_clustering':
            clustering_results = SpectralClustering(n_clusters=n_clusters).fit(adjacent_matrix)
        else:
            raise Exception('You should implement another clustering algorithm execution')
        
        print ('This are the labels from the clustering result:')
        print(clustering_results.labels_)
        
        myresults = clustering.boundery_change(clustering_results.labels_,gap_timestamp)
        recall,precision,tp,fp,fn = clustering.boundryevaluation(myresults,groundbase,accurrcy_shift)
        return recall,precision,tp,fp,fn 
        