import ast
from src.models.pipeline import pipeline

class functionsBuilder():
    
    def __init__(self,groundbase=None,transcripts=None):
        self.groundbase = groundbase
        self.transcripts = transcripts
        self.filter_params = {'filter_type': None,'mask_shape': None,'sim_thresh':None,'is_min_thresh':True}

    def build_f_to_optimize(self,workflow_label):
        pipeline_steps_values = workflow_label.split('-')
        steps_labels = ['segment','vectorize','similarity','added_filter','clustering']
        _pipeline = {}
        for key,value in list(zip(steps_labels,pipeline_steps_values)):
            _pipeline[key]=value
        
        '''takingcare of filters'''
        
        # if we have image filter 
        if _pipeline['added_filter'] != 'None':
            vals = _pipeline['added_filter'].split('_')
            self.filter_params['filter_type'] = vals[0]
            self.filter_params['mask_shape'] = ast.literal_eval(vals[1])    
            
            
        '''Defiing the functions to optimized'''     
            
        if _pipeline['segment'] =='sliding_window' and \
           _pipeline['vectorize'] =='tfidf' and \
           _pipeline['similarity'] == 'cosine' and  \
           _pipeline['clustering'] == 'spectral_clustering':
               return self.f_sd_tfidf_cosine_sc(_pipeline)
           
        if _pipeline['segment'] =='audio' and \
           _pipeline['vectorize'] =='tfidf' and \
           _pipeline['similarity'] == 'cosine' and \
           _pipeline['clustering'] == 'spectral_clustering':
               return self.f_audio_tfidf_cosine_sc(_pipeline)
        
        raise("No workflow %s were implemented" %(workflow_label))
    
    def f_sd_tfidf_cosine_sc(self,_pipeline):
        #print(self.transcripts[0])
        def _f(window_size,step_size,sim_thresh,n_clusters):
            window_size = int(window_size)
            step_size = int(step_size)
            n_clusters = int(n_clusters)
            self.filter_params['sim_thresh'] = sim_thresh
    
            return pipeline.run_for_baye(self.groundbase,self.transcripts,
                                slicing_method=_pipeline['segment'],
                                window_size=window_size,step_size_sd=step_size,
                                vector_method=_pipeline['vectorize'],
                                similarity_method=_pipeline['similarity'],
                                filter_params=self.filter_params,
                                clustering_params={'algorithm': _pipeline['clustering'],
                                                   'n_clusters':n_clusters},
                                accurrcy_shift=30
                               )
    
        return _f
    
    
    def f_audio_tfidf_cosine_sc(self,_pipeline):
        #print(self.transcripts[0])
        def _f(silence_threshold,slice_length,step_size_audio,sim_thresh,n_clusters):
                silence_threshold = int(silence_threshold)
                slice_length = int(slice_length)
                step_size_audio = int(step_size_audio)
                n_clusters = int(n_clusters)
                self.filter_params['sim_thresh'] = sim_thresh
    
                return pipeline.run_for_baye(self.groundbase,self.transcripts,
                                    slicing_method=_pipeline['segment'],
                                    silence_threshold=silence_threshold,slice_length=slice_length,
                                    step_size_audio=step_size_audio, 
                                    vector_method=_pipeline['vectorize'],
                                    similarity_method=_pipeline['similarity'],
                                    filter_params=self.filter_params,
                                    clustering_params={'algorithm': _pipeline['clustering'],
                                                       'n_clusters':n_clusters},
                                    accurrcy_shift=30
                                   )
    
        return _f
            