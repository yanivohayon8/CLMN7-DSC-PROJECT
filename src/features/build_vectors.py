# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 09:39:02 2020

@author: yaniv
"""


#word2vec_wiki_model = api.load('glove-wiki-gigaword-300')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
import gensim.corpora as corpora
import gensim


import nltk
from nltk.corpus import wordnet

class vectorizer():
    
    
    @staticmethod
    def find_word_vect(w,word2vec_model):
        try:
            #print(w)
            return word2vec_model[w]
        except:
            #print("in the exception")
            zero_vector = np.zeros(word2vec_model['world'].shape)
            #print(zero_vector)
            syns = wordnet.synsets(w) 
            #print(syns)
            for syn in syns:
                for l in syn.lemmas():
                    try:
                        return word2vec_model[l.name()]
                    except:
                        pass
            return zero_vector
    
    @staticmethod
    def calc(blocks,method='tfidf',word2vec_model=None,vectorizing_params =None):
        
        myvectorizer = None
        if method == 'tfidf':
            con_blocks = [' '.join(blk) for blk in blocks]
            myvectorizer = TfidfVectorizer()
            return myvectorizer.fit_transform(con_blocks).toarray()        
        
        if method == 'tf':
            con_blocks = [' '.join(blk) for blk in blocks]
            myvectorizer = CountVectorizer()
            return myvectorizer.fit_transform(con_blocks).toarray()        
        
        if method == 'word2vec_naive_mean':
            word_mat = []
            for blk in blocks:
                block_as_wordvector_shape = (blk.size,word2vec_model['world'].size)
                word_vectors_blocks = np.empty(block_as_wordvector_shape)
                for word_index in range(blk.size):
                    word_vectors_blocks[word_index] = vectorizer.find_word_vect(blk[word_index], word2vec_model)
                word_mat.append(np.sum(word_vectors_blocks,axis=0)/blk.size)
            
            return word_mat
        
        if method == 'do_nothing':
            con_blocks = [' '.join(blk) for blk in blocks]
            return con_blocks
        
        if method == 'doc2vec':
            '''MAKE THIS NOT HARD CODED'''
            #model = gensim.models.doc2vec.Doc2Vec.load('../models/doc2vec/PL2B42F74062A70327')
            model = gensim.models.doc2vec.Doc2Vec.load('../../models/doc2vec/PL2B42F74062A70327')
            '''Calculate the cosine similarity between vectors - make a lot of noise see results from 07/05 '''
            #return [model.infer_vector(blk) for blk in blocks]
            
            '''Ranking most similiar'''
            ranks = []
            for doc_id_1,blk_1 in enumerate(blocks):
                inferred_vector = model.infer_vector(blk_1).reshape(1,-1)
                blk_sim = [(doc_id_2,
                            cosine_similarity(inferred_vector,
                                              model.infer_vector(blk_2).reshape(1,-1))[0][0]
                            ) \
                              for doc_id_2,blk_2 in enumerate(blocks)]
                blk_sim = sorted(blk_sim,key=lambda x:x[1])    
                ranking = np.array([b_sim[0] for b_sim in blk_sim])
                ranks.append(ranking)
            return np.array(ranks)
        
        if method == 'lda':
            id2word = corpora.Dictionary(blocks)
            corpus = [id2word.doc2bow(doc) for doc in blocks]
            
            '''You need to train the LDA model you need to make here some loops and stabilizing params'''
            # low alpha means each document is only represented by a small number of topics, and vice versa
            # low eta means each topic is only represented by a small number of words, and vice versa
            # https://www.kaggle.com/ktattan/lda-and-document-similarity
            lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                  num_topics=vectorizing_params['n_clusters'],
                                                  id2word=id2word#,
                                                  #alpha=vectorizing_params['alpha']#, # consider this to be auto
                                                  #eta=vectorizing_params['eta'],
                                                  #chunksize=vectorizing_params['chunksize'],
                                                  #minimum_probability=vectorizing_params['minimum_probability'],
                                                  #passes=vectorizing_params['passes']
                                                  )
            
            return np.array([np.array([doc_dis[1] for doc_dis in lda.get_document_topics(bow=cp)]) for cp in corpus] )
            
        
        raise("No vector method %s was found " %(method))
    
    