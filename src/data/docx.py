# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:53:14 2020

@author: yaniv
"""
from docx import *
import re
from statistics import mode
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import spacy
from functools import reduce


stop_words = stopwords.words('english')
nlp = spacy.load('en',disable=['parser','ner'])
allowed_postags=['NOUN', 'ADJ', 'VERB']



def read_docx(doc_path):
    word_document = Document(doc_path)
    p_sent = re.compile('\w+')
    full_text = []
    font_sizes = []
    for para in word_document.paragraphs:
        for i,run in enumerate(para.runs):        
            matching = p_sent.match(run.text)
            
            # see if there is a number or letter ( it is not a trash)
            if matching is not None and run.text is not None and run.font.size is not None:
                    full_text.append(run.text)#re.sub('[\t\n\b\r]','',
                    font_sizes.append(run.font.size)
    return full_text,font_sizes


def process_docx(total_corpus):
    total_corpus_tokenized =[]

    for chapter_corpus in total_corpus:
        start_index = 0
        documents = []
        for m in re.finditer('[a-zA-Z]\.',chapter_corpus):
            chapter_tokenized_doc = chapter_corpus[start_index:m.start(0) + 1]
            
            # debug 
            #if len(chapter_tokenized_doc) == 0:
            #    print("my debug")
            
            doc_text_no_punc = simple_preprocess(chapter_tokenized_doc,deacc=True) 
            tokenized_text_non_stop_words = [ word for word in doc_text_no_punc \
                                             if word not in stop_words]
            text_non_stop_words = ' '.join(tokenized_text_non_stop_words)
            tokenized_lemmas = nlp(text_non_stop_words)
            tokenized_lemmas = [token.lemma_ for token in tokenized_lemmas \
                                if token.pos_ in allowed_postags]
            documents.append(tokenized_lemmas)
            start_index = m.end(0) + 1
        total_corpus_tokenized.append(documents)
    total_corpus_tokenized = [ [doc for doc in ch if len(doc) > 0 ] for ch in total_corpus_tokenized]
    return total_corpus_tokenized
        
    #    print(documents)
    #    break



def find_content(structure_pattern,full_text,font_sizes,main_chapter_keyword = 'Chapter'):
    if structure_pattern == 'statbook':
        return find_content_statbook(full_text,font_sizes,main_chapter_keyword)
    if structure_pattern == 'Dsa':
        return find_content_Dsa(full_text,font_sizes,main_chapter_keyword)
    raise('No structure pattern %s was identified with function' %(structure_pattern))



def find_content_Dsa(full_text,font_sizes,main_chapter_keyword = 'Chapter'):    
    most_common_font_size = mode(font_sizes)
    topic_titles = []
    main_chapter_titles = []
    p_Start_Section = re.compile('{} \d?'.format(main_chapter_keyword))
    p_subsection_numbering = re.compile(r'((\d+\.)+\d*\t[A-Za-z0-9? ]+)')
    p_words = re.compile('[A-Za-z]+')
    
    main_chapter_indexes = [i for i in range(len(full_text))\
                            if p_Start_Section.match(full_text[i]) is not None and font_sizes[i] > most_common_font_size]
    sub_chapter_indexes  = [i for i in range(len(full_text))\
                            if p_subsection_numbering.match(full_text[i]) is not None and font_sizes[i] > most_common_font_size]
    all_titles_indexes = main_chapter_indexes + sub_chapter_indexes    
    all_titles_indexes.sort()
    
    '''total_corpus = [reduce(lambda acc,x: acc+" " +x,
                           full_text[all_titles_indexes[i] + 1:all_titles_indexes[i+1]],"")\
                    for i in range(len(all_titles_indexes) - 1)]'''
    total_corpus  = []
    for i in range(len(all_titles_indexes) - 1):
        sec_as_text = reduce(lambda acc,x: acc+" " +x  if font_sizes[full_text.index(x)] == most_common_font_size else acc,
                             full_text[all_titles_indexes[i]:all_titles_indexes[i+1]],"")
        # adding dot for finishing the corpus
        if len(sec_as_text) == 0:
            pass
            #sec_as_text = 'section empty.' # planting plain text for non empty
        elif sec_as_text[-1] != '.':
            sec_as_text = sec_as_text + '.'
        total_corpus.append(sec_as_text)
        
    
    # adding the text of the last section
    last_sec_text = ""
    i = all_titles_indexes[-1] + 1
    while i < len(full_text) and font_sizes[i] <= most_common_font_size:
        last_sec_text = last_sec_text + " " + full_text[i]
        i+=1
    total_corpus.append(last_sec_text)
    
    
    # fill empty subssections with random string
     
    total_corpus = process_docx(total_corpus)
    
    for i in range(len(total_corpus)):
        if len(total_corpus[i]) == 0:
            total_corpus[i] = [['section','empty']]
    
    '''Finding titles values'''
    topic_titles  = []
    for j in all_titles_indexes:
        if j not in main_chapter_indexes:
           topic_titles.append(full_text[j])     
        else:
            topic_titles.append(full_text[j+1])

    main_chapter_titles = [full_text[j + 1] for j in main_chapter_indexes]
    
    
    return {'corpus':total_corpus,'titles':topic_titles,'main titles':main_chapter_titles}



def find_content_statbook(full_text,font_sizes,main_chapter_keyword = 'Topic'):
    most_common_font_size = mode(font_sizes)
    topic_titles = []
    main_chapter_titles = []
    p_Start_Section = re.compile('{} \d?'.format(main_chapter_keyword))
    p_subsection_numbering = re.compile(r'^((\d+\.)+\d*)$')
    p_words = re.compile('[A-Za-z]+')
    total_corpus = []
    chapter_corpus = ""
    
    for i,(doc_text,doc_size) in enumerate(zip(full_text,font_sizes)):
        ''' if it is a header '''
        #print(doc_size)
        #print(most_common_font_size)
        if(doc_size > most_common_font_size):
            is_title = False
            matching_start_chapter = p_Start_Section.match(doc_text)       
            # if it is start of new chapter\main topic
            if matching_start_chapter is not None:
                topic_titles.append(full_text[i+1])
                #print("main chapter %s size %s " %(full_text[i+1],font_sizes[i+1]))
                main_chapter_titles.append(full_text[i+1])
                is_title = True
            else:
                # if it is a start of sub topic
                '''comes from the structure in the input array as:
                [
                    1.1
                    Types of Data
                ]
                
                '''
                matching_start_section_numbering = p_subsection_numbering.match(doc_text)    
                matching_words = p_words.match(full_text[i+1])
                if matching_start_section_numbering is not None and matching_words is not None:
                    
                    # the subsection annotation (15.1 etc)
                    # must be at the same size as the label (Introduction etc)
                    if most_common_font_size < font_sizes[i+1]:
                        sub_section = ("%s %s" %(full_text[i],full_text[i+1]))
                        #print("subsection %s size %s %s" %(sub_section,font_sizes[i],font_sizes[i+1]))
                    else:
                        sub_section = full_text[i]
                        #print("subsection %s size %s" %(sub_section,font_sizes[i]))
                    is_title = True
                    
                    topic_titles.append(sub_section)
            if is_title:
                total_corpus.append(chapter_corpus[1:])
                chapter_corpus = ""
        
        elif doc_size == most_common_font_size:
            chapter_corpus = chapter_corpus + ' ' + doc_text
    
  
    total_corpus = process_docx(total_corpus)
    return {'corpus':total_corpus,'titles':topic_titles,'main titles':main_chapter_titles}
    


