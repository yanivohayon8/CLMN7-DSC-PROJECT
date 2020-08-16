# -*- coding: utf-8 -*-

# Extract the chapters as single documents

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

from gensim.models.phrases import Phrases, Phraser
from gensim import corpora
from numpy import setdiff1d
from nltk.stem import PorterStemmer

stop_words = stopwords.words('english')
nlp = spacy.load('en', disable=['parser', 'ner'])
allowed_postags = ['NOUN', 'ADJ', 'VERB', 'PROPN',
                   'ADV']  # ['NOUN', 'ADJ', 'VERB']#['NOUN', 'ADJ', 'VERB','ADV']#['NOUN', 'ADJ', 'VERB','PROPN']#['NOUN', 'ADJ', 'VERB']#['NOUN', 'ADJ', 'VERB','PROPN','ADV']#
porter = PorterStemmer()


def read_docx(doc_path):
    word_document = Document(doc_path)
    p_sent = re.compile('\w+')
    full_text = []
    font_sizes = []
    for para in word_document.paragraphs:
        for i, run in enumerate(para.runs):
            matching = p_sent.match(run.text)

            # see if there is a number or letter ( it is not a trash)
            if matching is not None and run.text is not None and run.font.size is not None:
                full_text.append(run.text)  # re.sub('[\t\n\b\r]','',
                font_sizes.append(run.font.size)
    return full_text, font_sizes


def process_docx(total_corpus, lemmatizing="lemma"):
    total_corpus_tokenized = []

    documents = []
    document_chapters_indexes = []
    for chapter_index, chapter_corpus in enumerate(total_corpus):

        sentences = list(filter(lambda x: len(x) > 0, chapter_corpus.split('.')))
        for sent in sentences:
            doc_text_no_punc = simple_preprocess(sent, deacc=True)
            tokenized_text_non_stop_words = [word for word in doc_text_no_punc \
                                             if word not in stop_words]

            if lemmatizing == "stemm":
                tokenized_lemmas = [porter.stem(w) for w in tokenized_text_non_stop_words]
            else:
                text_non_stop_words = ' '.join(tokenized_text_non_stop_words)
                tokenized_lemmas = nlp(text_non_stop_words)
                tokenized_lemmas = [token.lemma_ for token in tokenized_lemmas \
                                    if token.pos_ in allowed_postags]

            documents.append(tokenized_lemmas)
            document_chapters_indexes.append(chapter_index)

        ### if we are looking for sentences ending and we do care of words like fig. 14.
        '''
        start_index = 0
        #for m in re.finditer('[a-zA-Z]\.',chapter_corpus):
        for m in re.finditer('[a-zA-Z ]\.',chapter_corpus):
            chapter_tokenized_doc = chapter_corpus[start_index:m.start(0) + 1]

            doc_text_no_punc = simple_preprocess(chapter_tokenized_doc,deacc=True) 
            tokenized_text_non_stop_words = [ word for word in doc_text_no_punc \
                                             if word not in stop_words]

            text_non_stop_words = ' '.join(tokenized_text_non_stop_words)
            tokenized_lemmas = nlp(text_non_stop_words)
            tokenized_lemmas = [token.lemma_ for token in tokenized_lemmas \
                                if token.pos_ in allowed_postags]
            documents.append(tokenized_lemmas)
            document_chapters_indexes.append(chapter_index)
            start_index = m.end(0) + 1'''

        # total_corpus_tokenized.append(documents)
    # total_corpus_tokenized = [ [doc for doc in ch if len(doc) > 0 ] for ch in total_corpus_tokenized]

    '''Form bigram and trigram'''
    bigram = Phrases(documents, min_count=5, threshold=50)  # higher threshold fewer phrases.
    # bigram_mod = Phraser(bigram)
    # documents_bigrams = [bigram_mod[doc] for doc in tokenized_text_non_stop_words]
    trigram = Phrases(bigram[documents], threshold=3)
    trigram_documents = [trigram[bigram[doc]] for doc in documents]

    '''Insert the documents into chapters'''
    total_corpus_tokenized = []
    curr_ch_index = 0
    chapter_docs = []
    debug_indexes = []
    problematic_ch = setdiff1d(range(len(total_corpus)), document_chapters_indexes)
    for ch_index, doc in zip(document_chapters_indexes, trigram_documents):
        # if it is the same chapter
        if curr_ch_index == ch_index:
            chapter_docs.append(doc)
        else:
            # for some reason the documents of this chapter are not considered
            while ch_index - curr_ch_index > 1:
                debug_indexes.append(curr_ch_index)
                total_corpus_tokenized.append([])
                chapter_docs = []
                curr_ch_index = curr_ch_index + 1  # ch_index
            else:
                # if it is a new chapter
                # then add the previous to the total pool start a new chapter collection
                debug_indexes.append(curr_ch_index)
                total_corpus_tokenized.append(chapter_docs)
                curr_ch_index = curr_ch_index + 1  # ch_index
                chapter_docs = [doc]

    # adding the last chapter
    debug_indexes.append(curr_ch_index)
    total_corpus_tokenized.append(chapter_docs)

    # total_corpus_tokenized = [[doc for doc in ch if len(doc) > 0 ] for ch in total_corpus_tokenized]
    total_corpus_tokenized = [[doc for doc in ch if len(doc) > 0] for ch in total_corpus_tokenized]
    return total_corpus_tokenized

    #    print(documents)
    #    break


def find_content(structure_pattern, full_text, font_sizes,
                 main_chapter_keyword='Chapter', lemmatizing="lemma"):
    return find_content_MIT(full_text, font_sizes, lemmatizing)


def find_content_MIT(full_text, font_sizes, lemmatizing="lemma"):
    '''########################'''
    '''       find titles      '''
    '''########################'''

    p_subsection_numbering = re.compile(r'((\d+\.)+\d*\t[A-Za-z0-9? ]+)')
    p_subsection_numbering_2 = re.compile(r'((\d+\.)+\d*)')

    main_titles = []
    total_titles = []
    sub_titles = []
    main_titles_index = []
    sub_titles_index = []

    for i, txt in enumerate(full_text):

        '''If it is a main chapter title'''
        if re.match('[a-zA-Z]{4,}', txt) is not None and font_sizes[i] == 228600:
            total_titles.append(txt)
            main_titles.append(txt)
            main_titles_index.append(i)
        else:
            str_print = None
            ''' if the sub title is from the form 1.1.1 bla'''
            if p_subsection_numbering.match(full_text[i]) is not None:
                str_print = txt
            else:
                '''if the title is not in the section element in full_text array'''
                if p_subsection_numbering_2.match(full_text[i]) is not None:
                    '''satisfing only in the section numbering'''
                    str_print = txt
                    ''' if it from the form 1.1.1 and next element the title value'''
                    if '.' not in full_text[i + 1] and \
                            ',' not in full_text[i + 1] and \
                            len(full_text[i + 1]) < 50 and \
                            re.match('[A-Za-z]{2,}', full_text[i + 1]) is not None:
                        str_print = ("%s %s" % (txt, full_text[i + 1]))
            if str_print is not None and (font_sizes[i] == 152400 or font_sizes[i] == 184150):
                # print("%s %d " %(str_print,font_sizes[i]))
                total_titles.append(str_print)
                sub_titles.append(str_print)
                sub_titles_index.append(i)

    total_titles = total_titles[1:-1]
    main_titles = main_titles[1:]
    main_titles_index = main_titles_index[1:]
    sub_titles = sub_titles[:-1]
    sub_titles_index = sub_titles_index[:-1]
    total_titles_index = main_titles_index + sub_titles_index
    total_titles_index = sorted(total_titles_index)

    '''########################'''
    '''       find corpus      '''
    '''########################'''
    total_corpus = []
    for i in range(len(total_titles_index)):
        next_pointer = len(full_text) - 1
        if i < len(total_titles_index) - 1:
            next_pointer = total_titles_index[i + 1]
        sec_as_text = reduce(lambda acc, x: acc + " " + x,
                             full_text[total_titles_index[i]:next_pointer], "")
        if sec_as_text is not None:
            sec_as_text = sec_as_text + '.'
            total_corpus.append(sec_as_text)

    total_corpus = process_docx(total_corpus, lemmatizing=lemmatizing)

    return {'corpus': total_corpus, 'titles': total_titles, 'main titles': main_titles}

# total_corpus +=reducefull_text[total_titles_index[-1]:]







