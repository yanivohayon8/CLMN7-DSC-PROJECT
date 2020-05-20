# -*- coding: utf-8 -*-
"""
Created on Mon May 18 08:23:28 2020

@author: yaniv
"""



from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import re

from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import spacy
from gensim.models.phrases import Phrases, Phraser
import gensim

def read_pdf_raw(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'  # 'utf16','utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    #""    C:\\Users\\yaniv\\Downloads\\The_Apriori_Algorithm-a_Tutorial.pdf
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,
                                  password=password, caching=caching, check_extractable=True):
        interpreter.process_page(page)
    fp.close()
    device.close()
    str_ = retstr.getvalue()
    retstr.close()
    return str_


def get_chapters_titles(pdf_text_raw):
    p = re.compile(r'([\n]{2,2}(\d\.)+\d* [A-Za-z0-9? ]+[\n]{1,2})')
    all_iter = p.finditer(pdf_text_raw)
    titles = p.findall(pdf_text_raw)
    titles_clean = [re.sub(r'(\d\.)+\d*', '', str(tl[0])) for tl in titles]#[re.sub('\W+','', tl) for tl in titles]
    titles_clean = [re.sub('\\n', '', str(tl)) for tl in titles_clean]
    indexes = [m.span() for m in all_iter]
    return titles_clean,indexes
    #return titles,indexes

def get_chapters_content(pdf_text_raw,start_end_sections_titles):
    stop_words = stopwords.words('english')
    nlp = spacy.load('en',disable=['parser','ner'])
    allowed_postags=['NOUN', 'ADJ', 'VERB']
    
    
    start_section = 0
    section_text = []
    for range_ in start_end_sections_titles:
        section_text.append(pdf_text_raw[start_section:range_[0]])
        start_section = range_[1]
        
    section_text.append(pdf_text_raw[start_section:])
    section_text = section_text[1:]
    
    corpus = []
    for sc_text in section_text:
        documents = []
        doc_tokened = sc_text.split(".")
        for index,doc in enumerate(doc_tokened):
            doc_text_no_punc = simple_preprocess(doc,deacc=True) 
            tokenized_text_non_stop_words = [ word for word in doc_text_no_punc \
                                             if word not in stop_words]
            text_non_stop_words = ' '.join(tokenized_text_non_stop_words)
            tokenized_lemmas = nlp(text_non_stop_words)
            tokenized_lemmas = [token.lemma_ for token in tokenized_lemmas \
                                if token.pos_ in allowed_postags]
            documents.append(tokenized_lemmas)
        corpus.append(list(filter(lambda x: len(x) >= 2, documents)))
    return corpus