# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import json
import os
import re

from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import spacy
from gensim.models.phrases import Phrases, Phraser

def set_as_continous(video_id,channel_dir):
   text_json = []
   video_file_path = os.path.join(channel_dir,str(video_id + '.json'))
   with open(video_file_path,'r') as f:
           text_json  = json.load(f)
   text = ""
   for curr_json in text_json:
       text = text + " " + curr_json['text']
        
   return text

@click.command()
@click.argument('input_channel')
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path(),)
#def main(input_filepath, output_filepath):
def main(input_channel): #="nptelhrd"
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('making Intermidiate data set from raw data')
    #spacy_nlp = spacy.load('en_core_web_sm')
    stop_words = stopwords.words('english')
        
    channel_dir = os.path.join(raw_dir,"transcripts",input_channel)
    videos_transcript_file = os.listdir(channel_dir)
    channel_intermid_dir = os.path.join(interim_dir,input_channel)
    
    if not os.path.exists(channel_intermid_dir):
        os.mkdir(channel_intermid_dir)
    else:
        print("\tThe folder of the channel %s is already exist\n"
                "\tdelete it before executing this script -" 
                "we don't want to override your data" %(channel_intermid_dir))
        return

    
    for file in videos_transcript_file:
        channel_intermid_file = os.path.join(channel_intermid_dir,file)
        video_id = file.split('.')[0]
        text = set_as_continous(video_id,channel_dir)
        
        '''
         Processing the data
        '''
        print('Set the text as lower cases')
        text = text.lower()
        document_text = text.split('.')
        print('Remove special character')
        document_text =list(map(lambda d: re.sub(r"[-()\"#/@;:<>{}`+=~|!?,\n]", " ", d),document_text)) #
        print("Tokinzing each sentence")
        tokenized_documents = list(map(lambda doc: simple_preprocess(doc,deacc=True),document_text))
        print("Remove stop words")
        tokenized_documents_non_stop_words = [[word for word in doc if word not in stop_words] for doc in tokenized_documents]
        
        # form Bigrams - make sure u read the full documantation to utilize the framework as should be!
        # Save the bigram model to a picke file
        print("Form Bigrams")
        bigram = Phrases(tokenized_documents, min_count=5, threshold=100) # higher threshold fewer phrases.
        #trigram = Phrases(bigram[tokenized_documents], threshold=100)
        bigram_mod = Phraser(bigram)
        #trigram_mod = Phraser(trigram)
        documents_bigrams = [bigram_mod[doc] for doc in tokenized_documents_non_stop_words] 

        print("Form lemmatization")
        nlp = spacy.load('en',disable=['parser','ner'])
        documents_lemmatized = []
        allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
        for sent in documents_bigrams:
            doc = nlp(" ".join(sent))
            current_doc = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
            if len(current_doc) > 0:
                documents_lemmatized.append(current_doc)
            
        print("Writing to a file the intermidate data")
        with open(channel_intermid_file,'w') as f:
            json.dump(documents_lemmatized,f)
            
        print('Finish processing video %s' %(video_id))
        
        
        
    
    
    
        


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    raw_dir = os.path.join(project_dir,"data","raw")
    interim_dir = os.path.join(project_dir,"data","interim")
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    
    
    
    main()
    
    
    
    
    
    
    
    