import sys
sys.path.append('..')


from src.data.docx import find_content,read_docx
from src.data.paper import get_chapter_single_doc
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import spacy
from nltk.stem import PorterStemmer
from gensim.corpora import Dictionary
from gensim import models
from gensim import similarities
import seaborn as sns; sns.set()
import pandas as pd
from functools import reduce
from sklearn.feature_extraction.text import CountVectorizer
import heapq
import os

#warnings.filterwarnings("ignore")

def getLabeledDivision(division_time_shifts, division_words):
    # --- read text from document and devided it to hirarchycal sections ---
    #division_time_shifts = [1,56]
    print(division_time_shifts)
    #division_words = [{'bla','bla','bla','bla','bla'},{'bla','bla','bla','bla','bla'},{'bla','bla','bla','bla','bla'}]
    print(division_words)
    videos_division = {'topic_words': division_words, 'topic_shift': division_time_shifts}
    lemmatizing_method = 'lemma'
    dirname = os.path.dirname(__file__)
    doc_path = os.path.join(dirname, '../../data/raw/docx/MIT6_042JF10_notes.docx')
    full_text, font_sizes = read_docx(doc_path)
    paper_content = find_content('MIT6_042JF10_notes', full_text, font_sizes, 'Chapter', lemmatizing=lemmatizing_method)
    main_section_as_one_doc, first_dep_section_as_one_doc, paper_subsec_as_one_doc, titles_by_hierchy_ranges, tl_first_dep_by_hier, tl_first_dep_by_hier_indexes = get_chapter_single_doc(
        paper_content)


    correlations_levels = []

    '''Finding the correlation between topic and main chapters'''
    main_matching_topic, correlation = get_topic_chapter_corr_tfidf(main_section_as_one_doc,
                                                                        videos_division['topic_words'],
                                                                        videos_division['topic_shift'],
                                                                        paper_content['main titles'],
                                                                        is_print_mess=False)

    correlations_levels.append(correlation)

    dominent_chapter = find_dominent_main_chapter(main_matching_topic,
                                                     paper_content['main titles'],
                                                     correlation)

    '''Finding correlation to the first dep sections within the selected main chapter'''
    dom_main_cha_index = paper_content['main titles'].index(dominent_chapter)
    dom_subsec_text = [first_dep_section_as_one_doc[s_i]
                       for s_i in titles_by_hierchy_ranges[0][dom_main_cha_index]]
    dom_subsec_titles = [tl_first_dep_by_hier[tl_i]
                         for tl_i in titles_by_hierchy_ranges[0][dom_main_cha_index]]

    dom_subsec_text = emphasize_title(dom_subsec_text,
                                      dom_subsec_titles,
                                      main_section_as_one_doc,
                                      lemmatizing=lemmatizing_method)

    subsec_matching_topic, correlation = get_topic_chapter_corr_tfidf(dom_subsec_text,
                                                                      videos_division['topic_words'],
                                                                      videos_division['topic_shift'],
                                                                      dom_subsec_titles, is_print_mess=False)

    correlations_levels.append(correlation)
    section_to_topic = []

    '''Finding the third ganular level of chossing the chapter'''
    for tp_i, section in enumerate(subsec_matching_topic):
        '''
            Finding the subsections title and texts of the first dep section (4.1 etc)
        '''
        curr_first_dep_tl_index = tl_first_dep_by_hier.index(section)
        subsec_range = titles_by_hierchy_ranges[1][curr_first_dep_tl_index]
        subsection_titles = [paper_content['titles'][tl_i] for tl_i in subsec_range]
        subsection_text = [paper_subsec_as_one_doc[tx_i] for tx_i in subsec_range]
        subsection_text = emphasize_title(subsection_text,
                                          subsection_titles, main_section_as_one_doc,
                                          lemmatizing=lemmatizing_method)

        topic_matching, topic_corr = get_topic_chapter_corr_tfidf(subsection_text,
                                                                  [videos_division['topic_words'][tp_i]],
                                                                  None,
                                                                  subsection_titles,
                                                                  is_print_mess=False
                                                                  )
        section_to_topic = section_to_topic + topic_matching

    result = list(map(lambda section:
        {'start_time': 0,'lable': section }
        , section_to_topic))

    for index in range(1,len(result)):
        result[index]['start_time'] = division_time_shifts[index-1]

    return result



def get_sub_titles(all_titles,main_tl_indexes):
    subsec_mainchapter_indexes = [range(main_tl_indexes[index],main_tl_indexes[index + 1])
                                  for index in range(len(main_tl_indexes) - 1)]
    subsec_mainchapter_indexes.append(range(main_tl_indexes[-1],len(all_titles)))
    return subsec_mainchapter_indexes

def get_topic_chapter_corr_tfidf( book_chapters,
                                 vid_topics_words, vid_topics_shift,
                                 dispaly_titles, pre_labeled_title=None, is_print_mess=True):

    raw_book_video = []
    raw_book_video = raw_book_video + vid_topics_words
    raw_book_video = raw_book_video + book_chapters
    for tp_vid in vid_topics_words:
        raw_book_video = raw_book_video + [tp_vid]

    for ch in book_chapters:
        raw_book_video = raw_book_video + [ch]

    # creating dictionary of all of the words in the corpus of the video and the paper
    dictionary = Dictionary(raw_book_video)

    # ch_dict = Dictionary(book_chapters)
    # vid_dict = Dictionary(vid_topics_words)

    the_dictionary = dictionary

    # whole_corpus = vid_topics_words + book_chapters
    # bgw_corpus = [dictionary.doc2bow(doc) for doc in whole_corpus]
    bgw_chapter = [the_dictionary.doc2bow(doc) for doc in book_chapters]
    bgw_vids = [the_dictionary.doc2bow(doc) for doc in vid_topics_words]

    '''Calculate the pivot '''
    _tmp = [len(list(set([w for w in ch]))) for ch in book_chapters]
    book_pivot = sum(_tmp) / len(_tmp)
    _tmp = [len(list(set([w for w in ch]))) for ch in vid_topics_words]
    vid_pivot = sum(_tmp) / len(_tmp)

    tf_idf_model_ch = models.TfidfModel(bgw_chapter,
                                        dictionary=the_dictionary  # ,
                                        # pivot=book_pivot#,
                                        # slope=0.8,
                                        # smartirs='nnc'
                                        )  # ,
    tf_idf_model_vid = models.TfidfModel(bgw_vids,
                                         dictionary=the_dictionary  # ,
                                         # pivot=vid_pivot#,
                                         # slope=0.4
                                         )  # ,smartirs='lfc'

    index_sim = similarities.SparseMatrixSimilarity(tf_idf_model_ch[bgw_chapter],
                                                    num_features=len(the_dictionary))

    correlation = [[s for s in index_sim[tf_idf_model_vid[doc]]] for doc in bgw_vids]
    # print(tf_idf_model[bgw_vids])

    # find the top n words in the topic (in the video)
    # print('$$$$$$$$$$$$$$top words of video topic$$$$$$$$$$$$$$')
    # find_top_words_of_topic(videos_division[vid]['topic_words'])
    # print('$$$$$$$$$$$$$$top words of book chapters$$$$$$$$$$$$$$')
    # find_top_words_of_topic(book_chapters,chapter_titles=dispaly_titles)

    # normalizing the correlation between each topic and chapter
    for i_t in range(len(correlation)):
        sum_ = sum(correlation[i_t])
        for ch_index in range(len(correlation[i_t])):
            correlation[i_t][ch_index] = correlation[i_t][ch_index] / sum_

    corr_as_row = reduce(lambda x, y: x + y, correlation, [])  # to get the global max in min

    if is_print_mess:
        sns.heatmap(correlation, vmin=min(corr_as_row), vmax=max(corr_as_row))

    # find the cha
    ch_matching_top = []
    # print(len(correlation[0]))

    for i, corr in enumerate(correlation):
        max_cor = max(corr)  # max correlation with that topic
        # shift = vid_topics_shift[i]
        founded_title = dispaly_titles[corr.index(max_cor)]
        # ch_matching_top.append(paper_mainchapter_indexes[paper_name][corr.index(max_cor)])
        ch_matching_top.append(founded_title)

    hit = 0
    miss = 0

    return ch_matching_top, correlation


def find_dominent_main_chapter(ch_tp_corr, titles, correlation):
    '''Find the frequency of each chapter'''
    chapter_matching_counts_max = [max([ch_tp_corr.count(ch) for ch in ch_tp_corr])]
    for max_count in chapter_matching_counts_max:
        # print("####### #######")
        '''Find the most frequent chapter'''
        # print(([ch for ch in ch_tp_corr if ch_tp_corr.count(ch) == max_count]))
        dominent_chapters = list(set([ch for ch in ch_tp_corr if ch_tp_corr.count(ch) == max_count]))
        # print(dominent_chapters)

        # if we have absulote majority on topic
        if len(dominent_chapters) == 1:
            return dominent_chapters[0]
        else:
            # draw between topics, decide which one by taking this with the high variance
            index_winner = 0
            df_ch_corr = pd.DataFrame.from_records(correlation)
            for j_dom in range(len(dominent_chapters)):
                first = df_ch_corr.var()[titles.index(dominent_chapters[index_winner])]
                # print(first)
                second = df_ch_corr.var()[titles.index(dominent_chapters[j_dom])]
                # print(second)
                if first > second:
                    index_winner = j_dom
            return (dominent_chapters[index_winner])


def emphasize_title(book_chapters, titles, main_section_as_one_doc, factor_enrich=20, lemmatizing="lemma"):
    stop_words = stopwords.words('english')
    nlp = spacy.load('en', disable=['parser', 'ner'])
    allowed_postags = ['NOUN', 'ADJ', 'VERB', 'PROPN',
                       'ADV']  # ['NOUN', 'ADJ', 'VERB','ADV']#['NOUN', 'ADJ', 'VERB','PROPN']# #['NOUN', 'ADJ', 'VERB']
    porter = PorterStemmer()

    chapters_enriched = book_chapters[:]

    for index, tl in enumerate(titles):
        tl_text_no_punc = simple_preprocess(tl, deacc=True)
        tokenized_text_non_stop_words = [word for word in tl_text_no_punc
                                         if word not in stop_words]
        """text_non_stop_words = ' '.join(tokenized_text_non_stop_words)
        tokenized_lemmas = nlp(text_non_stop_words)
        tokenized_lemmas = [token.lemma_ for token in tokenized_lemmas \
                            if token.pos_ in allowed_postags]"""

        if lemmatizing == "stemm":
            tokenized_lemmas = [porter.stem(w) for w in tokenized_text_non_stop_words]
        else:
            text_non_stop_words = ' '.join(tokenized_text_non_stop_words)
            tokenized_lemmas = nlp(text_non_stop_words)
            tokenized_lemmas = [token.lemma_ for token in tokenized_lemmas \
                                if token.pos_ in allowed_postags]

        tokenized_lemmas_ph = '_'.join(tokenized_lemmas)  # reduce(lambda acc,x: acc+x,
        # print('tokenized_lemmas_ph %s' %(tokenized_lemmas_ph))
        paper_phrasers = list(set(reduce(lambda acc, x: acc + x,
                                                   [[w for w in ch if '_' in w]
                                                    for ch in main_section_as_one_doc])))

        if tokenized_lemmas_ph in paper_phrasers:
            tokenized_lemmas = [tokenized_lemmas_ph]

        chapters_enriched[index] = chapters_enriched[index] + tokenized_lemmas * factor_enrich
    return chapters_enriched

def find_top_words_of_topic(topic_words, chapter_titles=None):
    for tp_i, tp_words in enumerate(topic_words):
        #
        raw_text = ' '.join(tp_words)
        myvectorizer = CountVectorizer()
        mytf = myvectorizer.fit_transform([raw_text]).toarray()
        # print(mytf)
        maxes = heapq.nlargest(3, mytf[0])
        indexes = []
        for i, bal in enumerate(mytf[0]):
            if bal in maxes:
                indexes.append(i)
        ws = [myvectorizer.get_feature_names()[_] for i, _ in enumerate(indexes)]

        if chapter_titles is None:
            print('top words for topic %s are %s' % (tp_i, ws))
        else:
            print('top words for topic %s are %s' % (chapter_titles[tp_i], ws))


