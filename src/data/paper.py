# -*- coding: utf-8 -*-


'''Find the following for each paper:
    1) main chapter title index. for example [10,15....]
    2) range of subsection within each main chapter . for example [(0,9),(11,14)...]

    '''

# paper_mainchapter_indexes = {}
from functools import reduce
import re


def get_sub_titles(all_titles, main_tl_indexes):
    subsec_mainchapter_indexes = [range(main_tl_indexes[index], main_tl_indexes[index + 1])
                                  for index in range(len(main_tl_indexes) - 1)]
    subsec_mainchapter_indexes.append(range(main_tl_indexes[-1], len(all_titles)))
    return subsec_mainchapter_indexes


def get_chapter_single_doc(paper_content):
    '''
        first element array of subtitles by dividing based on main chapters like 4 number theory
        second element array of subtitles by dividing based on main first hierarchy depth titles like 4.1 simple graph
    '''
    titles_by_hierchy_ranges = []
    titles_by_hierchy_indexes = []

    '''Find the main chapter indexes in the list of the overall titles'''
    '''Titles like 4 Number theory or 5 Graphs'''
    mainchapter_indexes = [paper_content['titles'].index(ch_title)
                           for ch_title in paper_content['main titles']]
    titles_by_hierchy_indexes.append(mainchapter_indexes)

    '''
        Find the indexes of the first depth subsection
        for example 4.1 simple graphs
    '''
    tl_first_dep_by_hier_indexes = [i for i, tl in enumerate(paper_content['titles'])
                                    if re.match(r'(^([0-9]+\.[0-9]*)$)', tl) is not None or
                                    re.match(r'[0-9]+\.[0-9]*\s+', tl) is not None]

    titles_by_hierchy_indexes.append(tl_first_dep_by_hier_indexes)

    tl_first_dep_by_hier = [tl for i, tl in enumerate(paper_content['titles'])
                            if re.match(r'(^([0-9]+\.[0-9]*)$)', tl) is not None or
                            re.match(r'[0-9]+\.[0-9]*\s+', tl) is not None]

    '''Adding the indexes of the first dep subsections to the main indexes'''
    '''For example the fourth place will contains the indexes of the titles: 4.1,4.2,4.3'''
    top_maintitle_ranges = []

    for index in range(len(mainchapter_indexes) - 1):
        j = mainchapter_indexes[index]
        range_j = []
        while j < mainchapter_indexes[index + 1]:
            '''If this title is from the shape like 4.1 or 4.2'''
            curr_tl = paper_content['titles'][j]
            if j in tl_first_dep_by_hier_indexes:
                range_j.append(tl_first_dep_by_hier.index(curr_tl))
            j += 1
        top_maintitle_ranges.append(range_j)
    '''Adding the last one'''
    j = mainchapter_indexes[-1]
    range_j = []
    while j < len(paper_content['titles']):
        curr_tl = paper_content['titles'][j]
        if j in tl_first_dep_by_hier_indexes:
            range_j.append(tl_first_dep_by_hier.index(curr_tl))
        j += 1
    top_maintitle_ranges.append(range_j)

    titles_by_hierchy_ranges.append(top_maintitle_ranges)

    '''adding the ranges of the subtitles of the first depth of each '''
    '''like for example for title 4.1 the range will be the indexes of the titles
        4.1.1,4.1.2 etc'''
    titles_by_hierchy_ranges.append(get_sub_titles(paper_content['titles'],
                                                   titles_by_hierchy_indexes[1]))

    '''Making each chapter as a one documents'''

    '''Union all the documents in a atomic section into single document'''
    paper_subsec_as_one_doc = [list(reduce(lambda doc, acc: doc + acc, sec, []))
                               for sec in paper_content['corpus']]

    '''Sections are like 4.1 simple graph'''
    first_dep_section_as_one_doc = [list(reduce(lambda acc, s_i:
                                                paper_subsec_as_one_doc[s_i] + acc,
                                                subsec_indexes, []))
                                    for subsec_indexes in titles_by_hierchy_ranges[1]]

    '''Union all the sub section in a main chapter into one document'''
    '''sections are like 4 Number theory'''
    main_section_as_one_doc = [list(reduce(lambda acc, s_i:
                                           first_dep_section_as_one_doc[s_i] + acc,
                                           subsec_indexes, []))
                               for subsec_indexes in titles_by_hierchy_ranges[0]]

    return main_section_as_one_doc, first_dep_section_as_one_doc, paper_subsec_as_one_doc, titles_by_hierchy_ranges, tl_first_dep_by_hier, tl_first_dep_by_hier_indexes