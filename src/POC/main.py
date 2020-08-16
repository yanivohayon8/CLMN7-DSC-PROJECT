import sys
sys.path.append('..')


from data.docx import find_content,read_docx
from data.paper import get_chapter_single_doc

# parameters
#vid = 'NuY7szYSXSw'
#const
lemmatizing_method ='lemma'
doc_path = '../../data/raw/docx/MIT6_042JF10_notes.docx'

full_text,font_sizes = read_docx(doc_path)
paper_content = find_content('MIT6_042JF10_notes',full_text,font_sizes,'Chapter',lemmatizing=lemmatizing_method)
main_section_as_one_doc,first_dep_section_as_one_doc,paper_subsec_as_one_doc,titles_by_hierchy_ranges,tl_first_dep_by_hier,tl_first_dep_by_hier_indexes = get_chapter_single_doc(paper_content)
print("end")