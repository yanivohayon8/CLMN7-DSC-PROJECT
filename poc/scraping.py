# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 20:39:04 2020

@author: yaniv
"""

#from googlesearch import search
#google = importlib.import_module("C:\Users\yaniv\Anaconda3\Lib\site-packages\googlesearch\__init__.py")

# pip install   beautifulsoup4,google
from googlesearch import  search

query = "America filetype:pdf"

my_results_list = []
for i in search(query,        # The query you want to run
                tld = 'com',  # The top level domainpip install beautifulsoup4
                lang = 'en',  # The language
                num = 10,     # Number of results per page
                start = 0,    # First result to retrieve
                stop = 10, # None,  # Last result to retrieve
                pause = 2.0,  # Lapse between HTTP requests
               ):
    my_results_list.append(i)
    print(i)
    
from pathlib import Path
import requests

filename = Path("vika.pdf")
url = "https://www.state.gov/wp-content/uploads/2019/11/America-Crece-FAQs-003-508.pdf"
response = requests.get(url)
filename.write_bytes(response.content)

# pip install youtube-search
from youtube_search import YoutubeSearch
results = YoutubeSearch('how to dunk', max_results=10).to_json()
print(results)



