import os
import json

from glob import glob
from pprint import pprint
from tqdm import tqdm

path, file_name = 'G:/내 드라이브/Financial-Pretrained-Model/pretrained_dataset/Corporate Reports 10-K_10_Q/', '*/*.json'

edgar_list = glob(os.path.join(path, file_name), recursive=True)

total_list = []
for edgar_file in tqdm(edgar_list):
    with open(edgar_file, 'r', encoding='utf-8') as file :
        edgar_dict = json.load(file)
    
    data_text = []
    for key, val in edgar_dict.items():
        if 'item' in key:
            if val:
                data_text.append(val)

    data_text = ' '.join(data_text)

    data_dict = {
        "title" : edgar_dict['filing_type'] + '__' + edgar_dict['company'],
        "date" : edgar_dict['filing_date'],
        "patform": 'edgar_filings',
        "url" : edgar_dict['filing_html_index'],
        "category": edgar_dict['filing_type'],
        'contents': data_text
    }

    total_list.append(data_dict)

with open('edgar_filings_corpus.json', 'w', encoding='utf-8') as file :
    json.dump(total_list, file)


