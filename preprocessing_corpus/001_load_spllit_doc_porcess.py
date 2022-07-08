import os
import json
import time

import glob
from collections import Counter
from pprint import pprint
import pandas as pd
from sentence_splitter import SentenceSplitter, split_text_into_sentences


def split_sentence(doc, i, total_count, start_time, cleaninput):
    sentence_list = list()
    if 'content' in doc.keys():
        key = 'content'
    else: key = 'contents'
    
    if str(doc[key]) in ['nan', 'None']:
        pass
    else :
        input_text = doc[key]
        input_text = cleaninput.cleanInput(input_text)
        sentence_list = split_text_into_sentences(input_text)

    elapsed_time = time.time() - start_time
    
    if i % 10 == 0: 
        if i != 0:
            remain_time = (elapsed_time/i) * (total_count-i)
            pprint(f"[{i}/{total_count}], [경과 시간 : {elapsed_time}], 남은 시간 : {remain_time}")
    
    return sentence_list

def make_counter_dataset(path, f_name):
    json_file_list = glob.glob(os.path.join(path,f_name))
    for json_file in json_file_list:
        fname = json_file.rsplit('/')[-1]
        total_list = list()
        
        with open(os.path.join(json_file), 'r', encoding='utf-8') as file:
            doc_list = json.load(file)
            
        temp_list = []
        for doc in doc_list:
            temp_list.extend(doc)

        pprint(f'This is File name : {fname}')
        sentence_counter = Counter(temp_list)

        pprint(len(sentence_counter))
        sentence_counter -= Counter(sentence_counter.keys())

        '''counter to dataframe''' 
        pprint(len(sentence_counter))
        pprint(len([val for val in sentence_counter.values() if val > 10]))

        'Extract to Top 1000 Sentnece'
        sentence_df = pd.DataFrame.from_records(sentence_counter.most_common(1000)).reset_index()
        # sentence_df = pd.DataFrame.from_dict(sentence_counter, orient='index').reset_index()
        sentence_df = sentence_df.rename(columns={'index':'sentence', '0':'count'})
        sentence_df.to_excel(os.path.join(path, f'{fname}_most_common_1000.xlsx'))    


if __name__ == "__main__":
    path, f_name = "/home/ailab/Desktop/JY/split_document/","*.json"

    with open(os.path.join(path,'split_AIHUB_finance_dataset.json_277969.json'), 'r', encoding='utf-8') as file:
        edgar_list = json.load(file)

    for data in edgar_list:
        pprint(data)
    

        
            
    