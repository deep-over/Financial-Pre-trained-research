import os
import json
import time

import concurrent.futures
import glob
from nltk.tokenize import sent_tokenize
from collections import Counter
from tqdm import tqdm
from pprint import pprint
from common.CleanInput import CleanInput

def load_dataset(path, f_name):
    with open("train_dataset/train_Edgar_Filings_Corpus.json", 'r', encoding='utf-8') as file :
        edgar_list = json.load(file)
    
    return edgar_list

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
        sentence_list = sent_tokenize(input_text)

    elapsed_time = time.time() - start_time
    
    if i % 10 == 0: 
        if i != 0:
            remain_time = (elapsed_time/i) * (total_count-i)
            pprint(f"[{i}/{total_count}], [경과 시간 : {elapsed_time}], 남은 시간 : {remain_time}")
    
    return sentence_list




if __name__ == "__main__":
    
    path, f_name = "/home/ailab/Desktop/JY/corpus/","*.json"
    
    json_file_list = glob.glob(os.path.join(path,f_name))
    cleaninput = CleanInput()
    for json_file in json_file_list:
        
        fname = json_file.rsplit('/')[-1]
        total_list = list()
        
        with open(os.path.join(json_file), 'r', encoding='utf-8') as file:
            doc_list = json.load(file)
            
        start_time = time.time()
        
        if fname in 'Edgar_Filings_Corpus':
            pool = concurrent.futures.ProcessPoolExecutor(max_workers=5)
        else:
            pool = concurrent.futures.ProcessPoolExecutor(max_workers=30)
        
        procs = []
        for i in range(len(doc_list)):
            procs.append(pool.submit(split_sentence, doc_list[i], i, len(doc_list), start_time, cleaninput))
        
        second_time = time.time()
        total_procs = len(procs)
        for idx, p in enumerate(concurrent.futures.as_completed(procs)):
            total_list.append(p.result())
            
        with open(f'split_document/split_{fname}_{idx}.json', 'w', encoding='utf-8') as file:
            json.dump(total_cnter, file)
            
    