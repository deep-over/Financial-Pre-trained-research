import os
import json
import re
import string
import time

import multiprocessing
import concurrent.futures
import glob

import parmap
from collections import Counter
from tqdm import tqdm
from pprint import pprint
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer

def cleanInput(input):
    input = re.sub('\n+', " ", input)
    input = re.sub('\[[0-9]*\]', "", input)
    input = re.sub(' +', " ", input)
    input = bytes(input, "UTF-8")
    input = input.decode("ascii", "ignore")
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    input_index = tokenizer(input)['input_ids']
    return input_index

def load_dataset(path, f_name):
    with open("train_dataset/train_Edgar_Filings_Corpus.json", 'r', encoding='utf-8') as file :
        edgar_list = json.load(file)
    
    return edgar_list

def get_ngram(input_index):
    n = 13
    output = []
    for i in range(len(input_index)-n+1):
        output.append(input_index[i:i+n])
    return output

def preprocessing_text(doc, i, total_count, start_time):
    ngram_cnter = Counter()
    if 'content' in doc.keys():
        key = 'content'
    else: key = 'contents'
    
    if str(doc[key]) in ['nan', 'None']:
        return ngram_cnter
    else :
        input_text = doc[key]
        input_index = cleanInput(input_text)
        ngram_list = get_ngram(input_index)
        ngram_list = [str(x) for x in ngram_list]
        
        ngram_cnter = Counter(ngram_list)
    
    elapsed_time = time.time() - start_time
    
    
    if i % 10 == 0: 
        if i != 0:
            remain_time = (elapsed_time/i) * (total_count-i)
            pprint(f"[{i}/{total_count}], [경과 시간 : {elapsed_time}], 남은 시간 : {remain_time}")
    
    return ngram_cnter


if __name__ == "__main__":
    
    path, f_name = "train_dataset","*.json"
    path = "G:/내 드라이브/Financial-Pretrained-Model/Corpora/"
    
    json_file_list = glob.glob(os.path.join(path,f_name))
    
    for json_file in json_file_list:
        
        fname = json_file.rsplit('\\')[-1]
        ngram_total_cnter = Counter()
        
        with open(os.path.join(json_file), 'r', encoding='utf-8') as file:
            doc_list = json.load(file)
            
        start_time = time.time()
        i_count = 0
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=30)
        
        procs = []
        for i in range(len(doc_list)):
            procs.append(pool.submit(preprocessing_text, doc_list[i], i, len(doc_list), start_time))
        
        second_time = time.time()
        total_procs = len(procs)
        for p in concurrent.futures.as_completed(procs):
            i_count +=1
            second_elapsed_time = time.time() - second_time
            
            remain_time = (second_elapsed_time/i_count) * (total_procs-i_count)
            if i_count % 10 == 0:
                pprint(f"Ngram_Merge Count : {i_count}, [경과 시간 : {second_elapsed_time}], 남은 시간 : {remain_time}, Procs : {fname}, {total_procs}")
            
            ngram_total_cnter += p.result()
    
        # ngram_total_cnter = ngram_total_cnter - Counter(ngram_total_cnter.keys())
        with open(f'ngram_counter_{fname}.json', 'w', encoding='utf-8') as file:
            json.dump(ngram_total_cnter, file)
            
    