import os
import json
import glob
import time

import concurrent.futures

from collections import Counter
from pprint import pprint
from tqdm import tqdm
from common.cleanInput import CleanInput

def load_counter_json(json_file):
    with open(os.path.join(json_file), 'r', encoding='utf-8') as file :
        load_cnter = json.load(file)
    
    name = json_file.rsplit('\\')[-1]
    
    pprint(f"Success Load Json File : {name}, Counter Length : {len(load_cnter)}")
    
    return load_cnter

def merge_counters(path, name):
    total_counter = Counter()
    
    json_file_list = glob.glob(os.path.join(path,name))
    
    for json_file in json_file_list:
        name_counter = load_counter_json(json_file)
        total_counter += name_counter
     
    print(len(total_counter))   
    total_counter = total_counter - Counter(total_counter.keys())
    print(len(total_counter))

    return total_counter

def remove_overlap_document(origin_path, jname, total_counter):
    start_time = time.time()
    cleanInput = CleanInput()
    overlap_doc_list = list()
        
    origin_file_list = glob.glob(os.path.join(origin_path, jname))
    
    total_counter = total_counter
    for origin_file in origin_file_list:
        
        fname = origin_file.rsplit('\\')[-1]
        
        with open(origin_file, 'r', encoding="utf-8") as fp:
            origin_docs = json.load(fp)
        
        
        pprint("load_to_document")
        total_counter_cnt = len(total_counter)
        for idx in tqdm(range(len(origin_docs))):
            doc = origin_docs[idx]
            
            if 'content' in doc.keys():
                key = 'content'
            else: key = 'contents'
            
            if str(doc[key]) in ['nan', 'None']:
                overlap_doc_list.append(f"{fname}_{idx}")
                continue
            
            ngram_list = cleanInput.main(doc[key])
            
            for ngram_doc in ngram_list:
                if ngram_doc in total_counter:          
                    print(ngram_doc)
                    total_counter = total_counter - Counter(ngram_list)
                    trans_counter_time = time.time() - start_time
                    total_counter_cnt = len(total_counter)
                    pprint(f"success counter : {trans_counter_time}")
            if total_counter_cnt == len(total_counter):
                continue
            else:
                overlap_doc_list.append(f"{fname}_{idx}")
            print(len(overlap_doc_list))

    success_time = time.time() - start_time
    print(f"경과시간 : {success_time}")
        
if __name__ == "__main__":
    
    """
    counter_path : Ngram Counter Dictionary path
    origin_path  : Original Train dataset - This text file remove to overlap document
    jname        : *.json
    """
    counter_path, origin_path, jname = "ngram_dictionary", "G:/내 드라이브/Financial-Pretrained-Model/Corpora", "*.json"
    
    total_counter = merge_counters(counter_path, jname)
    
    cleanInput = CleanInput()
    tokenizer = cleanInput.get_tokenizer()
    for ids in total_counter.most_common(50):
        cnt = ids[1]
        key = json.loads(ids[0])
        print(tokenizer.decode(key))
        print(cnt)
    
    
    # remove_overlap_document(origin_path, jname, total_counter)
    
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
    
    procs = []
    
    
    
    