import json
import os
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob

path, name = '/home/ailab/Desktop/JY/split_document/', '*.json'
corpus_list = glob(os.path.join(path, name))

count = 0
corpus_count_list = []
for corpus in corpus_list:
    corpus_name = corpus.rsplit('/')[-1].split('.')[0]

    with open(corpus, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    tokens_count = 0
    for doc in dataset:
        token_len = sum([len(x.split()) for x in doc])
        tokens_count +=  token_len

    corpus_count_list.append([corpus_name, len(dataset), tokens_count])
    count += len(dataset)

df = pd.DataFrame(corpus_count_list, columns=['name','sentence','tokens'])
df.to_csv('corpus_count.csv')