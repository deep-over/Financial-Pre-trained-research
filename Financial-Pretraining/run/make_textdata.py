
from glob import glob
import json
import os


path, name = '/home/ailab/Desktop/JY/split_document/', 'nytimes_news_data.json'

json_list = glob(os.path.join(path,name))

for json_name in json_list :
    with open(json_name, 'r', encoding='utf-8') as file :
        document_list = json.load(file)

    text = ''
    for doc in document_list :
        text = ' '.join(doc)
        print(text)
