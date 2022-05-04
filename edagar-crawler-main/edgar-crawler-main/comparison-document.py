import json

import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

file_name_list = glob.glob('./datasets/EXTRACTED_FILINGS/*.*')

print(len(file_name_list))

column_list =['cik', 'company','filing_type','filing_date','item_1', 'item_2','item_3','item_4','item_5','item_6','item_7','item_8','item_9','item_10']

filings_df = pd.DataFrame(columns=column_list)

for file_name in file_name_list :
    file_name = file_name.replace('\\','/')
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data_dict = {'cik':data['cik'], 
                 'company':data['company'],
                 'filing_type':data['filing_type'],
                 'filing_date':data['filing_date'],
                 'item_1':data['item_1'], 
                 'item_2':data['item_2'],
                 'item_3':data['item_3'],
                 'item_4':data['item_4'],
                 'item_5':data['item_5'],
                 'item_6':data['item_6'],
                 'item_7':data['item_7'],
                 'item_8':data['item_8'],
                 'item_9':data['item_9'],
                 'item_10':data['item_10']}
    
    filings_df = filings_df.append(data_dict, ignore_index=True)
    

for idx in range(len(filings_df)):
    pre_name = filings_df.iloc[idx]['cik']
    
    if pre_name == filings_df.iloc[idx]['cik'] and idx > 0:
        # cik 일치
        document_list = [filings_df.iloc[idx-1]['item_1'], filings_df.iloc[idx]['item_1']]
        tfidf_model = TfidfVectorizer().fit_transform(document_list)
        cosine_sim = cosine_similarity(tfidf_model, tfidf_model)
        print(cosine_sim)
        
    else:
        pass
    
    
    