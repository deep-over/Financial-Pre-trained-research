test = ['As Adopted Pursuant to Section 302', 
'of the Sarbanes-Oxley Act of 2002 Filed Herewith 31.2 Certification of Chief ',
'Financial Officer Pursuant to Rule 13a-14 of the Securities Exchange Act of 1934', 
'As Adopted Pursuant to Section 302 of the Sarbanes-Oxley Act of 2002 Filed Herewith 32.1 ',
'Certification of Chief Executive Officer Pursuant to 18 U.S.C. Section 1350', 
'As Adopted Pursuant to Section 906 of the Sarbanes-Oxley Act of 2002 Filed Herewith ',
'32.2 Certification of Chief Financial Officer Pursuant to 18 U.S.C. Section 1350', 
'As Adopted Pursuant to Section 906 of the Sarbanes-Oxley Act of 2002',
'Filed Herewith 101.INS Inline XBRL Instance Document- the instance document does not appear', 
'in the Interactive Data File because its XBRL tags are embedded within the Inline', 
'XBRL document.', 'Filed Herewith 101.SCH Inline XBRL Schema Document Filed Herewith', 
'101.CAL Inline XBRL Calculation Linkbase Document Filed Herewith 101.LAB Inline ',
'XBRL Label Linkbase Docume']

import torch
import random
import numpy as np
from transformers import BartTokenizer
from nltk.tokenize import sent_tokenize
from sentence_splitter import split_text_into_sentences
import nltk
# nltk.download('punkt')

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

test = ' '.join(test)
test = sent_tokenize(test)
_lambda = np.random.poisson(lam=3, size=len(test))

for i in range(len(test)):
    word_list = test[i].split()
    span_mask_num = _lambda[i]
    word_index = random.randrange(int(len(word_list)-span_mask_num))
    
    test[i] = word_list[:word_index] + [tokenizer.mask_token] + word_list[word_index+span_mask_num:]
    test[i] = ' '.join(test[i])

random.shuffle(test)

test =' '.join(test)
textdataset = tokenizer(test,
        return_tensors = 'pt'
        )

decoder_input_ids = textdataset["input_ids"][0]
decoder_attention_mask = textdataset["attention_mask"][0]
encoder_input_ids = decoder_input_ids.clone()
encoder_attention_mask = decoder_attention_mask.clone()

print(textdataset)

# Masking
sequence_length = encoder_attention_mask.sum()
num_masking = int(sequence_length * 0.3)
indices = torch.randperm(sequence_length)[:num_masking]
encoder_input_ids[indices] = tokenizer.mask_token_id
print(encoder_input_ids)