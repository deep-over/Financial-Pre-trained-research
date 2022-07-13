import csv
import json
from lib2to3.pgen2.tokenize import tokenize
import random
from typing import Dict, List, Optional, Tuple

import torch

import numpy as np
from transformers.tokenization_utils import PreTrainedTokenizerBase, PreTrainedTokenizer

def load_json_data(path: str) -> Tuple[List[str], List[List[str]], List[str]]:
    """Load dialogue summarization dataset json files of https://aihub.or.kr/aidata/30714

    Args:
        path: path of json file
    Returns:
        result of file, which is a tuple of ids, dialogues, summaries
    """
    with open(path) as f:
        data = json.load(f)
    
    corpus_id = 0
    for char in str(path):
        corpus_id += ord(char) 

    ids = []
    texts = []
    "Transfer 'For loop' code --> Financial Corpus Format"
    """
    data format :
        "title"        : "Approximation ..."
        "contents"     : "We build a..."
        "date"         : "2022-11-26"
        "platform"     : "historical"
        "url"          : "http://www..."
        "category"     : "abstract"
    """
    for idx, datum in enumerate(data):
        # ids.append(datum["header"]["dialogueInfo"]["dialogueID"])
        ids.append(str(corpus_id) + "_" + str(idx))
        
        if 'content' in datum.keys():
            key = 'content'
        else: key = 'contents'
        
        if str(datum[key]) in ['nan', 'None']:
            continue
        else:
            texts.append(datum[key])
    return ids, texts


def load_tsv_data(path: str) -> Tuple[List[str], List[List[str]], List[str]]:
    """Load arbitrary tsv file of formed like (id, dialogue, summary) with header
    each `dialogue` should be dumped json string from a list of utterances.
    ex) '["안녕", "잘가"]'

    Args:
        path: path of tsv file
    Returns:
        result of file, which is a tuple of ids, dialogues, summaries
    """
    ids = []
    dialogues = []
    summaries = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ids.append(row["id"])
            dialogues.append(json.loads(row["dialogue"]))
            summaries.append(row.get("summary"))
    return ids, dialogues, summaries


class DialogueSummarizationDataset(torch.utils.data.Dataset):
    """Dataset for Dialogue Summarization

    Attributes:
        sep_token: token to seperate utterances
        ids: id of each example
        dialogues: dialogue of each example
        summaries: summary of each example
        dialogue_input_ids: dialogue input id tokens of each example
        dialogue_attention_masks: dialogue attention masks of each example
        summary_input_ids: summary input id tokens of each example
        summary_attention_masks: summary attention masks of each example
    """

    def __init__(
        self,
        paths: List[str],
        tokenizer: PreTrainedTokenizerBase,
        dialogue_max_seq_len: int,
        summary_max_seq_len: int,
        use_summary: bool,
    ):
        """
        Args:
            paths: list of dataset paths (tsv or json)
            tokenizer: tokenizer to tokenize dialogue and summary string
            dialogue_max_seq_len: max sequence length of dialouge
            summary_max_seq_len: max sequence length of summary
            use_summary: whether to use summary data or not (should be False for inference)
        """
        super().__init__()

        self.sep_token = tokenizer.sep_token
        (
            self.ids,
            self.dialogues,
            self.summaries,
            self.dialogue_input_ids,
            self.dialogue_attention_masks,
            self.summary_input_ids,
            self.summary_attention_masks,
        ) = self.load_dataset(paths, tokenizer, dialogue_max_seq_len, summary_max_seq_len, use_summary)

    def load_dataset(
        self,
        paths: List[str],
        tokenizer: PreTrainedTokenizerBase,
        dialogue_max_seq_len: int,
        summary_max_seq_len: int,
        use_summary: bool,
    ) -> Tuple[
        List[str],
        List[List[str]],
        List[str],
        List[torch.Tensor],
        List[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        """Load dataset files and featurize with tokenizer

        Args:
            paths: list of dataset paths (tsv or json)
            tokenizer: tokenizer to tokenize dialogue and summary string
            dialogue_max_seq_len: max sequence length of dialouge
            summary_max_seq_len: max sequence length of summary
            use_summary: whether to use summary data or not (should be False for inference)
        Returns:
            original ids, dialogues, summaries and input ids and attention masks for dialogues and summaries
        """
        ids, dialogues, summaries = [], [], []
        for path in paths:
            loader_fn = load_tsv_data if path.endswith(".tsv") else load_json_data

            file_ids, file_dialogues, file_summaries = loader_fn(path)
            ids.extend(file_ids)
            dialogues.extend(self.sep_token.join(x) for x in file_dialogues)
            summaries.extend(file_summaries)

        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        dialogue_inputs = tokenizer(
            [bos + x + eos for x in dialogues],
            padding="max_length",
            truncation=True,
            max_length=dialogue_max_seq_len,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        summary_inputs = (
            tokenizer(
                [bos + x + eos for x in summaries],
                padding="max_length",
                truncation=True,
                max_length=summary_max_seq_len,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            if use_summary
            else {}
        )

        return (
            ids,
            dialogues,
            summaries,
            dialogue_inputs["input_ids"],
            dialogue_inputs["attention_mask"],
            summary_inputs.get("input_ids"),
            summary_inputs.get("attention_mask"),
        )

    def __len__(self) -> int:
        return len(self.dialogue_input_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {"input_ids": self.dialogue_input_ids[index], "attention_mask": self.dialogue_attention_masks[index]}
        if self.summary_input_ids is not None and self.summary_attention_masks is not None:
            item.update(
                {
                    "decoder_input_ids": self.summary_input_ids[index],
                    "decoder_attention_mask": self.summary_attention_masks[index],
                }
            )
        return item


class PretrainDataset(torch.utils.data.Dataset):
    """Dataset for pretraining of BART

    Attributes:
        tokenizer: tokenizer to tokenize dialogue and summary string
        dialogue_max_seq_len: max sequence length of dialouge
        masking_rate: rate of the number of masked token / sequence length
        bos_token: bos token
        eos_token: eos token
        sep_token: turn seperation token to divide each utterances
        mask_token_id: mask token id for text infilling
        ids: id of each example
        dialogues: dialogue of each example
    """

    def __init__(
        self,
        paths: List[str],
        tokenizer: PreTrainedTokenizer,
        dialogue_max_seq_len: int,
        masking_rate: float = 0.3,
    ):
        """
        Args:
            paths: list of dataset paths (tsv or json)
            tokenizer: tokenizer to tokenize dialogue and summary string
            dialogue_max_seq_len: max sequence length of dialouge
            masking_rate: rate of the number of masked token / sequence length
        Returns:
            original ids, dialogues, summaries and input ids and attention masks for dialogues and summaries
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.max_seq_len = dialogue_max_seq_len
        self.masking_rate = masking_rate
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.sep_token = tokenizer.sep_token
        self.mask_token_id = tokenizer.mask_token_id
        self.ids, self.dialogues = self.load_dataset(paths)

    def load_dataset(self, paths: List[str]) -> Tuple[List[str], List[List[str]]]:
        """Load dataset files and featurize with tokenizer

        Args:
            paths: list of dataset paths (tsv or json)
        Returns:
            original ids, and sep token joined dialogues
        """
        block_size = 512 - self.tokenizer.num_special_tokens_to_add(pair=False)
        ids, texts = [], []
        for path in paths:
            with open(path, 'r', encoding='utf-8') as fp :
                loader = json.load(fp)
            
            count = 0
            for doc in loader :
                random.shuffle(doc)
                _lambda = np.random.poisson(lam=3, size=len(doc))
                
                tokenized_text = []
                for i in range(len(doc)):
                    word_list = doc[i].split()
                    span_mask_num = _lambda[i]

                    if span_mask_num == 0:
                        mask_word_index = random.randrange(int(len(word_list)))
                        word_list[mask_word_index] = self.tokenizer.mask_token
                    
                    sentence = word_list[:mask_word_index] + [self.tokenizer.mask_token] + word_list[mask_word_index+span_mask_num:]
                    sentence = ' '.join(sentence)

                    tokenized_sentence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence))
                    if len(tokenized_text) + len(tokenized_sentence) > block_size:
                        texts.append(self.tokenizer.build_inputs_with_special_tokens(tokenized_text))
                        ids.append(count)
                        count += 1
                        
            # file_ids, file_texts = loader_fn(path)
            # ids.extend(file_ids)
            # texts.extend(file_texts)

        return ids, texts

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        dialogue = list(self.dialogues[index])
        
        # Permutate
        random.shuffle(dialogue)

        # Tokenize
        dialogue_input = self.tokenizer(
            self.bos_token + self.sep_token.join(dialogue) + self.eos_token,
            padding="max_length",
            truncation=True,
            max_length=self.dialogue_max_seq_len,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        decoder_input_ids = dialogue_input["input_ids"][0]
        decoder_attention_mask = dialogue_input["attention_mask"][0]
        encoder_input_ids = decoder_input_ids.clone()
        encoder_attention_mask = decoder_attention_mask.clone()

        # Masking
        sequence_length = encoder_attention_mask.sum()
        num_masking = int(sequence_length * self.masking_rate)
        indices = torch.randperm(sequence_length)[:num_masking]
        encoder_input_ids[indices] = self.mask_token_id

        return {
            "input_ids": encoder_input_ids,
            "attention_mask": encoder_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }
        
        
class RoBERTaPretrainDataset(torch.utils.data.Dataset):
    """Dataset for pretraining of RoBERTa

    Attributes:
        tokenizer: from_pretrained('roberta-base)
        dialogue_max_seq_len: max sequence length of dialouge
        masking_rate: rate of the number of masked token / sequence length
    """

    def __init__(
        self,
        paths: List[str],
        tokenizer: PreTrainedTokenizerBase,
        dialogue_max_seq_len: int,
        masking_rate: float = 0.3,
    ):
        """
        Args:
            paths: list of dataset paths (tsv or json)
            tokenizer: tokenizer to tokenize dialogue and summary string
            dialogue_max_seq_len: max sequence length of dialouge
            masking_rate: rate of the number of masked token / sequence length
        Returns:
            original ids, dialogues, summaries and input ids and attention masks for dialogues and summaries
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.dialogue_max_seq_len = dialogue_max_seq_len
        self.masking_rate = masking_rate
        self.ids, self.examples = self.load_dataset(paths)

    def load_dataset(self, paths: List[str]) -> Tuple[List[str], List[List[str]]]:
        """Load dataset files and featurize with tokenizer
 
        Args:
            paths: list of dataset paths (tsv or json)
        Returns:
            original ids, and sep token joined dialogues
        """
        ids, texts = [], []
        for path in paths:
            loader_fn = load_tsv_data if path.endswith(".tsv") else load_json_data
            
            file_ids, file_texts = loader_fn(path)
            ids.extend(file_ids)
            texts.extend(file_texts)
            
        return ids, texts
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        text = self.examples[i]
        
        
        return torch.tensor(self.examples[i], dtype=torch.long)
