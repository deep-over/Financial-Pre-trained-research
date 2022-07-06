import re
from transformers import RobertaTokenizer

class CleanInput():
    def __init__(self) -> None:
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def cleanInput(self, input):
        input = re.sub('\n+', " ", input)
        input = re.sub('\[[0-9]*\]', "", input)
        input = re.sub(' +', " ", input)
        input = bytes(input, "UTF-8")
        input = input.decode("ascii", "ignore")
        return input

    def get_token_to_ids(self, input):

        input_index = self.tokenizer(input)['input_ids']
        
        return input_index
