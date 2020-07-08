
from transformers import BertTokenizer


class TokenizerWrapper:
    
    def __init__(self, bert_model):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
    
    # returns the value 
    def get_tokens(self, string):
        return self.tokenizer.vocab[string]
    
    # returns a list of tokens
    def encode(self, string):
        return self.tokenizer.encode(string)
    
    # returns a string (text)
    def decode(self, tokens_list): 
        return self.tokenizer.decode(tokens_list)
    
    # return a list with each token decoded (some can contain ##)
    def convert_tokens_to_strings(self, tokens_list):
        return self.tokenizer.convert_ids_to_tokens(tokens_list)

        
        
