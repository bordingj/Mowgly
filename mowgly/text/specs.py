

from mowgly.text.languages import danish, english

import numpy as np


class TextColumnSpec(object):

    def __init__(self, **kwargs):
        raise NotImplementedError
    
    def add_vocab_and_mappings(self):
        raise NotImplementedError



class WordColumnSpec(TextColumnSpec):
    pass
    


from mowgly.utils import check_random_state


class CharColumnSpec(TextColumnSpec):
    
    def __init__(self, column_name, language, lower=True, min_len=1, max_len=2000, 
                 fixed_length_subsample=False, uniform_start=False, sample_len=200,
                 random_state=None, dtype=np.int16):
        
        self.column_name = column_name
        self.language = language
        self.lower = lower
        self.min_len = min_len
        self.max_len = max_len
        self.sample_len = sample_len
        self.fixed_length_subsample = fixed_length_subsample
        self.dtype = dtype
        self.random_state = check_random_state(random_state)
        self.uniform_start = uniform_start
        
        self.add_vocab_and_mappings()
        
    def add_vocab_and_mappings(self):
        
        if self.language == 'danish':
            lang = danish
        elif self.language == 'english':
            lang = english
        else:
            raise NotImplementedError
        if self.lower:
            vocab = lang.LOWER_CHAR_VOCAB
            self.whitespace_chars = lang.WHITESPACES
            self.unknown_char = lang.UNKNOWN_CHAR
        else:
            raise NotImplementedError
            
        self.char2id = dict(zip(vocab, range(len(vocab))))
        self.id2char = {id: char for char, id in self.char2id.items()}
        self.vocab = set(vocab)
        self.vocabsize = len(vocab)
        self.unknown_id = self.char2id[self.unknown_char]
        self.eos_char = danish.EOS_CHAR
        self.eos_id = self.char2id[self.eos_char]
        self.whitespace_ids = np.asarray([self.char2id[c] for c in self.whitespace_chars], 
                                         dtype=self.dtype)
