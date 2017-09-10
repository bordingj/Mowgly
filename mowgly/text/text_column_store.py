

import joblib

import numpy as np
import pandas as pd

from mowgly.text import maps

from mowgly.text.specs import TextColumnSpec

class TextColumnStore(object):
    
    def __init__(self, corpus, column_specs, copy=True):
        
        assert isinstance(corpus, pd.DataFrame), "corpus must be a pandas.DataFrame"

        if isinstance(column_specs, TextColumnSpec):
            column_specs = [column_specs,]
            
        if not isinstance(column_specs, (list, tuple)):
            msg = "column_specs must be a list or a tuple of ColumnSpec's"
            raise TypeError(msg)
        
        if not isinstance(column_specs[0], TextColumnSpec):
            msg = "column_specs must be a list or a tuple of ColumnSpec's"
            raise TypeError(msg)

        if copy:
            self.corpus = corpus
        else:
            self.corpus = corpus.copy()    
        
        self.column_specs = column_specs

        self.corpus_arange = np.arange(len(self.corpus), dtype=np.int32)
        self.size = len(self.corpus_arange)

        self._cut_corpus()

        self._lowering_corpus()
        
        self._build_containers()
        
    def _cut_corpus(self):
        """ getting the length of texts and excluding those rows with too short texts """
        for spec in self.column_specs:
            new_col_name = spec.column_name + '_len'
            self.corpus[new_col_name] = [len(s) for s in self.corpus.loc[spec.column_name]]
        bool_ = self.corpus[new_col_name + '_len'] > spec.min_len
        for spec in self.column_specs[:-1]:
            new_col_name = spec.column_name + '_len'
            bool_ &= self.corpus[new_col_name] > spec.min_len
        self.corpus = self.corpus.loc[bool_]
        
    def _lowering_corpus(self):
        """ lowering texts """
        for spec in self.column_specs:
            if spec.lower:
                self.corpus[spec.column_name] = self.corpus[spec.column_name].str.lower()

    def _build_containers(self, n_jobs=4):
        """ building data containers for quick minibatch generation """
        for spec in self.column_specs:
            setattr(self, spec.column_name, 
                    maps.texts2numerical( self.corpus.loc[:,spec.column_name].values, spec, n_jobs) 
                    )
    
    def __getitem__(self, indices):
        out = {}
        for spec in self.column_specs:
            vecs = getattr(self, spec.column_name )
            if spec.fixed_length_subsample:
                start_indices = (spec.random_state.rand(len(indices))*vecs.lengths[indices]).astype(np.int32, copy=False)
                if not spec.uniform_start:
                    start_indices -= spec.max_len
                start_indices = np.maximum(start_indices, 0)
                out[spec.column_name] = vecs.make_padded_matrix_with_start_indices(
                                            indices, start_indices, spec.max_len, spec.eos_id)
            else:
                out[spec.column_name] = vecs.make_padded_matrix(indices, spec.eos_id)

        return out