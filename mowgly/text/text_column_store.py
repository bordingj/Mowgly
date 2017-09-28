
import numpy as np
import pandas as pd

from mowgly.text import maps
from mowgly.containers.vectors import (Int16Vectors,
                                       Int32Vectors, 
                                       Int64Vectors, 
                                       Float32Vectors)

from mowgly.text.specs import TextColumnSpec


class TextColumnStore(object):
    
    def __init__(self, corpus, column_specs, n_jobs=4, copy=True):
        
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
        
        self._column_specs_list = column_specs


        self._cut_corpus()

        self.size = len(self.corpus)

        self._lowering_corpus()
        
        self.n_jobs = n_jobs
        
        self._build_containers(n_jobs=self.n_jobs)
        
        self.size = len(self.corpus)
        
    def _cut_corpus(self):
        """ getting the length of texts and excluding those rows with too short texts """
        for spec in self._column_specs_list:
            new_col_name = spec.column_name + '_len'
            self.corpus[new_col_name] = [len(s) for s in self.corpus[spec.column_name]]
        bool_ = self.corpus[new_col_name] > spec.min_len
        for spec in self._column_specs_list[:-1]:
            new_col_name = spec.column_name + '_len'
            bool_ &= self.corpus[new_col_name] > spec.min_len
        self.corpus = self.corpus.loc[bool_]
        
    def _lowering_corpus(self):
        """ lowering texts """
        for spec in self._column_specs_list:
            if spec.lower:
                self.corpus.loc[:,spec.column_name] = self.corpus[spec.column_name].str.lower()

    def _build_containers(self, n_jobs=4):
        """ building data containers for quick minibatch generation """
        columns = {}
        column_specs = {}
        for spec in self._column_specs_list:
            columns[spec.column_name] = maps.texts2numerical( self.corpus.loc[:,spec.column_name].values, spec, n_jobs)
            column_specs[spec.column_name] = spec
        self._columns = columns
        self._column_specs = column_specs

    @property
    def columns(self):
        return self._columns.copy()

    @property
    def specs(self):
        return self._column_specs.copy()
    
    def __getitem__(self, indices):
        out = {}
        for column_name, spec in self._column_specs.items():
            d_in = self._columns[column_name]
            d_out = {}
            for key, vecs in d_in.items():
                if key == 'id_vectors':
                    if spec.fixed_length_subsample:
                        start_indices = (spec.random_state.rand(len(indices))*vecs.lengths[indices]
                                        ).astype(np.int32, copy=False)
                        start_indices -= spec.max_len//2 if spec.uniform_start else spec.max_len
                        start_indices = np.maximum(start_indices, 0)
                        arr = vecs.make_padded_matrix_with_start_indices(
                                                    indices, start_indices, spec.sample_len, spec.eos_id)
                    else:
                        arr = vecs.make_padded_matrix(indices, spec.eos_id)
                    d_out[key] = arr
                elif key == 'level':
                    d_out[key] = vecs
                else:
                    raise RuntimeError("something is wrong")
                out[column_name] = d_out

        return out