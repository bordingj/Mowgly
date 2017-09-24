

import numba as nb
import numpy as np

from mowgly.containers.vectors import to_vectors
from mowgly.text.specs import WordColumnSpec, CharColumnSpec

import joblib


def chars2ids(chars, char2id, unknown_id, dtype=np.short, max_len=None,
              begin_char='\n', end_char='\n'):
    if max_len is not None:
        chars = chars[:max_len]
    chars = chars.strip()
    L = len(chars)
    out = np.empty((L+2,), dtype=dtype)
    out[0] = char2id[begin_char]
    out[1:L+1] = [char2id[c] if c in char2id else unknown_id for c in chars]
    out[L+1] = char2id[end_char]
    return out


def texts2numerical(texts, spec, n_jobs):
    out = {}
    print('building data-container for "{0}" column ... '.format(spec.column_name))
    if isinstance(spec, WordColumnSpec):
        raise NotImplementedError
    elif isinstance(spec, CharColumnSpec):
        out = {}
        out['level'] = 'char'
        list_of_id_vecs = joblib.Parallel(n_jobs=n_jobs,max_nbytes=None)(
                joblib.delayed(chars2ids)(s, char2id=spec.char2id, 
                        unknown_id=spec.unknown_id, max_len=spec.max_len, dtype=spec.dtype) for s in texts)
        out['id_vectors'] = to_vectors(list_of_id_vecs)
    else:
        msg = "Unrecognized spec"
        raise NotImplementedError(msg)
    return out
    
