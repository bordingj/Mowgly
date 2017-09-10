
import numpy as np

from mowgly.containers.vectors import Vectors
from mowgly.text.specs import WordColumnSpec, CharColumnSpec

from sklearn.externals import joblib


def chars2ids(chars, char2id, unknown_id, dtype=np.short, begin_char='\n', end_char='\n'):
    chars = chars.strip()
    L = len(chars)
    out = np.empty((L+2,), dtype=dtype)
    out[0] = char2id[begin_char]
    out[1:L+1] = [char2id[c] if c in char2id else unknown_id for c in chars]
    out[L+1] = out[end_char]
    return out


def get_whitespace_indices_1d(seq, whitespace_ids):
    mask = np.in1d(seq, whitespace_ids)
    return np.argwhere(mask).ravel().astype(np.int32, copy=False)


def chartexts2numerical(texts, spec, n_jobs):

    print('building data-container for {0} char-ids... '.format(spec.column_name))
    out = {}
    out['level'] = 'char'
    list_of_id_vecs = joblib.Parallel(n_jobs=n_jobs,max_nbytes=None)(
            joblib.delayed(chars2ids)(s, char2id=spec.char2id, dtype=spec.dtype) for s in texts)
    out['vectors'] = Vectors(list_of_id_vecs)
    print('building data-container for {0} whicespace indices... '.format(spec.column_name)) 
    list_of_idx_vecs = [get_whitespace_indices_1d(seq, spec.whitespace_ids) for seq in list_of_id_vecs]
    out['whitespace_indices'] = np.asarray(list_of_idx_vecs)
    out['num_whitespaces'] =  np.asarray([len(v) for v in list_of_idx_vecs])
    return out


def texts2numerical(texts, spec, n_jobs):
    
    if isinstance(spec, WordColumnSpec):
        raise NotImplementedError
    elif isinstance(spec, CharColumnSpec):
        return chartexts2numerical(texts, spec, n_jobs)
    else:
        msg = "Unrecognized spec"
        raise NotImplementedError(msg)
    
