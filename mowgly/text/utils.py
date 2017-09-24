

import numba as nb
import numpy as np


@nb.jit(nopython=True, nogil=True, parallel=True)
def get_padded_whitespace_indices(char_text_ids, unique_whitespace_ids, eos_id):
    """
    :param char_text_ids: 2d array of char IDs.
    :param unique_whitespace_ids: 1d array of whitespace IDs
    :param eos_id: end-of-sentence ID.
    :return: tuple of whitespace indices and the number of whitespaces in each row og char_text_ids.
    """
    assert char_text_ids.ndim == 2
    assert unique_whitespace_ids.ndim == 1
    num_whitespaces = np.empty( (char_text_ids.shape[0],), dtype=np.int64 )
    for i in nb.prange(char_text_ids.shape[0]):
        num_whitespaces[i] = 0
        for j in range(char_text_ids.shape[1]):
            if char_text_ids[i,j] == eos_id:
                break
            for k in range(unique_whitespace_ids.shape[0]):
                if char_text_ids[i,j] == unique_whitespace_ids[k]:
                    num_whitespaces[i] += 1
                    break
    max_num_whitespaces = num_whitespaces.max()
    whitespace_indices = np.empty( (char_text_ids.shape[0], max_num_whitespaces), dtype=np.int64 )
    for i in nb.prange(char_text_ids.shape[0]):
        t = 0
        for j in range(char_text_ids.shape[1]):
            if char_text_ids[i,j] == eos_id:
                break
            for k in range(unique_whitespace_ids.shape[0]):
                if char_text_ids[i, j] == unique_whitespace_ids[k]:
                    whitespace_indices[i,t] = j
                    t += 1
                    break
    return whitespace_indices.astype(np.int32), num_whitespaces.astype(np.int32)