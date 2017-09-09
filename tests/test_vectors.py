

import numpy as np

from mowgly.containers import Vectors


def naive_make_padded_matrix(array_of_arrays, lens, indices, fill_value):
    selected_arrays = array_of_arrays[indices]
    selected_lens = lens[indices]
    max_len = selected_lens.max()
    num_indices = len(indices)
    mat = np.empty((num_indices,max_len),dtype=array_of_arrays[0].dtype)
    mat.fill(fill_value)
    for i in range(num_indices): 
        mat[i,:selected_lens[i]] = selected_arrays[i]
    return mat


def naive_make_padded_matrix_with_start_indices(array_of_arrays, start_indices, indices, max_len, fill_value):
    num_indices = len(indices)
    mat = np.empty((num_indices,max_len),dtype=array_of_arrays[0].dtype)
    mat.fill(fill_value)
    selected_arrays = array_of_arrays[indices]
    for i in range(num_indices):
        arr = selected_arrays[i][start_indices[i]:]
        mat[i,:len(arr)] = arr[:max_len]
    return mat


def test():
    num_threads = 4
    print('\nTesting mowgly.containers.Vectors ...')
    N = 100000
    bs = 256
    list_of_1darrays = [np.random.randint(0,150,np.random.randint(500,2500),dtype=np.int16) for i in range(N)]
    indices = np.random.randint(0,N,bs).astype(np.int32)
    vecs = Vectors(list_of_1darrays)
    out = vecs.make_padded_matrix(indices, 0)
    array_of_arrays = np.asarray(list_of_1darrays)
    lens = np.asarray([len(arr) for arr in array_of_arrays])
    out_expect = naive_make_padded_matrix(array_of_arrays, lens, indices, 0)
    if np.all(out == out_expect):
        print('make_padded_matrix: OK')
    else:
        print('make_padded_matrix: FAIL')
    import time
    start = time.time()
    n_iters = 1000
    for i in range(n_iters):
        vecs.make_padded_matrix(indices, 0, num_threads)
    end = time.time()
    elapsed = end - start
    print('native/cython took {0:1.6} microseconds'.format( elapsed / n_iters * 1e6))
    start = time.time()
    n_iters = 1000
    for i in range(n_iters):
        naive_make_padded_matrix(array_of_arrays, lens, indices, 0)
    end = time.time()
    elapsed = end - start
    print('pure numpy took {0:1.6} microseconds'.format( elapsed / n_iters * 1e6))

    max_len = 1000
    start_indices = np.maximum((np.random.rand(bs)*vecs.lengths[indices]).astype(np.int32) - max_len ,0)
    out = vecs.make_padded_matrix_with_start_indices(indices, start_indices, max_len, 0)
    out_expect = naive_make_padded_matrix_with_start_indices(array_of_arrays, start_indices, indices, max_len, 0)
    if np.all(out == out_expect):
        print('make_padded_matrix_with_start_indices: OK')
    else:
        print('make_padded_matrix_with_start_indices: FAIL')
    n_iters = 1000
    start = time.time()
    for i in range(n_iters):
        vecs.make_padded_matrix_with_start_indices(indices, start_indices, max_len, 0, num_threads)
    end = time.time()
    elapsed = end - start
    print('native/cython took {0:0.6} microseconds'.format( elapsed / n_iters * 1e6))
    start = time.time()
    n_iters = 1000
    for i in range(n_iters):
        naive_make_padded_matrix_with_start_indices(array_of_arrays, start_indices, indices, max_len, 0)
    end = time.time()
    elapsed = end - start
    print('pure numpy took {0:0.6} microseconds'.format( elapsed / n_iters * 1e6))