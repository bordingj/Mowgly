# distutils: language=c++
# cython: embedsignature=True
# cython: boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False, cdivision=True

from cython.parallel cimport prange

from libc.stdint cimport int32_t, int16_t, int64_t
                                       
import numpy as np
cimport numpy as np
from numpy cimport ndarray

cdef extern from "templates.h" nogil:
    void find_lower_bound_indices[Iter, T](const Iter start_iter, const Iter end_iter, 
                                           const T* values, int32_t* indices, const int64_t len, const int64_t num_threads)

ctypedef double* double_ptr


cdef ndarray[int32_t,ndim=1,mode='c'] random_sample_and_find_indices(ndarray[double,ndim=1,mode='c'] cdf,
                                                                     int64_t size, int64_t num_threads, object random_state):
    
    cdef ndarray[double,ndim=1,mode='c'] rands
    cdef ndarray[int32_t,ndim=1,mode='c'] indices = np.empty((size,),dtype=np.int32)
    
    if random_state is None:
        rands = np.random.rand(size)
    else:
        rands = random_state.rand(size)
        
    cdef double* start_iter = &cdf[0]
    cdef double* end_iter = &start_iter[len(cdf)]
    
    find_lower_bound_indices[double_ptr, double](start_iter, end_iter, &rands[0], &indices[0], size, num_threads)
    
    return indices
    
def random_choice_with_cdf(choices, size, cdf, replace=True, num_threads=4, random_state=None):
    
    assert choices.size == cdf.size
    assert choices.ndim == 1
        
    cdef ndarray indices = random_sample_and_find_indices(cdf, size, num_threads, random_state)
    
    if not replace:
        raise NotImplementedError

    return choices[indices]