# distutils: language=c++
# cython: embedsignature=True
# cython: boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False, cdivision=True

from libc.stdint cimport int32_t, int16_t, int64_t
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np
from numpy cimport ndarray

cdef extern from "templates.h" nogil:
    void fill_padded_matrix[T]( const vector[vector[T]]&, 
                             const int32_t* indices, const int32_t& num_indices,
                             T* mat, const int32_t& num_cols, const T& fill_value,
                             const int64_t num_threads)
    
    void fill_matrix_with_start_indices[T]( const vector[vector[T]]& vectors, 
                                         const int32_t* indices, const int32_t& num_indices,
                                         const int32_t* start_indices,
                                         T* mat, const int32_t& num_cols, const T& fill_value,
                                         const int64_t num_threads)
        
class Vectors(object):
    
    def __new__(self, list list_of_1darrays):
        cdef ndarray first_array = list_of_1darrays[0]
        if first_array.dtype == np.int16:
            return Int16Vectors(list_of_1darrays)
        elif first_array.dtype == np.int32:
            return Int32Vectors(list_of_1darrays)
        elif first_array.dtype == np.int64:
            return Int64Vectors(list_of_1darrays)
        elif first_array.dtype == np.float32:
            return Float32Vectors(list_of_1darrays)
        else:
            raise NotImplementedError("Unsupported dtype of 1darrays")
        

cdef class Int16Vectors(object):
    
    """
    This extention type implements a container for a list of variable length 1d int16 ndarrays.
    """
    cdef readonly:
        int32_t num_vectors
        vector[vector[int16_t]] vectors
        int32_t[::1] lengths_buffer
        ndarray lengths

    @staticmethod
    cdef vector[int16_t] ndarray2vector(int16_t[::1] arr) except +:
        cdef vector[int16_t] vec;
        vec.resize(len(arr));
        for i in range(len(arr)):
            vec[i] = arr[i];
        return vec

    def __cinit__(self, list list_of_1darrays):
        """ 
        Args:
            list of 1d ndarrays of variable length
        """
        # get number of vectors
        self.num_vectors = len(list_of_1darrays)
        if self.num_vectors < 1:
            raise ValueError('list must contain atleast 1 vector')
        # check vectors, copy them and get their lengths and locations in memory
        self.vectors.resize(self.num_vectors)
        self.lengths = np.empty((self.num_vectors,), dtype=np.int32)
        self.lengths_buffer = self.lengths
        for i in range(self.num_vectors):
            self.vectors[i] = Int16Vectors.ndarray2vector(list_of_1darrays[i])
            self.lengths_buffer[i] = self.vectors[i].size()        
    
    cpdef ndarray[int16_t,ndim=2,mode='c'] make_padded_matrix(self, 
                                                 ndarray[int32_t,ndim=1,mode='c'] indices,
                                                 int16_t fill_value, int64_t num_threads=4) except +:
        cdef int32_t max_len = self.lengths[indices].max()
        cdef int32_t num_indices = len(indices)
        cdef ndarray[int16_t,ndim=2,mode='c'] mat = np.empty((num_indices, max_len), dtype=np.int16)
        fill_padded_matrix[int16_t]( self.vectors, <int32_t*>&indices[0], num_indices, <int16_t*>&mat[0,0], 
                           max_len, fill_value, num_threads)
        return mat
        
    cpdef ndarray[int16_t,ndim=2,mode='c'] make_padded_matrix_with_start_indices(self, 
                                                 ndarray[int32_t,ndim=1,mode='c'] indices,
                                                 ndarray[int32_t,ndim=1,mode='c'] start_indices,
                                                 int32_t max_len,
                                                 int16_t fill_value, int64_t num_threads=4) except +:
        
        cdef int32_t num_indices = len(indices)
        cdef ndarray[int16_t,ndim=2,mode='c'] mat = np.empty((num_indices, max_len), dtype=np.int16)
        fill_matrix_with_start_indices[int16_t]( self.vectors, <int32_t*>&indices[0], num_indices,
                                     <int32_t*>&start_indices[0],<int16_t*>&mat[0,0], max_len, fill_value, num_threads)
        return mat
    

cdef class Float32Vectors(object):
    
    """
    This extention type implements a container for a list of variable length 1d float32 ndarrays.
    """
    cdef readonly:
        int32_t num_vectors
        vector[vector[float]] vectors
        int32_t[::1] lengths_buffer
        ndarray lengths

    @staticmethod
    cdef vector[float] ndarray2vector(float[::1] arr) except +:
        cdef vector[float] vec;
        vec.resize(len(arr));
        for i in range(len(arr)):
            vec[i] = arr[i];
        return vec

    def __cinit__(self, list list_of_1darrays):
        """ 
        Args:
            list of 1d ndarrays of variable length
        """
        # get number of vectors
        self.num_vectors = len(list_of_1darrays)
        if self.num_vectors < 1:
            raise ValueError('list must contain atleast 1 vector')
        # check vectors, copy them and get their lengths and locations in memory
        self.vectors.resize(self.num_vectors)
        self.lengths = np.empty((self.num_vectors,), dtype=np.int32)
        self.lengths_buffer = self.lengths
        for i in range(self.num_vectors):
            self.vectors[i] = Float32Vectors.ndarray2vector(list_of_1darrays[i])
            self.lengths_buffer[i] = self.vectors[i].size()
        
    
    cpdef ndarray[float,ndim=2,mode='c'] make_padded_matrix(self, 
                                                 ndarray[int32_t,ndim=1,mode='c'] indices,
                                                 float fill_value, int64_t num_threads=4) except +:
        cdef int32_t max_len = self.lengths[indices].max()
        cdef int32_t num_indices = len(indices)
        cdef ndarray[float,ndim=2,mode='c'] mat = np.empty((num_indices, max_len), dtype=np.float32)
        fill_padded_matrix[float]( self.vectors, <int32_t*>&indices[0], num_indices, <float*>&mat[0,0], 
                           max_len, fill_value, num_threads)
        return mat
        
    cpdef ndarray[float,ndim=2,mode='c'] make_padded_matrix_with_start_indices(self, 
                                                 ndarray[int32_t,ndim=1,mode='c'] indices,
                                                 ndarray[int32_t,ndim=1,mode='c'] start_indices,
                                                 int32_t max_len,
                                                 float fill_value, int64_t num_threads=4) except +:
        
        cdef int32_t num_indices = len(indices)
        cdef ndarray[float,ndim=2,mode='c'] mat = np.empty((num_indices, max_len), dtype=np.float32)
        fill_matrix_with_start_indices[float]( self.vectors, <int32_t*>&indices[0], num_indices,
                                     <int32_t*>&start_indices[0],<float*>&mat[0,0], max_len, fill_value, num_threads)
        return mat
    

cdef class Int32Vectors(object):
    
    """
    This extention type implements a container for a list of variable length 1d int32 ndarrays.
    """
    cdef readonly:
        int32_t num_vectors
        vector[vector[int32_t]] vectors
        int32_t[::1] lengths_buffer
        ndarray lengths

    @staticmethod
    cdef vector[int32_t] ndarray2vector(int32_t[::1] arr) except +:
        cdef vector[int32_t] vec;
        vec.resize(len(arr));
        for i in range(len(arr)):
            vec[i] = arr[i];
        return vec

    def __cinit__(self, list list_of_1darrays):
        """ 
        Args:
            list of 1d ndarrays of variable length
        """
        # get number of vectors
        self.num_vectors = len(list_of_1darrays)
        if self.num_vectors < 1:
            raise ValueError('list must contain atleast 1 vector')
        # check vectors, copy them and get their lengths and locations in memory
        self.vectors.resize(self.num_vectors)
        self.lengths = np.empty((self.num_vectors,), dtype=np.int32)
        self.lengths_buffer = self.lengths
        for i in range(self.num_vectors):
            self.vectors[i] = Int32Vectors.ndarray2vector(list_of_1darrays[i])
            self.lengths_buffer[i] = self.vectors[i].size()
        
    
    cpdef ndarray[int32_t,ndim=2,mode='c'] make_padded_matrix(self, 
                                                 ndarray[int32_t,ndim=1,mode='c'] indices,
                                                 int32_t fill_value, int64_t num_threads=4) except +:
        cdef int32_t max_len = self.lengths[indices].max()
        cdef int32_t num_indices = len(indices)
        cdef ndarray[int32_t,ndim=2,mode='c'] mat = np.empty((num_indices, max_len), dtype=np.int32)
        fill_padded_matrix[int32_t]( self.vectors, <int32_t*>&indices[0], num_indices, <int32_t*>&mat[0,0], 
                           max_len, fill_value, num_threads)
        return mat
        
    cpdef ndarray[int32_t,ndim=2,mode='c'] make_padded_matrix_with_start_indices(self, 
                                                 ndarray[int32_t,ndim=1,mode='c'] indices,
                                                 ndarray[int32_t,ndim=1,mode='c'] start_indices,
                                                 int32_t max_len,
                                                 int32_t fill_value, int64_t num_threads=4) except +:
        
        cdef int32_t num_indices = len(indices)
        cdef ndarray[int32_t,ndim=2,mode='c'] mat = np.empty((num_indices, max_len), dtype=np.int32)
        fill_matrix_with_start_indices[int32_t]( self.vectors, <int32_t*>&indices[0], num_indices,
                                     <int32_t*>&start_indices[0],<int32_t*>&mat[0,0], max_len, fill_value, num_threads)
        return mat
    

cdef class Int64Vectors(object):
    
    """
    This extention type implements a container for a list of variable length 1d int64 ndarrays.
    """
    cdef readonly:
        int32_t num_vectors
        vector[vector[int64_t]] vectors
        int32_t[::1] lengths_buffer
        ndarray lengths

    @staticmethod
    cdef vector[int64_t] ndarray2vector(int64_t[::1] arr) except +:
        cdef vector[int64_t] vec;
        vec.resize(len(arr));
        for i in range(len(arr)):
            vec[i] = arr[i];
        return vec

    def __cinit__(self, list list_of_1darrays):
        """ 
        Args:
            list of 1d ndarrays of variable length
        """
        # get number of vectors
        self.num_vectors = len(list_of_1darrays)
        if self.num_vectors < 1:
            raise ValueError('list must contain atleast 1 vector')
        # check vectors, copy them and get their lengths and locations in memory
        self.vectors.resize(self.num_vectors)
        self.lengths = np.empty((self.num_vectors,), dtype=np.int32)
        self.lengths_buffer = self.lengths
        for i in range(self.num_vectors):
            self.vectors[i] = Int64Vectors.ndarray2vector(list_of_1darrays[i])
            self.lengths_buffer[i] = self.vectors[i].size()
        
    
    cpdef ndarray[int64_t,ndim=2,mode='c'] make_padded_matrix(self, 
                                                 ndarray[int32_t,ndim=1,mode='c'] indices,
                                                 int32_t fill_value, int64_t num_threads=4) except +:
        cdef int32_t max_len = self.lengths[indices].max()
        cdef int32_t num_indices = len(indices)
        cdef ndarray[int64_t,ndim=2,mode='c'] mat = np.empty((num_indices, max_len), dtype=np.int64)
        fill_padded_matrix[int64_t]( self.vectors, <int32_t*>&indices[0], num_indices, <int64_t*>&mat[0,0], 
                           max_len, fill_value, num_threads)
        return mat
        
    cpdef ndarray[int64_t,ndim=2,mode='c'] make_padded_matrix_with_start_indices(self, 
                                                 ndarray[int32_t,ndim=1,mode='c'] indices,
                                                 ndarray[int32_t,ndim=1,mode='c'] start_indices,
                                                 int32_t max_len,
                                                 int32_t fill_value, int64_t num_threads=4) except +:
        
        cdef int32_t num_indices = len(indices)
        cdef ndarray[int64_t,ndim=2,mode='c'] mat = np.empty((num_indices, max_len), dtype=np.int64)
        fill_matrix_with_start_indices[int64_t]( self.vectors, <int32_t*>&indices[0], num_indices,
                                     <int32_t*>&start_indices[0],<int64_t*>&mat[0,0], max_len, fill_value, num_threads)
        return mat
