#pragma once

#include <vector>
#include <algorithm>
#include <omp.h>

using std::vector;

template<typename T>
void fill_padded_matrix( const vector<vector<T>>& vectors, 
                         const int32_t* indices, const int32_t& num_indices,
                         T* mat, const int32_t& num_cols, const T& fill_value,
                         const int64_t num_threads){
    
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int32_t i = 0; i < num_indices; i++){

        T* row = &mat[i * num_cols];

        int32_t idx = indices[i];
        const vector<T>& vec = vectors[idx];

        const int32_t vec_len = vec.size();

        if (num_cols >= vec_len){

            std::copy(vec.begin(), vec.end(), row);

            std::fill(&row[vec_len], &row[num_cols], fill_value);

        }
    }
}

template<typename T>
void fill_matrix_with_start_indices( const vector<vector<T>>& vectors, 
                                     const int32_t* indices, const int32_t& num_indices,
                                     const int32_t* start_indices,
                                     T* mat, const int32_t& num_cols, const T& fill_value,
                                     const int64_t num_threads){
    
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int32_t i = 0; i < num_indices; i++){

        T* row = &mat[i * num_cols];

        int32_t idx = indices[i];
        const vector<T>& vec = vectors[idx];

        const int32_t vec_len = vec.size();
        const int32_t start_idx = start_indices[i];

        const int32_t vec_elements = std::min( vec_len - start_idx, num_cols);

        std::copy(&vec[start_idx], &vec[start_idx+vec_elements], row);

        if (num_cols > vec_elements){

            std::fill(&row[vec_elements], &row[num_cols], fill_value);

        }
    }
}


