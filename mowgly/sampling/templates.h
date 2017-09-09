#pragma once

#include <iterator>
#include <algorithm>

template<typename Iter, typename T>
Iter lower_bound_2(Iter first, Iter last, const T& val){
    typedef typename std::iterator_traits<Iter>::value_type      Value;
    typedef typename std::iterator_traits<Iter>::difference_type Distance;
 
    --last;
 
    while (first < last){
        
        Value shift_val = val - *first;
        Value shift_last = *last - *first;
 
        Distance len = std::distance(first, last);
        Distance part = len * shift_val / shift_last;
 
        Iter middle = first;
        std::advance(middle, part);
 
        if (val > *middle) {
            first = middle;
        } else {
            last = middle;
        }
 
        if (part == 0) {
            return std::lower_bound(first, last, val);
        }
 
        if (part == len) {
            return std::lower_bound(first, last, val);
        }
    }
 
    return first;
}

template<typename Iter, typename T>
void find_lower_bound_indices(const Iter start_iter, const Iter end_iter, 
                              const T* values, int32_t* indices, const int64_t len, int64_t num_threads){
    
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int32_t i = 0; i < len; i++){

        auto found_iter = lower_bound_2<Iter, T>(start_iter, end_iter, values[i]);
        indices[i] =  found_iter - start_iter;
    }
}  

