//
// Created by Szg on 2018/6/16.
//

#include <stdio.h>
#include "array.h"

void array_copy(float* src, size_t size, float* dst){
    memcpy(dst, src, size);
}

void array_apply(const float *x, const int n, float(*func)(float), float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= func(x[i]);
    }
}
