//
// Created by Szg on 2018/6/16.
//

#ifndef METRIXNET_ARRAY_H
#define METRIXNET_ARRAY_H

#include <stdlib.h>
#include <string.h>

void array_copy(float*, size_t, float*);
void array_apply(const float *x, const int n, float(*func)(float), float *delta);

#endif //METRIXNET_ARRAY_H
