//
// Created by Szg on 2018/6/12.
//

#ifndef METRIXNET_UTILS_H
#define METRIXNET_UTILS_H

#include "matrix.h"

#define PI 3.1415926
#define TWO_PI 6.2831853071795864769252866f

float normal_distribution(float, float);
float uniform_distribution(float min, float max);
float stand_uniform();
float stand_normal();

void matrix_softmax(pMatrix);

void im2col(float*, int, int, int, int, int, int, float*);
float im2col_get(float*, int, int, int, int, int, int);
void col2im_set(float*, int, int, int, int, int, int, float);
void col2im(float*, int,  int,  int, int, int, int, float*);

#endif //METRIXNET_UTILS_H
