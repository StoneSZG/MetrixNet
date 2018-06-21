//
// Created by Szg on 2018/6/12.
//

#ifndef METRIXNET_MATRIX_H
#define METRIXNET_MATRIX_H

#include <stdlib.h>

typedef struct matrix {
    int row;
    int col;
    float *values;
}matrix, Matrix, *pMatrix, *pmatrix;

Matrix make_matrix(size_t, size_t);
Matrix make_matrix_normal(size_t, size_t);
Matrix make_matrix_ones(size_t, size_t);
Matrix make_matrix_zeros(size_t, size_t);
Matrix make_matrix_eyes(size_t);
Matrix make_matrix_initializer(size_t, size_t, float);
Matrix make_matrix_incre(size_t, size_t, float f);

void matrix_normal(pMatrix);

void matrix_copy(pMatrix, pMatrix);

void matrix_mean(pMatrix, pMatrix, int);
//void matrix_mean_row(pMatrix, pMatrix);
//void matrix_mean_col(pMatrix m, pMatrix v);

float matrix_sum_all(pMatrix);
void matrix_sum(pMatrix, pMatrix, int);
//void matrix_sum_row(pMatrix, pMatrix);
//void matrix_sum_col(pMatrix m, pMatrix v);

void matrix_fill(pMatrix, int);

void free_matrix(pmatrix);

//void print_matrix(matrix);
void print_matrix(pmatrix);

void T(pMatrix);

void matrix_transpose(pMatrix);

Matrix matrix_dot(pMatrix, pMatrix);

void matrix_add(pMatrix, pMatrix);

void matmul_additive(pMatrix, pMatrix, pMatrix);

void matrix_sub(pMatrix, pMatrix);

void matmul_subtract(pMatrix, pMatrix, pMatrix);

void matrix_add_vector(pMatrix, pMatrix, int);

void matrix_sub_vector(pMatrix, pMatrix, int);

void matrix_div_vector(pMatrix, pMatrix, int);

void matrix_additive_vector(pMatrix, pMatrix, pMatrix);

void matrix_matmul(pMatrix, pMatrix, pMatrix);

void matrix_mul(pMatrix, pMatrix);

void matrix_multiply(pMatrix, pMatrix, pMatrix);

void matrix_reshape(pMatrix, int, int);

void matrix_map(pMatrix,  float(*)(float));

void matrix_mapfunc(pMatrix, pMatrix, float(*)(float));

void matrix_scale(pMatrix, float);

void matrix_set(pMatrix, int, int, float);



#endif //METRIXNET_MATRIX_H
