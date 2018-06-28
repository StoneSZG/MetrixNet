//
// Created by Szg on 2018/6/12.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "utils.h"
#include "matrix.h"

Matrix make_matrix(size_t row, size_t col){
    size_t size = row * col;
    Matrix m = {0};
    float *values = malloc(size * sizeof(float));
    if(NULL == values){
        fprintf(stderr, "malloc failed!\n");
        exit(0);
    }
    m.row = (int)row;
    m.col = (int)col;
    m.values = values;
    return m;
}

Matrix make_matrix_normal_mean_std(size_t row, size_t col, float mean, float std){
    matrix m = make_matrix(row, col);
    for(size_t i = 0; i < row * col; i++){
        m.values[i] = normal_distribution(mean, std);
    }

    return m;
}

Matrix make_matrix_normal(size_t row, size_t col){
    matrix m = make_matrix(row, col);
    for(size_t i = 0; i < row * col; i++){
        m.values[i] = normal_distribution(0, 0.1);
    }

    return m;
}

Matrix make_matrix_ones(size_t row, size_t col){
    matrix m = make_matrix(row, col);
//    printf("make_matrix_ones m row:%d, col:%d\n", m.row, m.col);
    for(size_t i = 0; i < row * col; i++){
        m.values[i] = 1.0;
    }

    return m;

}

Matrix make_matrix_zeros(size_t row, size_t col){
    size_t size = row * col;
    Matrix m = {0};
//    float *values = malloc(size * sizeof(float));
    float *values = calloc(size, sizeof(float));
    if(NULL == values){
        fprintf(stderr, "malloc failed!\n");
        exit(0);
    }
    m.row = (int)row;
    m.col = (int)col;
    m.values = values;
    return m;
}

Matrix make_matrix_initializer(size_t row, size_t col, float data){
    matrix m = make_matrix(row, col);

    for(int i = 0; i < row * col; i++){
        m.values[i] = data;
    }

    return m;
}

Matrix make_matrix_eyes(size_t row){
    matrix m = make_matrix_zeros(row, row);
    for(int i =0; i < row; i++){
        m.values[i * row + i] = 1;
    }

    return m;
}

Matrix make_matrix_incre(size_t row, size_t col, float f){
    matrix m = make_matrix(row, col);
//    print_matrix(m);

    for(int i = 0; i < row * col; i++){
        m.values[i] = i + f;
    }
//    print_matrix(m);
    return m;
}

void print_matrix(pmatrix m){
    int row = m->row;
    int col = m->col;
    printf("%d X %d Matrix:\n",row, col);

    printf("|");
    for(int j = 0; j < 12 * col; ++j) printf("-");
    printf("|\n");
    for(int i = 0; i < row; ++i){
        printf("| ");
        for(int j = 0; j < col; ++j){
            printf("%10.8f |", m->values[i * col + j]);
        }
        printf("\n");
        printf("|");
        for(int j = 0; j < 12 * col; ++j) printf("-");
        printf("|\n");
    }
}

void free_matrix(pMatrix m){
//    printf("free_matrix:\n");
    free(m->values);
//    printf("free success:\n");
}

void matrix_fill(pMatrix m, float value){

    int row = m->row;
    int col = m->col;
    for(int i = 0;i < row * col; i++){
        m->values[i] = value;
    }
}

void matrix_mean_all(pMatrix m, pMatrix v){
    int row = m->row;
    int col = m->col;
    float sum = 0;
    for(int i = 0; i < row * col; i++){
        sum += m->values[i];
    }
    v->values[0] = sum / (row * col);
}

void matrix_mean_row(pMatrix m, pMatrix v){
    int row = m->row;
    int col = m->col;
    float sum = 0;
    for(int i = 0; i < col ; i++){
        sum = 0;
        for(int j = 0;j < row; j++)
            sum += m->values[j * col + i];
        v->values[i] = sum / row;
    }

}

void matrix_mean_col(pMatrix m, pMatrix v){
    int row = m->row;
    int col = m->col;
    float sum = 0;
    for(int i = 0; i < row ; i++){
        sum = 0;
        for(int j = 0;j < col; j++)
            sum += m->values[i * col + j];
        v->values[i] = sum / col;
    }

}

void matrix_mean(pMatrix m, pMatrix v, int dim){
    if(dim == 0){
        matrix_mean_row(m, v);
    }else if(dim == 1){
        matrix_mean_col(m, v);
    }else{
        matrix_mean_all(m, v);
    }
}

float matrix_sum_all(pMatrix m){
    int row = m->row;
    int col = m->col;
    float sum = 0;
    for(int i = 0; i < row * col; i++){
        sum += m->values[i];
    }
    return sum;
}

void matrix_sum_row(pMatrix m, pMatrix v){
    int row = m->row;
    int col = m->col;
    float sum = 0;
    for(int i = 0; i < col ; i++){
        sum = 0;
        for(int j = 0;j < row; j++)
            sum += m->values[j * col + i];
        v->values[i] = sum ;
    }

}

void matrix_sum_col(pMatrix m, pMatrix v){
    int row = m->row;
    int col = m->col;
    float sum = 0;
    for(int i = 0; i < row ; i++){
        sum = 0;
        for(int j = 0;j < col; j++)
            sum += m->values[i * col + j];
        v->values[i] = sum;
    }

}

void matrix_sum(pMatrix m, pMatrix v, int dim){
    if(dim == 0){
        matrix_sum_row(m, v);
    }else{
        matrix_sum_col(m, v);
    }
}

void T(pMatrix m){
//    pMatrix ma_copy = matrix_copy(m);
    return matrix_transpose(m);
}

void matrix_transpose(pMatrix m){
    int row = m->row;
    int col = m->col;
//    matrix data = make_matrix_zeros(col, row);
    float *data = malloc(col * row * sizeof(float));

    for(int j = 0; j < col ;j ++){
        for(int i = 0; i < row; i++){
            data[i + j * row] = m->values[i * col + j];
        }

    }
//    matrix_copy(&(data), m);
    for(int i = 0;i < row * col; i++){
        m->values[i] = data[i];
    }
//    memcpy(m->values, data, row * col* sizeof(float));
    m->row = col;
    m->col = row;
//    free_matrix(&(data));
    free(data);
}

void matrix_copy(const pMatrix ma, pMatrix mb){
    if (ma->row * ma->col != mb->row * mb->col){
        fprintf(stderr, "matrix_copy() Matrix a shape must be equal Matrix b shape!\n");
        exit(-1);
    }
    int col = ma->col;
    int row = ma->row;
//    for (int i = 0; i< row * col ;i++){
//        mb->values[i] = ma->values[i];
//    }
    memcpy(mb->values, ma->values, row * col * sizeof(float));
    mb->row = row;
    mb->col = col;
}

Matrix matrix_dot(pMatrix ma, pMatrix mb){
//    assert (ma->col == mb->row);
    if (ma->col != mb->row){
        fprintf(stderr, "matrix_dot() Matrix a col must be equal Matrix b row!\n");
        exit(-1);
    }
    Matrix m_res = make_matrix_zeros(ma->row, mb->col);

    int row = ma->row;
    int col = mb->col;
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j ++){
            float res = 0.0;
            for(int k = 0; k < ma->col; k++)
                res += ma->values[i * ma->col + k] * mb->values[j + col * k];
            m_res.values[i * col + j] = res;
        }
    }

    return m_res;

}

void matrix_add(pMatrix ma, const pMatrix mb){
//    assert ((ma->row == mb->row) && (ma->col == mb->col));
    if ((ma->row != mb->row) || (ma->col != mb->col)){
        fprintf(stderr, "matrix_add() Matrix a shape must be equal Matrix b shape!\n");
        exit(-1);
    }
    int row = ma->row;
    int col = ma->col;
//    Matrix m_res = make_matrix_ones(row, col);
    for (int i = 0;i < row * col; i++){
        ma->values[i] += mb->values[i];
    }

}

void matrix_additive(pMatrix ma, pMatrix mb, pMatrix m_res){
//    assert ((ma->row == mb->row) && (ma->col == mb->col));
    if ((ma->row != mb->row) || (ma->col != mb->col)){
        fprintf(stderr, "matrix_additive() Matrix a shape must be equal Matrix b shape!\n");
        exit(-1);
    }
    int row = ma->row;
    int col = ma->col;
//    Matrix m_res = make_matrix_ones(row, col);
    for (int i = 0;i < row * col; i++){
        m_res->values[i] = ma->values[i] + mb->values[i];
    }

}

void matrix_sub(pMatrix ma, pMatrix mb){
    if ((ma->row != mb->row) || (ma->col != mb->col)){
        fprintf(stderr, "matrix_sub() Matrix a shape must be equal Matrix b shape!\n");
        exit(-1);
    }
    int row = ma->row;
    int col = ma->col;
//    Matrix m_res = make_matrix_ones(row, col);
    for (int i = 0;i < row * col; i++){
        ma->values[i] -= mb->values[i];
    }
}

void matmul_subtract(pMatrix ma, pMatrix mb, pMatrix m_res){
    if ((ma->row != mb->row) || (ma->col != mb->col)){
        fprintf(stderr, "matmul_subtract() Matrix a shape must be equal Matrix b shape!\n");
        exit(-1);
    }
    int row = ma->row;
    int col = ma->col;

    for (int i = 0;i < row * col; i++){
        m_res->values[i] = ma->values[i] - mb->values[i];
    }
}

void matrix_add_vector_col(pMatrix m, pMatrix v){
    if ((m->col != v->col)){
        fprintf(stderr, "matrix_add_vector() Matrix m col must be equal Vector v col!\n");
        exit(-1);
    }
    if ((v->row != 1)){
        fprintf(stderr, "matrix_add_vector() Vector v row must be equal 1!\n");
        exit(-1);
    }
    int row = m->row;
    int col = m->col;
    for(int i = 0;i < row * col; i++){
        m->values[i] += v->values[i % col];
    }
}

void matrix_add_vector_row(pMatrix m, pMatrix v){
    if ((m->row != v->col)){
        fprintf(stderr, "matrix_add_vector_row() Matrix m row must be equal Vector v col!\n");
        exit(-1);
    }
    if ((v->row != 1)){
        fprintf(stderr, "matrix_add_vector_row() Vector v row must be equal 1!\n");
        exit(-1);
    }
    int row = m->row;
    int col = m->col;
    for(int i = 0;i < row * col; i++){
        m->values[i] += v->values[i / col];
    }
}

void matrix_add_vector(pMatrix m, pMatrix v, int dim){
    if (dim == 0)
        matrix_add_vector_row(m, v);
    else if(dim == 1)
        matrix_add_vector_col(m, v);
    else{
        fprintf(stderr, "Matrix add vector dim should be equal 0 or 1\n");
        exit(-1);
    }
}

void matrix_sub_vector_row(pMatrix m, pMatrix v){
    if ((m->row != v->col)){
        fprintf(stderr, "matrix_sub_vector() Matrix m col must be equal Vector v col!\n");
        exit(-1);
    }
    if ((v->row != 1)){
        fprintf(stderr, "matrix_sub_vector() Vector v row must be equal 1!\n");
        exit(-1);
    }
    int row = m->row;
    int col = m->col;
    for(int i = 0;i < row * col; i++){
        m->values[i] -= v->values[i / col];
    }
}

void matrix_sub_vector_col(pMatrix m, pMatrix v){
    if ((m->col != v->col)){
        fprintf(stderr, "matrix_sub_vector() Matrix m col must be equal Vector v col!\n");
        exit(-1);
    }
    if ((v->row != 1)){
        fprintf(stderr, "matrix_sub_vector() Vector v row must be equal 1!\n");
        exit(-1);
    }
    int row = m->row;
    int col = m->col;
    for(int i = 0;i < row * col; i++){
        m->values[i] -= v->values[i % col];
    }
}

void matrix_sub_vector(pMatrix m, pMatrix v, int dim){
    if (dim == 0)
        matrix_sub_vector_row(m, v);
    else if(dim == 1)
        matrix_sub_vector_col(m, v);
    else{
        fprintf(stderr, "Matrix div vector dim should be equal 0 or 1\n");
        exit(-1);
    }
}

void matrix_div_vector_row(pMatrix m, pMatrix v){
    if ((m->row != v->col)){
        fprintf(stderr, "matrix_div_vector() Matrix m row must be equal Vector v col!\n");
        exit(-1);
    }
    if ((v->row != 1)){
        fprintf(stderr, "matrix_div_vector() Vector v row must be equal 1!\n");
        exit(-1);
    }
    int row = m->row;
    int col = m->col;
    for(int i = 0;i < row * col; i++){
        m->values[i] /= v->values[i / col];
    }
}

void matrix_div_vector_col(pMatrix m, pMatrix v){
    if ((m->row != v->col)){
        fprintf(stderr, "matrix_div_vector() Matrix m row must be equal Vector v col!\n");
        exit(-1);
    }
    if ((v->row != 1)){
        fprintf(stderr, "matrix_div_vector() Vector v row must be equal 1!\n");
        exit(-1);
    }
    int row = m->row;
    int col = m->col;
    for(int i = 0;i < row * col; i++){
        m->values[i] /= v->values[i % col];
    }
}

void matrix_div_vector(pMatrix m, pMatrix v, int dim){
    if (dim == 0)
        matrix_div_vector_row(m, v);
    else if(dim == 1)
        matrix_div_vector_col(m, v);
    else{
        fprintf(stderr, "Matrix div vector dim should be equal 0 or 1\n");
        exit(-1);
    }
}

void matrix_additive_vector(pMatrix m, pMatrix v, pMatrix m_res){
    if ((m->col != v->col)){
        fprintf(stderr, "matrix_additive_vector() Matrix m col must be equal Vector v col!\n");
        exit(-1);
    }
    if ((v->row != 1)){
        fprintf(stderr, "matrix_additive_vector() Vector v row must be equal 1!\n");
        exit(-1);
    }
    int row = m->row;
    int col = m->col;
    for(int i = 0;i < row * col; i++){
        m_res->values[i] = m->values[i] + v->values[i % col];
    }
}

void matrix_matmul(pMatrix ma, pMatrix mb, pMatrix m_res){
    if (ma->col != mb->row){
        fprintf(stderr, "matrix_matmul() Matrix a col must be equal Matrix b row!\n");
        exit(-1);
    }

    int row = ma->row;
    int col = mb->col;
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j ++){
            float res = 0.0;
            for(int k = 0; k < ma->col; k++)
                res += ma->values[i * ma->col + k] * mb->values[j + col * k];
            m_res->values[i * col + j] = res;
        }
    }

}

void matrix_mul(pMatrix ma, pMatrix mb){
    if ((ma->row != mb->row) || (ma->col != mb->col)){
        fprintf(stderr, "matrix_mul() Matrix a shape must be equal Matrix b shape!\n");
        exit(-1);
    }

    int row = ma->row;
    int col = ma->col;

//    assert((m_res->row == row) && (m_res->col == col));

    for(int i = 0; i < row * col; i++){
        ma->values[i] *= mb->values[i];
    }
}

void matrix_multiply(pMatrix ma, pMatrix mb, pMatrix m_res){
    if ((ma->row != mb->row) || (ma->col != mb->col)){
        fprintf(stderr, "matrix_multiply() Matrix a shape must be equal Matrix b shape!\n");
        exit(-1);
    }

    int row = ma->row;
    int col = ma->col;

    assert((m_res->row == row) && (m_res->col == col));

    for(int i = 0; i < row * col; i++){
        m_res->values[i] = ma->values[i] * mb->values[i];
    }

}

void matrix_reshape(pMatrix m, int row, int col){
    int _row = m->row;
    int _col = m->col;
    if (row * col != _row * _col){
        fprintf(stderr, "matrix_reshape() Matrix shape(%d, %d), cann't reshape (%d, %d)", _row, _col, row, col);
        exit(-1);
    }
    m->row = row;
    m->col = col;
}

void matrix_scale(pMatrix m, float ratio){
    int row = m->row;
    int col = m->col;
    for (int i = 0; i < row * col; i++){
        m->values[i] *= ratio;
    }
}

void matrix_map(pMatrix ma, float(*map_fun)(float)){
    int row = ma->row;
    int col = ma->col;
    for(int i = 0;i< row * col; i++){
        ma->values[i] = map_fun(ma->values[i]);
    }
}

void matrix_mapfunc(pMatrix ma, pMatrix mb, float(*map_fun)(float)){
    if ((ma->row != mb->row) || (ma->col != mb->col)){
        fprintf(stderr, "matrix_mapfunc() Matrix a shape must be equal Matrix b shape!\n");
        exit(-1);
    }
    int row = ma->row;
    int col = ma->col;
    for(int i = 0;i < row * col; i++){
        mb->values[i] = (float)map_fun(ma->values[i]);
    }
}

void matrix_normal(pMatrix m){
    int row = m->row;
    int col = m->col;

    for(int i = 0; i < row * col; i++)
        m->values[i] = stand_normal();
}

float matrix_at(pMatrix m, int x, int y){
    int row = m->row;
    int col = m->col;
    if((x < 0) || (y < 0) || (x >= row) || (y >= col))
        return 0.0;
    return m->values[x * col + y];
}

void matrix_set(pMatrix m, int x, int y, float value){
    int row = m->row;
    int col = m->col;
    if((x < 0) || (y < 0) || (x >= row) || (y >= col)){
        fprintf(stderr, "matrix set error! %d, %d, %d, %d\n", x, y, row, col);
        exit(0);
    }
    m->values[x * col + y] = value;
}

