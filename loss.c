//
// Created by Szg on 2018/6/14.
//

#include <stdio.h>
#include <math.h>

#include "loss.h"
#include "activations.h"

float mean_sequare_error(pMatrix pre, pMatrix y){
    if ((pre->row != y->row) || (pre->col != y->col)){
        fprintf(stderr, "mean_sequare_error() Matrix a shape must be equal Matrix b shape!\n");
        exit(-1);
    }
    float loss = 0.0;
    int row = pre->row;
    int col = pre->col;
    for(int i = 0; i < row * col; i++){
        loss += pow(pre->values[i] - y->values[i], 2);
    }

    matrix_sub(pre, y);
    matrix_scale(pre, 2);

    loss /= (row * col);
    return loss;
}

float mean_absolute_error(pMatrix pre, pMatrix y){
    if ((pre->row != y->row) || (pre->col != y->col)){
        fprintf(stderr, "mean_absolute_error() Matrix a shape must be equal Matrix b shape!\n");
        exit(-1);
    }

    float loss = 0.0;
    int row = pre->row;
    int col = pre->col;
//    printf("size: row:%d, col:%d\n", row, col);
    for(int i = 0; i < row * col; i++){
        loss += abs_activate(pre->values[i] - y->values[i]);
    }

//    print_matrix(pre);
//    printf("mean_absolute_error: %0.2f\n", loss);
    matrix_map(pre, abs_gradient);
//    print_matrix(pre);

    loss /= (row * col);
    return loss;
}

float cross_entropy_error(pMatrix pre, pMatrix y){
    if ((pre->row != y->row) || (pre->col != y->col)){
        fprintf(stderr, "cross_entropy_error() Matrix a shape must be equal Matrix b shape!\n");
        exit(-1);
    }
    float loss = 0.0;
    int row = pre->row;
    int col = pre->col;

    for(int i = 0; i < row * col; i++){
        loss += -y->values[i] * log(pre->values[i]);
    }

    matrix_sub(pre, y);

    loss /= (row * col);
    return loss;
}