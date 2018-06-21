//
// Created by Szg on 2018/6/14.
//

#include <stdio.h>
#include <math.h>

#include "loss.h"
#include "utils.h"
#include "activations.h"

float clip_by_value(float value, float min, float max){
    if (value < min){
        return min;
    }else if(value > max){
        return max;
    }
    return value;
}

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
//    matrix_scale(pre, 2);

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
    matrix_sub(pre, y);
//    printf("size: row:%d, col:%d\n", row, col);
    for(int i = 0; i < row * col; i++){
        loss += abs_activate(pre->values[i]);
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
//    printf("cross_entropy_error() pre:");
//    print_matrix(pre);

    for(int i = 0; i < row * col; i++){
        loss += -y->values[i] * log(pre->values[i]);
    }

//    matrix_sub(pre, y);
    matrix_map(pre, log_gradient);
    matrix_mul(pre, y);

    loss /= (row * col);
    return loss;
}

float softmax_with_cross_entropy_error(pMatrix pre, pMatrix y){
    if ((pre->row != y->row) || (pre->col != y->col)){
        fprintf(stderr, "cross_entropy_error() Matrix a shape must be equal Matrix b shape!\n");
        exit(-1);
    }
    float loss = 0.0;
    int row = pre->row;
    int col = pre->col;

//    printf("softmax_with_cross_entropy_error:\n");

    matrix_softmax(pre);

    float value = 0.0;
    float eps = 1e-3;

    for(int i = 0; i < row * col; i++){
        value = clip_by_value(pre->values[i], eps, 1.0);
//        if(value) value = eps;
//        printf("loss: %0.8f %0.8f %0.8f\n", loss, log(value), value);
        loss += - y->values[i] * log(value);
    }

    matrix_sub(pre, y);
//    printf("loss: %0.2f\n", loss);
    loss /= (row * col);
    return loss;
}

