//
// Created by Szg on 2018/6/14.
//
#include <stdio.h>
#include "layer.h"

//#include "matrix.h"

Layer make_fully_connected_layer(int batch_size, int input, int output, int use_bias){
    Layer l = {0};
    l.input = make_matrix_zeros(batch_size, input);
    l.output = make_matrix_zeros(batch_size, output);
    l.delta = make_matrix_zeros(batch_size, output);
    l.weight = make_matrix_normal(input, output);
    l.update_weight = make_matrix_zeros(input, output);
    if(use_bias){
        l.bias = make_matrix_zeros(1, output);
        l.update_bias = make_matrix_zeros(1, output);
    }
    l.forward = fully_connected_forward;
    l.backward = fully_connected_backward;
    l.update = fully_connected_update;
    l.use_bias = use_bias;
    l.use_weights = 1;

    return l;
}

void fully_connected_forward(player l){
    matrix_matmul(&(l->input), &(l->weight), &(l->output));
    if(l->use_bias){
        matrix_add_vector(&(l->output), &(l->bias), 1);
    }
}

void fully_connected_backward(player l){
    if(l->use_bias){
        matrix_sum(&(l->delta), &(l->update_bias), 0);
//        matrix_sub_vector(&(l->output), &(l->bias), 1);
    }

//    printf("fully_connected_backward: update_bias\n");

    Matrix input_copy = make_matrix_zeros(l->input.row, l->input.col);
    matrix_copy(&(l->input), &input_copy);
    Matrix weight_copy = make_matrix_zeros(l->weight.row, l->weight.col);
    matrix_copy(&(l->weight), &weight_copy);

    matrix_transpose(&(input_copy));
    matrix_matmul(&(input_copy), &(l->delta), &(l->update_weight));


    matrix_transpose(&(weight_copy));
    matrix_matmul(&(l->delta), &(weight_copy), &(l->input));


    free_matrix(&input_copy);
    free_matrix(&weight_copy);
}

void fully_connected_update(player l){
    if(l->use_bias){
        matrix_sub(&(l->bias), &(l->update_bias));
    }
    matrix_sub(&(l->weight), &(l->update_weight));
}

void layer_update(pLayer l){
    if(l->use_bias){
        matrix_sub(&(l->bias), &(l->update_bias));
    }
    matrix_sub(&(l->weight), &(l->update_weight));
}

void layer_update_none_op(pLayer l){
    return ;
}

void free_layer(pLayer l){
    free_matrix(&(l->input));
    free_matrix(&(l->output));
    free_matrix(&(l->delta));
    free_matrix(&(l->weight));
    free_matrix(&(l->update_weight));
    if(l->use_bias){
        free_matrix(&(l->bias));
        free_matrix(&(l->update_bias));
    }
}