//
// Created by Szg on 2018/6/14.
//

#include <stdio.h>
#include <assert.h>

#include "utils.h"
#include "array.h"
#include "layer.h"
#include "matrix.h"
#include "activations.h"

Layer make_relu_layer(int batch_size, int input){
    Layer l = {0};
    l.input = make_matrix_zeros(batch_size, input);
    l.output = make_matrix_zeros(batch_size, input);
    l.delta = make_matrix_zeros(batch_size, input);

    l.forward = relu_forward;
    l.backward = relu_backward;
    l.update = layer_update_none_op;
    l.use_bias = 0;
    l.use_weights = 0;

    return l;
}

void relu_forward(pLayer l){
    matrix_mapfunc(&(l->input), &(l->output), relu_activate);
}

void relu_backward(pLayer l){
    int row = l->output.row;
    int col = l->output.col;

    array_apply(l->input.values, row * col, relu_gradient, l->delta.values);
    matrix_copy(&(l->delta), &(l->input));

}

Layer make_sigmoid_layer(int batch_size, int input){
    Layer l = {0};
    l.input = make_matrix_ones(batch_size, input);
    l.output = make_matrix_ones(batch_size, input);
    l.delta = make_matrix_ones(batch_size, input);

    l.forward = sigmoid_forward;
    l.backward = sigmoid_backward;
    l.update = layer_update_none_op;
    l.use_bias = 0;

    return l;
}

void sigmoid_forward(pLayer l){
    matrix_mapfunc(&(l->input), &(l->output), sigmoid_activate);
}



void sigmoid_backward(pLayer l){
    int row = l->output.row;
    int col = l->output.col;
    array_apply(l->output.values, row * col, sigmoid_gradient, l->delta.values);
}

Layer make_softmax_layer(int batch_size, int input){
    Layer l = {0};
    l.input = make_matrix_ones(batch_size, input);
    l.output = make_matrix_ones(batch_size, input);
    l.delta = make_matrix_ones(batch_size, input);

    l.forward = softmax_forward;
    l.backward = softmax_backward;
    l.update = layer_update_none_op;
    l.use_bias = 0;
    l.use_weights = 0;

    return l;
}

void softmax_forward(pLayer l){
//    matrix_mapfunc(&(l->input), &(l->output), exp_activate);
//    Matrix vector = make_matrix_zeros(1, l->output.row);
//    matrix_sum(&(l->output), &vector, 1);
//    matrix_div_vector(&(l->output), &vector, 0);
    matrix_softmax(&(l->input));
    matrix_copy(&(l->input), &(l->output));

}

void softmax_backward(pLayer l){
    int row = l->input.row;
    int col = l->input.col;
    float values = 0.0;
    Matrix m = make_matrix_zeros(col, col);

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            for(int k = 0; k < col; k++){
                if(j == k)
                    values = l->input.values[i * col + j] * ( 1 - l->input.values[i * col + k]);
                else
                    values = -(l->input.values[i * col + j]) * (l->input.values[i * col + k]);
                m.values[j * col + k] = values;
            }
        }
        for(int j = 0; j < col; j++) {
            values = 0.0;
            for(int k = 0; k < col; k++)
                values += l->output.values[i * col + k] * m.values[j + col * k];
            l->delta.values[i * col + j] = values;
        }

    }
//    matrix_copy(&(l->delta), &(l->input));
    free_matrix(&m);

}

float abs_activate(float x){
//    float eps = 1e-3;
    return (x >= 0) ? x: -x;
}

float abs_gradient(float x){
//    float eps = 1e-3;
    return (x >= 0) ? x : -x;
}

float exp_activate(float x){
//    float eps = 1e-3;
    return (float)exp(x );
}


float stair_activate(float x) {
    int n = floor(x);
    if (n%2 == 0) return floor(x/2.);
    else return (x - n) + floor(x/2.);
}

float hardtan_activate(float x) {
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
float linear_activate(float x){return x;}
float sigmoid_activate(float x){return 1. / (1. + exp(-x));}
float loggy_activate(float x){return 2. / (1. + exp(-x)) - 1;}
float relu_activate(float x){return x * (x > 0);}
float elu_activate(float x){return (x >= 0) * x + (x < 0) * (exp(x) - 1);}
float relie_activate(float x){return (x > 0) ? x : .01 * x;}
float ramp_activate(float x){return x * (x > 0) + .1 * x;}
float leaky_activate(float x){return (x > 0) ? x : .1 * x;}
float tanh_activate(float x){return (exp(2 * x) - 1) / (exp(2 * x) + 1);}
float plse_activate(float x) {
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125 * x + .5;
}

float lhtan_activate(float x) {
    if(x < 0) return .001 * x;
    if(x > 1) return .001 * (x - 1) + 1;
    return x;
}
float lhtan_gradient(float x) {
    if(x > 0 && x < 1) return 1;
    return .001;
}

float hardtan_gradient(float x) {
    if (x > -1 && x < 1) return 1;
    return 0;
}
float linear_gradient(float x){return 1;}
float sigmoid_gradient(float x){return (1 - x) * x;}
float loggy_gradient(float x) {
    float y = (x + 1.) / 2.;
    return 2 * (1 - y) * y;
}
float stair_gradient(float x) {
    if (floor(x) == x) return 0;
    return 1;
}
float relu_gradient(float x){return (x >= 0);}
float elu_gradient(float x){return (x >= 0) + (x < 0)*(x + 1);}
float relie_gradient(float x){return (x>0) ? 1 : .01;}
float ramp_gradient(float x){return (x>0)+.1;}
float leaky_gradient(float x){return (x>0) ? 1 : .1;}
float tanh_gradient(float x){return 1-x*x;}
float plse_gradient(float x){return (x < 0 || x > 1) ? .01 : .125;}

float log_gradient(float x){
    float eps = 1e-3;
    x = x == 0? x + eps: x;
    return -1.0 / (x);
}
