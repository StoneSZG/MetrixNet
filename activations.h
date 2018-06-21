//
// Created by Szg on 2018/6/14.
//

#ifndef METRIXNET_ACTIVATIONS_H
#define METRIXNET_ACTIVATIONS_H

#include <math.h>
#include "layer.h"

Layer make_relu_layer(int batch_size, int input);
Layer make_sigmoid_layer(int batch_size, int input);
Layer make_softmax_layer(int batch_size, int input);

void relu_forward(pLayer);
void relu_backward(pLayer);
//void relu_update(pLayer);

void update_none_op(pLayer);
void sigmoid_backward(pLayer);
void sigmoid_forward(pLayer);

void softmax_forward(pLayer);
void softmax_backward(pLayer);


float relu_gradient(float x);
float sigmoid_activate(float x);
float relu_activate(float x);
float sigmoid_gradient(float x);
float exp_activate(float x);
float abs_activate(float x);
float abs_gradient(float x);
float log_gradient(float);

#endif //METRIXNET_ACTIVATIONS_H
