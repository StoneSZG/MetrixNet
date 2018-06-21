//
// Created by Szg on 2018/6/15.
//

#ifndef METRIXNET_CONVOLUTIONAL_LAYER_H
#define METRIXNET_CONVOLUTIONAL_LAYER_H

#include "layer.h"

Layer make_convolutional_layer(int, int, int, int, int, int, int, int, int);

void convolutional_forward(pLayer);

void convolutional_backward(pLayer);

void convolutional_update(pLayer);

#endif //METRIXNET_CONVOLUTIONAL_LAYER_H
