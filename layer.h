//
// Created by Szg on 2018/6/14.
//

#ifndef METRIXNET_LAYER_H
#define METRIXNET_LAYER_H

#include "matrix.h"

typedef struct layer{
    Matrix input;
    Matrix delta;
    Matrix output;
    Matrix weight;
    int use_bias;
    int use_weights;
    Matrix bias;
    Matrix update_weight;
    Matrix update_bias;
    void (*forward)(struct layer*);
    void (*backward)(struct layer*);
    void (*update)(struct layer*);
    int h, w, c, n, size, stride, pad;
    int out_h, out_w, out_c;
}layer, Layer, *pLayer, *player;

Layer make_fully_connected_layer(int batch_size, int input, int output, int);

void fully_connected_forward(pLayer);
void fully_connected_backward(player);
void fully_connected_update(player);

void layer_update(pLayer);
void layer_update_none_op(pLayer);
void free_layer(pLayer);

#endif //METRIXNET_LAYER_H
