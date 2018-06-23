//
// Created by Szg on 2018/6/20.
//

#ifndef METRIXNET_POOL_H
#define METRIXNET_POOL_H


#include "layer.h"

Layer make_maxpool_layer(int, int, int, int, int, int, int);
Layer make_avgpool_layer(int, int, int, int, int);

void maxpool_forward(pLayer);
void avgpool_forward(pLayer);

void maxpool_backward(pLayer);
void avgpool_backward(pLayer);

void pool_update(pLayer);

#endif //METRIXNET_POOL_H
