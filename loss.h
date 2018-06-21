//
// Created by Szg on 2018/6/14.
//

#ifndef METRIXNET_LOSS_H
#define METRIXNET_LOSS_H

#include "matrix.h"

float mean_sequare_error(pMatrix, pMatrix);
float cross_entropy_error(pMatrix, pMatrix);
float mean_absolute_error(pMatrix, pMatrix);
float softmax_with_cross_entropy_error(pMatrix pre, pMatrix y);

#endif //METRIXNET_LOSS_H
