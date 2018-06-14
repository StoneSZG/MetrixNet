//
// Created by Szg on 2018/6/14.
//

#ifndef METRIXNET_LOSS_H
#define METRIXNET_LOSS_H

#include "matrix.h"

float mean_sequare_error(pMatrix, pMatrix);
float cross_entropy_error(pMatrix, pMatrix);
float mean_absolute_error(pMatrix, pMatrix);

#endif //METRIXNET_LOSS_H
