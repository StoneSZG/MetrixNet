//
// Created by Szg on 2018/6/21.
//

#ifndef METRIXNET_DATA_H
#define METRIXNET_DATA_H

#include "matrix.h"

void get_mnist_batch(FILE *fp, pMatrix X, pMatrix Y);
Matrix get_mnist_data(FILE *fp, int batch);

Matrix load_mnist(char *filename);

#endif //METRIXNET_DATA_H
