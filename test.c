//
// Created by Szg on 2018/6/14.
//
#include <stdio.h>
#include <time.h>

#include "loss.h"
#include "test.h"
#include "layer.h"
#include "matrix.h"
#include "activations.h"

void dnn_test(){
    srand((unsigned)time(NULL));
    int length = 4;
    int n_epochs = 500;
    Layer ls[4] = {0};
    int batch_size = 6;
    float leanring_rate = 1e-2;
    Matrix m_x = make_matrix_normal(batch_size, 4);
    Matrix m_y = make_matrix_eyes(batch_size);

    ls[0] = make_fully_connected_layer(batch_size, 4, 16, 1);
    ls[1] = make_relu_layer(batch_size, 16);
    ls[2] = make_fully_connected_layer(batch_size, 16, 6, 1);
    ls[3] = make_softmax_layer(batch_size, 6);
    float loss = 0.0;
    for(int k = 0; k < n_epochs; k++) {
//        printf("epches %d\n", k);
        matrix_normal(&m_x);
        matrix_copy(&(ls[0].input), &m_x);
//        printf("before for:\n");
        for (int i = 0; i < length; i++) {
            if (i != 0) {
                matrix_copy(&(ls[i - 1].output), &(ls[i].input));
            }
            ls[i].forward(&(ls[i]));
        }

//        printf("before mean_absolute_error:\n");
        loss = cross_entropy_error(&(ls[length - 1].output), &m_y);
//        printf("after mean_absolute_error:\n");

        if(((k + 1) % 100) == 0){
            printf("epches %d:loss is :%0.2f\n", k, loss);
        }

        for (int i = length - 1; i > 0; i--) {
            if (i != length - 1) {
                matrix_copy(&(ls[i + 1].input), &(ls[i].output));
            }
            ls[i].backward(&(ls[i]));
        }

        for (int i = 0; i < length; i++) {
            if(ls[i].use_weights)
                matrix_scale(&(ls[i].update_weight), leanring_rate);
            if(ls[i].use_bias)
                matrix_scale(&(ls[i].update_bias), leanring_rate);
            ls[i].update(&(ls[i]));
        }
    }

    matrix_normal(&m_x);
    matrix_copy(&(ls[0].input), &m_x);
    for (int i = 0; i < length; i++) {
        if (i != 0) {
            matrix_copy(&(ls[i - 1].output), &(ls[i].input));
        }
        ls[i].forward(&(ls[i]));
    }

    printf("Predict: \n");
    print_matrix(&(ls[length - 1].output));

    for(int i = 0; i < length; i++){
        free_layer(&(ls[i]));
    }
}