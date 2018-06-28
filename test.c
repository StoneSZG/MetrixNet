//
// Created by Szg on 2018/6/14.
//
#include <stdio.h>
#include <time.h>

#include "test.h"
#include "loss.h"
#include "data.h"
#include "utils.h"
#include "layer.h"
#include "matrix.h"
#include "convolutional_layer.h"
#include "activations.h"
#include "pool.h"

void dnn_test(){
//    srand((unsigned)time(NULL));
    int length = 5;
    int n_epochs = 8000;
    Layer ls[7] = {0};
    int batch_size = 6;
    float leanring_rate = 1e-4;
    Matrix m_x = make_matrix_normal(batch_size, 4);
    Matrix m_y = make_matrix_eyes(batch_size);

    ls[0] = make_fully_connected_layer(batch_size, 4, 8, 1);
    ls[1] = make_relu_layer(batch_size, 8);
    ls[2] = make_fully_connected_layer(batch_size, 8, 16, 1);
    ls[3] = make_relu_layer(batch_size, 16);
    ls[4] = make_fully_connected_layer(batch_size, 16, 6, 1);

    float loss = 0.0;
    for(int k = 0; k < n_epochs; k++) {
        matrix_normal(&m_x);
        matrix_copy(&(ls[0].input), &m_x);
        for (int i = 0; i < length; i++) {
            if (i != 0) {
                matrix_copy(&(ls[i - 1].output), &(ls[i].input));
            }
            ls[i].forward(&(ls[i]));
        }


        loss = softmax_with_cross_entropy_error(&(ls[length - 1].delta), &m_y);

        if (((k + 1) % 1) == 0) {
            printf("epches %d:loss is :%0.2f\n", k, loss);
        }

        for (int i = length - 1; i > 0; i--) {
            if (i != length - 1) {
                matrix_copy(&(ls[i + 1].input), &(ls[i].output));
            }
            ls[i].backward(&(ls[i]));
        }

        for (int i = 0; i < length; i++) {
            if (ls[i].use_weights)
                matrix_scale(&(ls[i].update_weight), leanring_rate);
            if (ls[i].use_bias)
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


void dnn_mnist(){
    int length = 5;
    int n_epochs = 10000;
    Layer ls[7] = {0};
    int batch = 100;
    float leanring_rate = 1e-3;
    float loss = 0.0;
    float accuracy = 0.0;

    char *filename = "../data/train.csv";
    FILE *fp = fopen(filename, "r");
    if(!fp){
        fprintf(stderr, "Open file: %s error!\n", filename);
        exit(0);
    }

    Matrix m_x = make_matrix_zeros(batch, 784);
    Matrix m_y = make_matrix_zeros(batch, 10);

    ls[0] = make_fully_connected_layer(batch, 784, 1000, 1);
    ls[1] = make_relu_layer(batch, 1000);
//    ls[1] = make_sigmoid_layer(batch, 1000);
    ls[2] = make_fully_connected_layer(batch, 1000, 512, 1);
    ls[3] = make_relu_layer(batch, 512);
//    ls[3] = make_sigmoid_layer(batch, 512);
    ls[4] = make_fully_connected_layer(batch, 512, 10, 1);

    for(int k = 0; k < n_epochs; k++) {
        matrix_fill(&m_y, 0);
        matrix_fill(&m_x, 0);
        get_mnist_batch(fp, &m_x, &m_y);
        matrix_copy(&m_x, &(ls[0].input));

        for (int i = 0; i < length; i++) {
            if (i != 0) {
                matrix_copy(&(ls[i - 1].output), &(ls[i].input));
            }
            ls[i].forward(&(ls[i]));
        }

//        printf("%d output:\n", k);
//        print_matrix(&(ls[length - 1].output));
        accuracy = compute_accuracy(&(ls[length - 1].output), &m_y);
        loss = softmax_with_cross_entropy_error(&(ls[length - 1].output), &m_y);
        matrix_copy(&(ls[length - 1].output), &(ls[length - 1].delta));

//        printf("%d output:\n", k);
//        print_matrix(&(ls[length - 1].output));

//        printf("softmax_with_cross_entropy_error:\n");

        if ((k % 50) == 0) {
//            printf("Output:\n");
//            print_matrix(&(ls[length - 1].output));
            printf("epches %d:loss is :%0.4f accuracy is :%0.4f\n", k, loss, accuracy);
        }

        for (int i = length - 1; i >= 0; i--) {
//            printf("layer: %d \n", i);
            if (i < length - 1) {
                matrix_copy(&(ls[i + 1].input), &(ls[i].delta));
            }
//            printf("%d backward delta:\n", i);
//            print_matrix(&(ls[i].delta));
            ls[i].backward(&(ls[i]));
//            printf("%d update_bias:\n", i);
//            print_matrix(&(ls[i].update_bias));
//
        }

        for (int i = 0; i < length; i++) {
            if (ls[i].use_weights)
                matrix_scale(&(ls[i].update_weight), leanring_rate);
            if (ls[i].use_bias){
                matrix_scale(&(ls[i].update_bias), leanring_rate);
//                printf("%d update_bias:\n", i);
//                print_matrix(&(ls[i].update_bias));
            }

            ls[i].update(&(ls[i]));
        }
    }

//    matrix_normal(&m_x);
//    matrix_copy(&(ls[0].input), &m_x);
//    for (int i = 0; i < length; i++) {
//        if (i != 0) {
//            matrix_copy(&(ls[i - 1].output), &(ls[i].input));
//        }
//
//        ls[i].forward(&(ls[i]));
//    }
//
//    printf("Predict: \n");
//    print_matrix(&(ls[length - 1].output));

    for(int i = 0; i < length; i++){
        free_layer(&(ls[i]));
    }

    fclose(fp);

    free_matrix(&m_x);
    free_matrix(&m_y);

}

void cnn_mnist(){
    int length = 9;
    int n_epochs = 5000;
    Layer ls[15] = {0};
    int batch = 100;
    float leanring_rate = 1e-3;
    float accuracy = 0.0;
    float loss = 0.0;

    char *filename = "../data/train.csv";
    FILE *fp = fopen(filename, "r");
    if(!fp){
        fprintf(stderr, "Open file: %s error!\n", filename);
        exit(0);
    }

    Matrix m_x = make_matrix_zeros(batch, 784);
    Matrix m_y = make_matrix_zeros(batch, 10);

    ls[0] = make_convolutional_layer(batch, 28, 28, 1, 8, 3, 1, 1, 1);
    ls[1] = make_relu_layer(batch, 28 * 28 * 8);
    ls[2] = make_maxpool_layer(batch, 28, 28, 8, 2, 2, 0);

    ls[3] = make_convolutional_layer(batch, 14, 14, 8, 32, 3, 1, 1, 1);
    ls[4] = make_relu_layer(batch, 14 * 14 * 32);
    ls[5] = make_maxpool_layer(batch, 14, 14, 32, 2, 2, 0);

    ls[6] = make_fully_connected_layer(batch, 7 * 7 * 32, 512, 1);
    ls[7] = make_relu_layer(batch, 512);
    ls[8] = make_fully_connected_layer(batch, 512, 10, 1);

    for(int k = 0; k < n_epochs; k++) {
        matrix_fill(&m_y, 0);
        matrix_fill(&m_x, 0);
        get_mnist_batch(fp, &m_x, &m_y);
        matrix_copy(&m_x, &(ls[0].input));


        for (int i = 0; i < length; i++) {
            if (i != 0) {
                matrix_copy(&(ls[i - 1].output), &(ls[i].input));
            }
//            printf("Output layer %d:\n", i);
            ls[i].forward(&(ls[i]));
        }

        accuracy = compute_accuracy(&(ls[length - 1].output), &m_y);
        loss = softmax_with_cross_entropy_error(&(ls[length - 1].output), &m_y);
        matrix_copy(&(ls[length - 1].output), &(ls[length - 1].delta));

//        printf("network output:\n");
//        print_matrix(&(ls[length - 1].output));

        if ((k % 100) == 0) {
//            printf("Output layer 5:\n");
//            print_matrix(&(ls[5].output));
            printf("epches %d:loss is :%0.4f accuracy is :%0.2f\n", k, loss, accuracy);
        }

        for (int i = length - 1; i >= 0; i--) {
            if (i < length - 1) {
                matrix_copy(&(ls[i + 1].input), &(ls[i].delta));
            }
//            printf("backward delta:\n");
//            print_matrix(&(ls[i].delta));
            ls[i].backward(&(ls[i]));
//            printf("backward input:\n");
//            print_matrix(&(ls[i].input));

        }

        for (int i = 0; i < length; i++) {
            if (ls[i].use_weights){
                matrix_scale(&(ls[i].update_weight), leanring_rate);
//                printf("%d update_weight:\n", i);
//                print_matrix(&(ls[i].update_weight));
            }

            if (ls[i].use_bias){
                matrix_scale(&(ls[i].update_bias), leanring_rate);
//                printf("%d update_bias:\n", i);
//                print_matrix(&(ls[i].update_bias));
            }

            ls[i].update(&(ls[i]));
        }

    }


    for(int i = 0; i < length; i++){
        free_layer(&(ls[i]));
    }

    fclose(fp);

    free_matrix(&m_x);
    free_matrix(&m_y);

}