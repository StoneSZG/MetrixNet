//
// Created by Szg on 2018/6/12.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "utils.h"
#include "matrix.h"
#include "activations.h"

#define INF 0x7fffffff

float normal_distribution(float mean, float std){
    static int sign = 0;
    static float U, V;
    if(sign){
        sign = 0;
        float value = sqrt(U)* sin(V);
        return mean + value * std;
    }
    sign = 1;
    float rand1 = rand() / (RAND_MAX + 1.0);
    float rand2 = rand() / (RAND_MAX + 1.0);
    U = -2.0 * log(rand1);
    V = 2.0 * PI * rand2;
    float value = sqrt(U)* cos(V);
    return mean + value * std;
}

float stand_normal(){
    return normal_distribution(0, 0.1);
}

float uniform_distribution(float min, float max){
    if(max < min){
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand()/RAND_MAX * (max - min)) + min;
}

float stand_uniform(){
    return uniform_distribution(0, 1);
}

void matrix_softmax(pMatrix m){
    matrix_map(m, exp_activate);
    Matrix vector = make_matrix_zeros(1, m->row);
    matrix_sum(m, &vector, 1);
    matrix_div_vector(m, &vector, 0);

}

void im2col(float* im, int channels,  int height,  int width,
            int ksize, int stride, int pad, float* col){

    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;

    for(int c = 0; c < channels_col; c++){
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for(int h = 0; h < height_col; h++)
            for(int w = 0; w < width_col; w++){
                int im_row = w_offset + h * stride;
                int im_col = h_offset + w * stride;
//                printf("im row: %d, col:%d ", im_row, im_col);
                int col_index = (c * height_col + h) * width_col + w;
                float value = im2col_get(im, height, width, im_row, im_col, c_im, pad);
//                printf("%d:%0.2f, ", col_index, value);
                col[col_index] = value;
            }
//        printf("\n");
    }

}

float compute_accuracy(pMatrix pre, pMatrix y){
    int row = pre->row;
    int col = pre->col;
    int pre_argmax = 0;
    int y_argmax = 0;
    float pre_max = -INF;
    float y_max = -INF;
    float accuracy = 0.0;

    for(int i = 0;i < row; i++){
        pre_max = -INF;
        y_max = -INF;
//        accuracy = 0.0;
        for(int j = 0;j < col; j++){
            float pre_j = matrix_at(pre, i, j);
            float y_j = matrix_at(y, i, j);
            if(pre_max < pre_j){
                pre_max = pre_j;
                pre_argmax = j;
            }
            if(y_max < y_j){
                y_max = y_j;
                y_argmax = j;
            }

        }
        if(pre_argmax == y_argmax){
//            printf("pre_argmax:%d, y_argmax:%d", pre_argmax, y_argmax);
            accuracy += 1;
        }

    }
    return accuracy / row;
}

float im2col_get(float* m, int height, int width,
                 int row, int col, int c, int pad){
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return m[col + width*(row + height*c)];
}


void col2im_set(float* m, int height, int width,
                 int row, int col, int c, int pad, float val){
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return;

    m[col + width*(row + height*c)] += val;
}


void col2im(float* col, int channels,  int height,  int width,
            int ksize, int stride, int pad, float* im){
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;

    for(int c = 0; c < channels_col; c++){
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for(int h = 0; h < height_col; h++)
            for(int w = 0; w < width_col; w++){
                int im_row = w_offset + h * stride;
                int im_col = h_offset + w * stride;
//                printf("im row: %d, col:%d ", im_row, im_col);
                int col_index = (c * height_col + h) * width_col + w;
                float val = col[col_index];
                col2im_set(im, height, width, im_row, im_col, c_im, pad, val);
//                printf("%d:%0.2f, ", col_index, value);
//                col[col_index] = value;
            }
//        printf("\n");
    }

}