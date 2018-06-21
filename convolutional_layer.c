//
// Created by Szg on 2018/6/15.
//

#include <stdio.h>

#include "utils.h"
#include "array.h"
#include "convolutional_layer.h"

Layer make_convolutional_layer(int batch_size, int h, int w, int c, int n,
                               int size, int stride, int padding, int use_bias){
    Layer l = {0};
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.size = size;
    l.stride = stride;
    l.pad = padding;
    l.use_bias = use_bias;
    l.out_w = (w - size + 2 * padding) / stride + 1;
    l.out_h = (h - size + 2 * padding) / stride + 1;
    l.out_c = n;

    l.input = make_matrix_ones(batch_size, w * h * c);
    l.output = make_matrix_zeros(batch_size, l.out_w * l.out_h * n);
    l.delta = make_matrix_zeros(batch_size, l.out_w * l.out_h * n);

    l.weight = make_matrix_ones(n, c * size * size);
    l.update_weight = make_matrix_zeros(n, c * size * size);

    if(use_bias){
        l.bias = make_matrix_incre(1, n, 1);
        l.update_bias = make_matrix_zeros(1, n);
    }

    l.forward = convolutional_forward;
    l.backward = convolutional_backward;
    l.update = convolutional_update;

    return l;
}

void convolutional_forward(pLayer l){
    int batch = l->input.row;
//    int width = l->input.col;
    int ksize = l->size;
    int h = l->h;
    int w = l->w;
    int c = l->c;
    int n = l->n;

    int height_col = l->out_h;
    int width_col = l->out_w;

//    printf("height_col: %d, width_col: %d\n", height_col, width_col);
    Matrix col = make_matrix_zeros(c * ksize * ksize, height_col * width_col);
    Matrix res = make_matrix_zeros(n, height_col * width_col);
    size_t size = height_col * width_col * n;


    for(int i = 0; i < batch; i++){
        im2col(l->input.values + (i * c * h * w), c, w, h, ksize, l->stride, l->pad, col.values);
        matrix_matmul(&(l->weight), &col, &res);
        if(l->use_bias){
            matrix_add_vector(&res, &(l->bias), 0);
        }
        array_copy(res.values, size * sizeof(float), l->output.values + i * size);
    }

    free_matrix(&col);
    free_matrix(&res);

}

void convolutional_backward_bias(pLayer l){
    int batch = l->input.row;
//    int width = l->input.col;
    int ksize = l->size;
    int h = l->h;
    int w = l->w;
    int c = l->c;
    int n = l->n;
    int height_col = l->out_h;
    int width_col = l->out_w;
    size_t size = height_col * width_col;
    matrix_fill(&(l->update_bias), 0);

    for(int i = 0; i < batch; i++){
        for(int j = 0;j < n * size; j++){
            l->update_bias.values[j / size] += l->delta.values[i];
        }
    }

}

void convolutional_backward(pLayer l){
    int batch = l->input.row;
    int ksize = l->size;
    int h = l->h;
    int w = l->w;
    int c = l->c;
    int n = l->n;
    int height_col = l->out_h;
    int width_col = l->out_w;
    size_t size = height_col * width_col * n;

    if(l->use_bias){
        convolutional_backward_bias(l);
    }

    matrix_fill(&(l->update_weight), 0);

    Matrix col = make_matrix_zeros(c * ksize * ksize, height_col * width_col);
    Matrix res = make_matrix_zeros(n, height_col * width_col);
    Matrix weight_update = make_matrix_zeros(n, c * ksize * ksize);

    Matrix input_copy = make_matrix_zeros(l->input.row, l->input.col);
    Matrix weight_copy = make_matrix_zeros(l->weight.row, l->weight.col);
    matrix_copy(&(l->weight), &weight_copy);
    matrix_transpose(&(weight_copy));

    for(int i = 0; i < batch; i++){
        im2col(l->input.values + (i * c * h * w), c, w, h, ksize, l->stride, l->pad, col.values);
        array_copy(l->delta.values + i * size, size * sizeof(float), res.values);

        matrix_transpose(&col);
        matrix_matmul(&(res), &col, &weight_update);
        matrix_add(&(l->update_weight), &weight_update);

        matrix_matmul(&(weight_copy), &res, &col);

        col2im(col.values, c, w, h, ksize, l->stride, l->pad, input_copy.values + (i * c * h * w));

        matrix_reshape(&col, c * ksize * ksize, height_col * width_col);
    }

    matrix_copy(&input_copy, &(l->input));

    free_matrix(&col);
    free_matrix(&res);
    free_matrix(&input_copy);
    free_matrix(&weight_update);

}

void convolutional_update(pLayer l){
    if(l->use_bias){
        matrix_sub(&(l->bias), &(l->update_bias));
    }
    matrix_sub(&(l->weight), &(l->update_weight));

}
