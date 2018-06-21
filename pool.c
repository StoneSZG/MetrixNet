//
// Created by Administrator on 2018/6/20.
//

#include "pool.h"

Layer make_maxpool_layer(int batch_size, int h, int w, int c,
                         int size, int stride, int padding){
    Layer l = {0};
    l.h = h;
    l.w = w;
    l.c = c;
    l.size = size;
    l.stride = stride;
    l.pad = padding;
    l.use_bias = 0;
    l.out_w = (w + 2 * padding) / stride;
    l.out_h = (h + 2 * padding) / stride;
    l.out_c = c;

    l.input = make_matrix_ones(batch_size, w * h * c);
    l.output = make_matrix_zeros(batch_size, l.out_w * l.out_h * l.out_c);
    l.delta = make_matrix_zeros(batch_size, l.out_w * l.out_h * l.out_c);


    l.forward = maxpool_forward;
    l.backward = maxpool_backward;
    l.update = layer_update_none_op;

    return l;

}
Layer make_avgpool_layer(int batch_size, int h, int w, int c,
                         int size){
    Layer l = {0};
    l.h = h;
    l.w = w;
    l.c = c;
    l.size = size;
    l.use_bias = 0;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;

    l.input = make_matrix_ones(batch_size, w * h * c);
    l.output = make_matrix_zeros(batch_size, l.out_w * l.out_h * l.out_c);
    l.delta = make_matrix_zeros(batch_size, l.out_w * l.out_h * l.out_c);


    l.forward = maxpool_forward;
    l.backward = maxpool_backward;
    l.update = layer_update_none_op;

    return l;

}

void maxpool_forward(pLayer l){
    int batch = l->input.row;
    int ksize = l->size;
    int h = l->h;
    int w = l->w;
    int c = l->c;
    int n = l->n;

    for(int b = 0;b < batch; b++){
        for(int i = 0; i < c; i++){

        }
    }
}
void avgpool_forward(pLayer l){

}

void maxpool_backward(pLayer l){

}
void avgpool_backward(pLayer l){

}

void pool_update(pLayer l){

}
