//
// Created by Szg on 2018/6/20.
//

#include <stdio.h>

#include "pool.h"

#define INF 0x7fffffff

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
    l.out_w = (w - size + 2 * padding) / stride + 1;
    l.out_h = (h - size + 2 * padding) / stride + 1;
    l.out_c = c;

    l.input = make_matrix_ones(batch_size, c * w * h);
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
    int out_w = l->out_w;
    int out_h = l->out_h;
    int stride = l->stride;
    int pad = l->pad;


    int w_offset = -pad;
    int h_offset = -pad;

    matrix_fill(&(l->output), -INF);

    for(int b = 0;b < batch; b++){
        for(int i = 0; i < c; i++){
            for(int j = 0; j < out_w; j++){
                for(int k = 0; k < out_h; k++){
                    int out_index = k + j * out_h + i * out_w * out_h + b * c * out_h * out_w;
                    float max_value = -INF;
                    for(int n = 0; n < ksize; n++){
                        for(int m = 0; m < ksize; m++){
                            int cur_w = w_offset + j * stride + n;
                            int cur_h = h_offset + k * stride + m;
                            int valid = (cur_w >= 0) && (cur_h >= 0) &&
                                        (cur_w < w) && (cur_h < h);
                            int ids = cur_w * h + cur_h + i * w * h + b * w * h * c;
                            float value = valid?l->input.values[ids]:-INF;
                            max_value = value > max_value?value:max_value;
                        }
                    }
                    l->output.values[out_index] = max_value;
                }
            }

        }
    }
}
void avgpool_forward(pLayer l){
    int batch = l->input.row;
    int ksize = l->size;
    int h = l->h;
    int w = l->w;
    int c = l->c;
    int n = l->n;
    int out_w = l->out_w;
    int out_h = l->out_h;
    int stride = l->stride;

    float max_value = -INF;
    float value = 0.0;

    matrix_fill(&(l->output), -INF);

    for(int b = 0;b < batch; b++){
        for(int i = 0; i < c; i++){
            for(int j = 0; j < h; j++)
                for(int k = 0; k < w; k++) {
                    value = matrix_at(&(l->input), b, i * j * k);
                    max_value = matrix_at(&(l->output), b, i * (j / ksize) * (k / ksize));
                    if(max_value < value){
                        matrix_set(&(l->output), b, i * (j / ksize) * (k / ksize), value);
                    }
                }
        }
    }
}

void maxpool_backward(pLayer l){
    int batch = l->input.row;
    int ksize = l->size;
    int h = l->h;
    int w = l->w;
    int c = l->c;
    int n = l->n;
    int out_w = l->out_w;
    int out_h = l->out_h;
    int stride = l->stride;
    int pad = l->pad;


    int w_offset = -pad;
    int h_offset = -pad;

//    matrix_fill(&(l->output), -INF);
    Matrix m = make_matrix_zeros(l->input.row, l->input.col);

    for(int b = 0;b < batch; b++){
        for(int i = 0; i < c; i++){
            for(int j = 0; j < out_w; j++){
                for(int k = 0; k < out_h; k++){
                    int out_index = k + j * out_h + i * out_w * out_h + b * c * out_h * out_w;
//                    int out_index = k * c + j * out_h * c + i + b * c * out_h * out_w;
                    int max_idx = -1;
                    float max_value = -INF;
                    for(int n = 0; n < ksize; n++){
                        for(int m = 0; m < ksize; m++){
                            int cur_w = w_offset + j * stride + n;
                            int cur_h = h_offset + k * stride + m;
                            int valid = (cur_w >= 0) && (cur_h >= 0) &&
                                        (cur_w < w) && (cur_h < h);
//                            int ids = cur_w * h + cur_h + i * w * h + b * w * h * c;
                            int ids = cur_w * h + cur_h + i * w * h + b * w * h * c;
                            float value = valid?l->input.values[ids]:-INF;
                            if (value > max_value){
                                max_value = value;
                                max_idx = ids;
                            }
                        }
                    }
//                    printf("delta values: %0.2f output value: %0.2f\n",l->delta.values[out_index], l->output.values[out_index]);
//                    printf("max pool :max idx: %d\n", max_idx);
                    m.values[max_idx] = l->delta.values[out_index];
//                    m.values[max_idx] = 1;
                }
            }

        }
    }

    matrix_copy(&m, &(l->input));

    free_matrix(&m);
}
void avgpool_backward(pLayer l){

}

void pool_update(pLayer l){

}
