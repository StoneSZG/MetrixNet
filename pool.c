//
// Created by Szg on 2018/6/20.
//

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
    int out_w = l->out_w;
    int out_h = l->out_h;
    int stride = l->stride;
    int pad = l->pad;

    int w_offset = -pad;
    int h_offset = -pad;

    matrix_fill(&(l->output), -INF);

    for(int b = 0; b < batch; ++b){
        for(int k = 0; k < c; ++k){
            for(int i = 0; i < h; ++i){
                for(int j = 0; j < w; ++j){
                    int out_index = j + w * (i + h * (k + c * b));
                    float max = -INF;
                    int max_i = -1;
                    for(int n = 0; n < ksize; ++n){
                        for(int m = 0; m < ksize; ++m){
                            int cur_h = h_offset + i * stride + n;
                            int cur_w = w_offset + j * stride + m;
                            int index = cur_w + w * (cur_h + h * (k + b * c));
                            int valid = (cur_h >= 0 && cur_h < h &&
                                         cur_w >= 0 && cur_w < w);
                            float val = (valid != 0) ? l->input.values[index] : -INF;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l->output.values[out_index] = max;
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


}
void avgpool_backward(pLayer l){

}

void pool_update(pLayer l){

}
