//
// Created by Szg on 2018/6/21.
//

#include <stdio.h>
#include <assert.h>

#include "data.h"

void get_mnist_batch(FILE *fp, pMatrix X, pMatrix Y){
    assert(X->row == Y->row);
    assert(X->col == 784);
    assert(Y->col == 10);
    if(feof(fp)) fseek(fp, 0, SEEK_SET);
    int batch = X->row;
    int ch = 0;
    float value = 0.0;
    int i = 0;
    for(int b = 0; b < batch; b++){
        i = 0;
        while(i < 785){
            ch = fgetc(fp);
            if(feof(fp)) fseek(fp, 0, SEEK_SET);
            if(ch >= '0' && ch <= '9'){
                value = value * 10 + ch - '0';
            }else if(ch == ',' || ch == '\n'){
                if(i == 0){
                    matrix_set(Y, b, (int)value, 1.0);
                }else{
                    value = (value - 0.0) / 255.;
                    matrix_set(X, b, i - 1, value);
                }
                i++;
                value = 0.0;
            }
        }

    }

//    float mean = matrix

}

Matrix get_mnist_data(FILE *fp, int batch){
    if(feof(fp)) fseek(fp, 0, SEEK_SET);
    Matrix m = make_matrix_zeros(batch, 785);
    int ch = 0;
    float value = 0.0;
    int i = 0;
    for(int b = 0; b < batch; b++){
        i = 0;
        while(i < 785){
            ch = fgetc(fp);
            if(feof(fp)) fseek(fp, 0, SEEK_SET);
            if(ch >= '0' && ch <= '9'){
                value = value * 10 + ch - '0';
            }else if(ch == ',' || ch == '\n'){
                matrix_set(&m, b, i, value);
                i++;
                value = 0.0;
            }
        }

    }

    return m;
}

Matrix load_mnist(char *filename){
    FILE *fp = fopen(filename, "r");
    if(!fp){
        fprintf(stderr, "Open file: %s error!\n", filename);
        exit(0);
    }
    Matrix m = make_matrix_zeros(1500, 785);
    char * line;
//    line = fgetc(fp);

    printf("line: %s\n", line);
}
