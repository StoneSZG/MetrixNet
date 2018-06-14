//
// Created by Szg on 2018/6/12.
//


#include <math.h>
#include <stdlib.h>
#include "utils.h"

float normal_distribution(float mean, float std){
    float U = rand() / (RAND_MAX + 1.0);
    float V = rand() / (RAND_MAX + 1.0);
    float value = sqrt(-2.0 * log(U))* sin(2.0 * PI * V);
    return mean + value * std;
}

float stand_normal(){
    return normal_distribution(0, 1);
}
