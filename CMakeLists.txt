cmake_minimum_required(VERSION 3.10)
project(MetrixNet C)

set(CMAKE_C_STANDARD 99)

add_executable(MetrixNet main.c matrix.c matrix.h utils.c utils.h layer.c layer.h activations.c activations.h loss.c loss.h test.c test.h convolutional_layer.c convolutional_layer.h array.c array.h pool.c pool.h data.c data.h rnn_layer.c rnn_layer.h)

