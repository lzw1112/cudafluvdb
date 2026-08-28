#pragma once
#include <cuda_runtime.h>
template<typename T>
T tex1Dfetch(cudaTextureObject_t texObj, int x);