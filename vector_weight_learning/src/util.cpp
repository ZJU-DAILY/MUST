/***************************
@Author: xxx
@Contact: xxx@xx.com
@File: util.cpp
@Time: 2022/11/12 9:57 AM
@Desc: util detail
***************************/

#include <malloc.h>
#include <cstring>
#include "util.h"

namespace WeightLearning {

float* data_align(float* data_ori, unsigned point_num, unsigned& dim) {
#ifdef __GNUC__
#ifdef __AVX__
#define DATA_ALIGN_FACTOR 8
#else
#ifdef __SSE2__
#define DATA_ALIGN_FACTOR 4
#else
#define DATA_ALIGN_FACTOR 1
#endif
#endif
#endif
  float* data_new = 0;
  unsigned new_dim =
      (dim + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
#ifdef __APPLE__
  data_new = new float[(size_t)new_dim * (size_t)point_num];
#else
  data_new =
      (float*)memalign(DATA_ALIGN_FACTOR * 4,
                       (size_t)point_num * (size_t)new_dim * sizeof(float));
#endif

  for (size_t i = 0; i < point_num; i++) {
    memcpy(data_new + i * new_dim, data_ori + i * dim, dim * sizeof(float));
    memset(data_new + i * new_dim + dim, 0, (new_dim - dim) * sizeof(float));
  }

  dim = new_dim;

#ifdef __APPLE__
  delete[] data_ori;
#else
//  free(data_ori);
#endif

  return data_new;
}

}  // namespace WeightLearning
