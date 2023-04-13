/***************************
@Author: wmz
@Contact: wmengzhao@qq.com
@File: util.h
@Time: 2022/11/12 12:10 PM
@Desc: util include
***************************/

#ifndef WEIGHTLEARNING_UTIL_H
#define WEIGHTLEARNING_UTIL_H

namespace WeightLearning {

    float* data_align(float* data_ori, unsigned point_num, unsigned& dim);

}  // namespace WeightLearning

#endif //WEIGHTLEARNING_UTIL_H
