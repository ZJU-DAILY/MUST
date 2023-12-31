/***************************
@Author: xxx
@Contact: xxx@xxx.com
@File: distances_include.h
@Time: 2022/4/12 2:28 PM
@Desc:
***************************/

#ifndef GRAPHANNS_DISTANCES_INCLUDE_H
#define GRAPHANNS_DISTANCES_INCLUDE_H

#include "euclidean_distance.h"
#include "hamming_distance.h"
#include "attribute_distance.h"
#include "manhattan_distance.h"
#include "inner_product.h"
#include "attribute_similarity.h"
#include "euclidean_distance_eigen.h"

using VecValType1 = float;   // vector value type
using VecValType2 = int;   // vector value type
using DistResType1 = float;
using DistResType2 = float;
using DistResType = float;  // distances value type

// distances calculation type
using DistInnerProduct = CGraph::UDistanceCalculator<VecValType1, DistResType1, InnerProduct<VecValType1, DistResType1> >;
using DistAttributeSimilarity = CGraph::UDistanceCalculator<VecValType2, DistResType2, AttributeSimilarity<VecValType2, DistResType2> >;
using DistManhattanDistance = CGraph::UDistanceCalculator<VecValType2, DistResType2, ManhattanDistance<VecValType2, DistResType2> >;

template<typename DistCalcType1 = DistInnerProduct,
        typename DistCalcType2 = DistAttributeSimilarity,
        typename TVec1 = VecValType1,    // vector type of modal1
        typename TVec2 = VecValType2,    // vector type of modal2
        typename TRes1 = DistResType1,
        typename TRes2 = DistResType2,
        typename TRes = DistResType>    // fusion distance type
struct BiDistanceCalculator {
    DistCalcType1 dist_op1_;
    DistCalcType2 dist_op2_;

    explicit BiDistanceCalculator() = default;

    explicit BiDistanceCalculator(float w1, float w2) {
        set_weight(w1, w2);
    }

    float weight_1_ = Params.w1_;
    float weight_2_ = Params.w2_;

    void set_weight(float a, float b) {
        weight_1_ = a;
        weight_2_ = b;
    }

    CStatus calc_vec1(const TVec1 *vec_11, const TVec1 *vec_21, unsigned dim_11, unsigned dim_21,
                      DistResType &result) {
        CStatus status;
        DistResType1 res_1;

        status += dist_op1_.calculate(vec_11, vec_21, dim_11, dim_21, res_1);
        result = status.isOK() ? -res_1 : 0.f;
        return status;
    }

    CStatus calc_vec2(const TVec2 *vec_12, const TVec2 *vec_22, unsigned dim_12, unsigned dim_22,
                      DistResType &result) {
        CStatus status;
        DistResType2 res_2;

        status += dist_op2_.calculate(vec_12, vec_22, dim_12, dim_22, res_2);
        result = status.isOK() ? -res_2 : 0.f;
        return status;
    }

    CStatus calc_vec2(const std::vector<TVec2> &vec_12, const std::vector<TVec2> &vec_22,
                 DistResType &result) {
        CStatus status;
        DistResType2 res_2;

        status += dist_op2_.calculate(vec_12, vec_22, res_2);
        result = status.isOK() ? -res_2 : 0.f;
        return status;
    }

    CStatus calc(const std::vector<TVec1> &vec_11, const std::vector<TVec1> &vec_21,
                 const std::vector<TVec2> &vec_12, const std::vector<TVec2> &vec_22,
                 DistResType &result) {
        CStatus status;
        DistResType1 res_1;
        DistResType2 res_2;

        status += dist_op1_.calculate(vec_11, vec_21, res_1);
        status += dist_op2_.calculate(vec_12, vec_22, res_2);
//        weight_2_ = res_1 / (float)vec_12.size();
        result = status.isOK() ? res_1 * weight_1_ + (DistResType) res_2 * weight_2_ : 0.f;
        return status;
    }

    CStatus calculate(const TVec1 *vec_11, const TVec1 *vec_21, unsigned dim_11, unsigned dim_21,
                      const TVec2 *vec_12, const TVec2 *vec_22, unsigned dim_12, unsigned dim_22,
                      DistResType &result) {
        CStatus status;
        DistResType1 res_1;
        DistResType2 res_2;

        if (weight_1_ == 0) {
            res_1 = 0;
        } else {
            status += dist_op1_.calculate(vec_11, vec_21, dim_11, dim_21, res_1);
        }
        if (weight_2_ == 0) {
            res_2 = 0;
        } else {
            status += dist_op2_.calculate(vec_12, vec_22, dim_12, dim_22, res_2);
//            res_2 /= (DistResType2) dim_12;
        }
//        res_2 = (DistResType2)(res_2 == 0 ? 0 : (4.32 - 1 / (std::log10(res_2) + 1)));
//        weight_2_ = res_1 / (float)dim_12;
        result = status.isOK() ? res_1 * weight_1_ + (DistResType) res_2 * weight_2_ : 0.f;
        return status;
    }
};

#endif //GRAPHANNS_DISTANCES_INCLUDE_H
