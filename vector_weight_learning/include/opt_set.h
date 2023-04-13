/***************************
@Author: wmz
@Contact: wmengzhao@qq.com
@File: opt_set.h
@Time: 2022/11/12 12:09 PM
@Desc: operation set include
***************************/

#ifndef WEIGHTLEARNING_OPT_SET_H
#define WEIGHTLEARNING_OPT_SET_H

#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include "distance.h"
#include "util.h"

namespace WeightLearning {

    class OptSet {
    public:
        explicit OptSet(Metric m, unsigned thread_num = 1, unsigned is_norm = 0);

        virtual ~OptSet() = default;

        virtual void load_base_float(float *base, size_t base_num, size_t dim);

        virtual void load_base_int(const int *base, size_t base_num, size_t dim);

        virtual void load_query_float(float *query, size_t query_num, size_t dim);

        virtual void load_query_int(const int *query, size_t query_num, size_t dim);

        virtual void load_1float_2int(float *base_modal1, float *query_modal1, const int *base_modal2,
                                      const int *query_modal2, size_t base_num, size_t query_num,
                                      size_t dim1, size_t dim2);

        virtual void load_1float_2int_345float(float *base_modal1, float *query_modal1, const int *base_modal2,
                                               const int *query_modal2, float *base_modal3, float *query_modal3,
                                               float *base_modal4, float *query_modal4,
                                               const size_t base_num, const size_t query_num,
                                               const size_t dim1, const size_t dim2, const size_t dim3, const size_t dim4);

        virtual void load_1float_2float(float *base_modal1, float *query_modal1, float *base_modal2,
                                      float *query_modal2, size_t base_num, size_t query_num,
                                      size_t dim1, size_t dim2);

        virtual void float_gen_dist(unsigned query_st, unsigned query_ed, std::vector<std::vector<float>> &dist_list);

        virtual void float_dist_by_id(unsigned id_1, unsigned id_2, float &dist);

        virtual void int_dist_by_id(unsigned id_1, unsigned id_2, float &dist);

        virtual void int_gen_dist(unsigned query_st, unsigned query_ed, std::vector<std::vector<float>> &dist_list);

        virtual void float_int_topk_id(unsigned query_st, unsigned query_ed, const unsigned *k,
                                       std::vector<std::vector<unsigned>> &id_list);

        virtual void float_float_topk_id(unsigned query_st, unsigned query_ed, const unsigned *k,
                                       std::vector<std::vector<unsigned>> &id_list);

        virtual void modal1_dist_by_id(unsigned query_id, const unsigned *base_id, unsigned base_id_num,
                                       std::vector<float> &dist_list);

        virtual void modal2_int_dist_by_id(unsigned query_id, const unsigned *base_id, unsigned base_id_num,
                                           std::vector<float> &dist_list);

        virtual void modal2_float_dist_by_id(unsigned query_id, const unsigned *base_id, unsigned base_id_num,
                                           std::vector<float> &dist_list);

        virtual void modal3_dist_by_id(unsigned query_id, const unsigned *base_id, unsigned base_id_num,
                                       std::vector<float> &dist_list);

        virtual void modal4_dist_by_id(unsigned query_id, const unsigned *base_id, unsigned base_id_num,
                                       std::vector<float> &dist_list);

        virtual void norm(float *data, unsigned n, unsigned d);

        virtual void load_weight(float w1, float w2, float w3, float w4);

    public:
        size_t thread_num_{};
        size_t base_num_{};
        size_t query_num_{};
        size_t dim_{};
        size_t dim1_{};
        size_t dim2_{};
        size_t dim3_{};
        size_t dim4_{};
        unsigned is_norm_{};
        DistanceBasic<float> *dist_op_float_{};
        DistanceBasic<int> *dist_op_int_{};
        float w_ = 1;
        float w1_ = 1;
        float w2_ = 1;
        float w3_ = 1;
        float w4_ = 1;
        float* base_modal1_{};
        float* query_modal1_{};
        float* base_modal2_float_{};
        float* query_modal2_float_{};
        const int* base_modal2_int_{};
        const int* query_modal2_int_{};
        float* base_modal3_{};
        float* query_modal3_{};
        float* base_modal4_{};
        float* query_modal4_{};
    };

}  // namespace WeightLearning

#endif //WEIGHTLEARNING_OPT_SET_H
