/***************************
@Author: wmz
@Contact: wmengzhao@qq.com
@File: opt_set.cpp
@Time: 2022/11/12 9:57 AM
@Desc: operation set
***************************/

#include <omp.h>
#include <chrono>
#include <cmath>
#include <queue>

#include "opt_set.h"
#include "util.h"

namespace WeightLearning {

    struct cmp {
        template<typename T, typename U>
        bool operator()(T const &left, U const &right) {
            if (left.first < right.first) return true;
            return false;
        }
    };

    OptSet::OptSet(Metric metric, const unsigned thread_num, const unsigned is_norm) {
        switch (metric) {
            case L2_Float:
                dist_op_float_ = new EuclideanDistance_Float();
                break;
            case IP_Float:
                dist_op_float_ = new InnerProduct_Float();
                break;
            case AS_Int_Skip0:
                dist_op_int_ = new AttributeSimilarity_Int_SKIP0();
                break;
            case AS_Int:
                dist_op_int_ = new AttributeSimilarity_Int();
            case IP_Float_AS_Int_Skip0:
                dist_op_float_ = new InnerProduct_Float();
                dist_op_int_ = new AttributeSimilarity_Int_SKIP0();
                break;
            case IP_Float_AS_Int:
                dist_op_float_ = new InnerProduct_Float();
                dist_op_int_ = new AttributeSimilarity_Int();
                break;
            default:
                dist_op_float_ = new InnerProduct_Float();
                break;
        }
        is_norm_ = is_norm;
        thread_num_ = thread_num;
    }

    void OptSet::load_base_float(float *base, const size_t base_num, const size_t dim) {
        base_modal1_ = base;
        base_num_ = base_num;
        size_t dim_tmp = dim;
        if (is_norm_) norm(base_modal1_, base_num_, dim);
        base_modal1_ = data_align(base_modal1_, base_num_,
                                  reinterpret_cast<unsigned int &>(dim_tmp));
        dim1_ = dim_tmp;
        std::cout << "base shape: " << base_num_ << ", " << dim << " --> " << dim1_ << std::endl;
    }

    void OptSet::load_base_int(const int *base, const size_t base_num, const size_t dim) {
        base_modal2_int_ = base;
        base_num_ = base_num;
        dim2_ = dim;
        std::cout << "base shape: " << base_num_ << ", " << dim2_ << std::endl;
    }

    void OptSet::load_query_float(float *query, const size_t query_num, const size_t dim) {
        query_modal1_ = query;
        query_num_ = query_num;
        size_t dim_tmp = dim;
        if (is_norm_) norm(query_modal1_, query_num_, dim);
        query_modal1_ = data_align(query_modal1_, query_num_,
                                   reinterpret_cast<unsigned int &>(dim_tmp));
        dim1_ = dim_tmp;
        std::cout << "query shape: " << query_num_ << ", " << dim << " --> " << dim1_ << std::endl;
    }

    void OptSet::load_query_int(const int *query, const size_t query_num, const size_t dim) {
        query_modal2_int_ = query;
        query_num_ = query_num;
        dim2_ = dim;
        std::cout << "query shape: " << query_num_ << ", " << dim2_ << std::endl;
    }

    void OptSet::load_1float_2int(float *base_modal1, float *query_modal1, const int *base_modal2,
                                  const int *query_modal2, const size_t base_num, const size_t query_num,
                                  const size_t dim1, const size_t dim2) {
        base_modal1_ = base_modal1;
        query_modal1_ = query_modal1;
        base_modal2_int_ = base_modal2;
        query_modal2_int_ = query_modal2;
        base_num_ = base_num;
        query_num_ = query_num;
        size_t dim1_base = dim1;
        size_t dim1_query = dim1;
        dim2_ = dim2;
        if (is_norm_) {
            norm(base_modal1_, base_num_, dim1);
            norm(query_modal1_, query_num_, dim1);
        }
        base_modal1_ = data_align(base_modal1_, base_num_,
                                  reinterpret_cast<unsigned int &>(dim1_base));
        query_modal1_ = data_align(query_modal1_, query_num_,
                                   reinterpret_cast<unsigned int &>(dim1_query));
        assert(dim1_base == dim1_query);
        dim1_ = dim1_base;
        std::cout << "base modal1 shape: " << base_num_ << ", " << dim1 << " --> " << dim1_ << std::endl;
        std::cout << "query modal1 shape: " << query_num_ << ", " << dim1 << " --> " << dim1_ << std::endl;
        std::cout << "base modal2 shape: " << base_num_ << ", " << dim2_ << std::endl;
        std::cout << "query modal2 shape: " << query_num_ << ", " << dim2_ << std::endl;
    }

    void OptSet::load_1float_2int_345float(float *base_modal1, float *query_modal1, const int *base_modal2,
                                  const int *query_modal2, float *base_modal3, float *query_modal3,
                                  float *base_modal4, float *query_modal4,
                                  const size_t base_num, const size_t query_num,
                                  const size_t dim1, const size_t dim2, const size_t dim3, const size_t dim4) {
        base_modal1_ = base_modal1;
        query_modal1_ = query_modal1;
        base_modal2_int_ = base_modal2;
        query_modal2_int_ = query_modal2;
        base_modal3_ = base_modal3;
        query_modal3_ = query_modal3;
        base_modal4_ = base_modal4;
        query_modal4_ = query_modal4;
        base_num_ = base_num;
        query_num_ = query_num;
        size_t dim1_base = dim1;
        size_t dim1_query = dim1;
        dim2_ = dim2;
        size_t dim3_base = dim3;
        size_t dim3_query = dim3;
        size_t dim4_base = dim4;
        size_t dim4_query = dim4;
        if (is_norm_) {
            norm(base_modal1_, base_num_, dim1);
            norm(query_modal1_, query_num_, dim1);
            norm(base_modal3_, base_num_, dim3);
            norm(query_modal3_, query_num_, dim3);
            norm(base_modal4_, base_num_, dim4);
            norm(query_modal4_, query_num_, dim4);
        }
        base_modal1_ = data_align(base_modal1_, base_num_,
                                  reinterpret_cast<unsigned int &>(dim1_base));
        query_modal1_ = data_align(query_modal1_, query_num_,
                                   reinterpret_cast<unsigned int &>(dim1_query));
        base_modal3_ = data_align(base_modal3_, base_num_,
                                  reinterpret_cast<unsigned int &>(dim3_base));
        query_modal3_ = data_align(query_modal3_, query_num_,
                                   reinterpret_cast<unsigned int &>(dim3_query));
        base_modal4_ = data_align(base_modal4_, base_num_,
                                  reinterpret_cast<unsigned int &>(dim4_base));
        query_modal4_ = data_align(query_modal4_, query_num_,
                                   reinterpret_cast<unsigned int &>(dim4_query));
        assert(dim1_base == dim1_query);
        assert(dim3_base == dim3_query);
        assert(dim4_base == dim4_query);
        dim1_ = dim1_base;
        dim3_ = dim3_base;
        dim4_ = dim4_base;
        std::cout << "base modal1 shape: " << base_num_ << ", " << dim1 << " --> " << dim1_ << std::endl;
        std::cout << "query modal1 shape: " << query_num_ << ", " << dim1 << " --> " << dim1_ << std::endl;
        std::cout << "base modal2 shape: " << base_num_ << ", " << dim2_ << std::endl;
        std::cout << "query modal2 shape: " << query_num_ << ", " << dim2_ << std::endl;
        std::cout << "base modal3 shape: " << base_num_ << ", " << dim3 << " --> " << dim3_ << std::endl;
        std::cout << "query modal3 shape: " << query_num_ << ", " << dim3 << " --> " << dim3_ << std::endl;
        std::cout << "base modal4 shape: " << base_num_ << ", " << dim4 << " --> " << dim4_ << std::endl;
        std::cout << "query modal4 shape: " << query_num_ << ", " << dim4 << " --> " << dim4_ << std::endl;
    }

    void OptSet::load_1float_2float(float *base_modal1, float *query_modal1, float *base_modal2,
                                    float *query_modal2, const size_t base_num, const size_t query_num,
                                    const size_t dim1, const size_t dim2) {
        base_modal1_ = base_modal1;
        query_modal1_ = query_modal1;
        base_modal2_float_ = base_modal2;
        query_modal2_float_ = query_modal2;
        base_num_ = base_num;
        query_num_ = query_num;
        size_t dim1_base = dim1;
        size_t dim1_query = dim1;
        size_t dim2_base = dim2;
        size_t dim2_query = dim2;
        dim2_ = dim2;
        if (is_norm_) {
            norm(base_modal1_, base_num_, dim1);
            norm(query_modal1_, query_num_, dim1);
            norm(base_modal2_float_, base_num_, dim2);
            norm(query_modal2_float_, query_num_, dim2);
        }
        base_modal1_ = data_align(base_modal1_, base_num_,
                                  reinterpret_cast<unsigned int &>(dim1_base));
        query_modal1_ = data_align(query_modal1_, query_num_,
                                   reinterpret_cast<unsigned int &>(dim1_query));
        base_modal2_float_ = data_align(base_modal2_float_, base_num_,
                                  reinterpret_cast<unsigned int &>(dim2_base));
        query_modal2_float_ = data_align(query_modal2_float_, query_num_,
                                        reinterpret_cast<unsigned int &>(dim2_query));
        assert(dim1_base == dim1_query);
        assert(dim2_base == dim2_query);
        dim1_ = dim1_base;
        dim2_ = dim2_base;
        std::cout << "base modal1 shape: " << base_num_ << ", " << dim1 << " --> " << dim1_ << std::endl;
        std::cout << "query modal1 shape: " << query_num_ << ", " << dim1 << " --> " << dim1_ << std::endl;
        std::cout << "base modal2 shape: " << base_num_ << ", " << dim2 << " --> " << dim2_ << std::endl;
        std::cout << "query modal2 shape: " << query_num_ << ", " << dim2 << " --> " << dim2_ << std::endl;
    }

    void OptSet::float_gen_dist(unsigned query_st, unsigned query_ed, std::vector<std::vector<float>> &dist_list) {
        assert(query_st >= 0 && query_st < query_num_);
        assert(query_ed >= 0 && query_ed < query_num_);
        auto *cur_query = const_cast<float *>(query_modal1_ + query_st * dim1_);
        int cur_query_num = (int) query_ed - (int) query_st + 1;
        dist_list.resize(cur_query_num);
#pragma omp parallel for num_threads(thread_num_) schedule(dynamic) default(none) shared(cur_query, \
        cur_query_num, dist_list)
        for (size_t i = 0; i < cur_query_num; i++) {
            dist_list[i].resize(base_num_);
            for (size_t j = 0; j < base_num_; j++) {
                float d = dist_op_float_->calculate(cur_query + i * dim1_, base_modal1_ + j * dim1_, dim1_);
                dist_list[i][j] = d;
            }
        }
    }

    void OptSet::float_dist_by_id(unsigned id_1, unsigned id_2, float &dist) {
        assert(id_1 >= 0 && id_1 < base_num_);
        assert(id_2 >= 0 && id_2 < base_num_);
        dist = dist_op_float_->calculate(base_modal1_ + id_1 * dim1_, base_modal1_ + id_2 * dim1_, dim1_);
    }

    void OptSet::int_dist_by_id(unsigned id_1, unsigned id_2, float &dist) {
        assert(id_1 >= 0 && id_1 < base_num_);
        assert(id_2 >= 0 && id_2 < base_num_);
        dist = dist_op_int_->calculate(base_modal2_int_ + id_1 * dim2_, base_modal2_int_ + id_2 * dim2_, dim2_);
    }

    void OptSet::int_gen_dist(unsigned query_st, unsigned query_ed, std::vector<std::vector<float>> &dist_list) {
        assert(query_st >= 0 && query_st < query_num_);
        assert(query_ed >= 0 && query_ed < query_num_);
        auto *cur_query = const_cast<int *>(query_modal2_int_ + query_st * dim2_);
        int cur_query_num = (int) query_ed - (int) query_st + 1;
        dist_list.resize(cur_query_num);
#pragma omp parallel for num_threads(thread_num_) schedule(dynamic) default(none) shared(cur_query, \
        cur_query_num, dist_list)
        for (size_t i = 0; i < cur_query_num; i++) {
            dist_list[i].resize(base_num_);
            for (size_t j = 0; j < base_num_; j++) {
                float d = dist_op_int_->calculate(cur_query + i * dim2_, base_modal2_int_ + j * dim2_, dim2_);
                dist_list[i][j] = d;
            }
        }
    }

    void OptSet::float_int_topk_id(unsigned query_st, unsigned query_ed, const unsigned *k,
                                   std::vector<std::vector<unsigned>> &id_list) {
        assert(query_st >= 0 && query_st < query_num_);
        assert(query_ed >= 0 && query_ed < query_num_);
        auto *cur_query_modal1 = const_cast<float *>(query_modal1_ + query_st * dim1_);
        auto *cur_query_modal2 = const_cast<int *>(query_modal2_int_ + query_st * dim2_);
        auto *cur_query_modal3 = const_cast<float *>(query_modal3_ + query_st * dim3_);
        auto *cur_query_modal4 = const_cast<float *>(query_modal4_ + query_st * dim4_);
        int cur_query_num = (int) query_ed - (int) query_st + 1;
        id_list.resize(cur_query_num);
#pragma omp parallel for num_threads(thread_num_) schedule(dynamic) default(none) shared(cur_query_modal1, \
        cur_query_modal2, cur_query_modal3, cur_query_modal4, cur_query_num, k, id_list)
        for (size_t i = 0; i < cur_query_num; i++) {
            std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>, cmp> obj;
            for (size_t j = 0; j < base_num_; j++) {
                float d1 = dist_op_float_->calculate(cur_query_modal1 + i * dim1_, base_modal1_ + j * dim1_, dim1_);
                float d2 = dist_op_int_->calculate(cur_query_modal2 + i * dim2_, base_modal2_int_ + j * dim2_, dim2_);
                float d3 = dist_op_float_->calculate(cur_query_modal3 + i * dim3_, base_modal3_ + j * dim3_, dim3_);
                float d4 = dist_op_float_->calculate(cur_query_modal4 + i * dim4_, base_modal4_ + j * dim4_, dim4_);
                float d = d1 * w1_ + d2 * w2_ + d3 * w3_ + d4 * w4_;
                obj.push(std::pair<float, int>(d, j));
            }

            for (size_t j = 0; j < k[i]; j++) {
                std::pair<float, unsigned> p = obj.top();
                obj.pop();
                id_list[i].emplace_back(p.second);
            }
        }
    }

    void OptSet::float_float_topk_id(unsigned query_st, unsigned query_ed, const unsigned *k,
                                   std::vector<std::vector<unsigned>> &id_list) {
        assert(query_st >= 0 && query_st < query_num_);
        assert(query_ed >= 0 && query_ed < query_num_);
        auto *cur_query_modal1 = const_cast<float *>(query_modal1_ + query_st * dim1_);
        auto *cur_query_modal2 = const_cast<float *>(query_modal2_float_ + query_st * dim2_);
        int cur_query_num = (int) query_ed - (int) query_st + 1;
        id_list.resize(cur_query_num);
#pragma omp parallel for num_threads(thread_num_) schedule(dynamic) default(none) shared(cur_query_modal1, \
        cur_query_modal2, cur_query_num, k, id_list)
        for (size_t i = 0; i < cur_query_num; i++) {
            std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>, cmp> obj;
            for (size_t j = 0; j < base_num_; j++) {
                float d1 = dist_op_float_->calculate(cur_query_modal1 + i * dim1_, base_modal1_ + j * dim1_, dim1_);
                float d2 = dist_op_float_->calculate(cur_query_modal2 + i * dim2_, base_modal2_float_ + j * dim2_, dim2_);
                float d = d1 * w1_ + d2 * w2_;
                obj.push(std::pair<float, int>(d, j));
            }

            for (size_t j = 0; j < k[i]; j++) {
                std::pair<float, unsigned> p = obj.top();
                obj.pop();
                id_list[i].emplace_back(p.second);
            }
        }
    }

    void OptSet::modal1_dist_by_id(unsigned query_id, const unsigned *base_id, unsigned base_id_num,
                                   std::vector<float> &dist_list) {
        assert(query_id >= 0 && query_id < query_num_);
        auto *cur_query = const_cast<float *>(query_modal1_ + query_id * dim1_);
        dist_list.resize(base_id_num);
#pragma omp parallel for num_threads(thread_num_) schedule(dynamic) default(none) shared(std::cout, cur_query, \
        base_id, base_id_num, dist_list)
        for (size_t i = 0; i < base_id_num; i++) {
            float d = dist_op_float_->calculate(cur_query, base_modal1_ + base_id[i] * dim1_,
                                                dim1_);
            dist_list[i] = d;
        }

    }

    void OptSet::modal2_int_dist_by_id(unsigned query_id, const unsigned *base_id, unsigned base_id_num,
                                       std::vector<float> &dist_list) {
        assert(query_id >= 0 && query_id < query_num_);
        auto *cur_query = const_cast<int *>(query_modal2_int_ + query_id * dim2_);
        dist_list.resize(base_id_num);
#pragma omp parallel for num_threads(thread_num_) schedule(dynamic) default(none) shared(cur_query, base_id, \
        base_id_num, dist_list)
        for (size_t i = 0; i < base_id_num; i++) {
            float d = dist_op_int_->calculate(cur_query, base_modal2_int_ + base_id[i] * dim2_,
                                              dim2_);
            dist_list[i] = d;
        }

    }

    void OptSet::modal2_float_dist_by_id(unsigned query_id, const unsigned *base_id, unsigned base_id_num,
                                       std::vector<float> &dist_list) {
        assert(query_id >= 0 && query_id < query_num_);
        auto *cur_query = const_cast<float *>(query_modal2_float_ + query_id * dim2_);
        dist_list.resize(base_id_num);
#pragma omp parallel for num_threads(thread_num_) schedule(dynamic) default(none) shared(cur_query, base_id, \
        base_id_num, dist_list)
        for (size_t i = 0; i < base_id_num; i++) {
            float d = dist_op_float_->calculate(cur_query, base_modal2_float_ + base_id[i] * dim2_,
                                              dim2_);
            dist_list[i] = d;
        }

    }

    void OptSet::modal3_dist_by_id(unsigned query_id, const unsigned *base_id, unsigned base_id_num,
                                   std::vector<float> &dist_list) {
        assert(query_id >= 0 && query_id < query_num_);
        auto *cur_query = const_cast<float *>(query_modal3_ + query_id * dim3_);
        dist_list.resize(base_id_num);
#pragma omp parallel for num_threads(thread_num_) schedule(dynamic) default(none) shared(std::cout, cur_query, \
        base_id, base_id_num, dist_list)
        for (size_t i = 0; i < base_id_num; i++) {
            float d = dist_op_float_->calculate(cur_query, base_modal3_ + base_id[i] * dim3_,
                                                dim3_);
            dist_list[i] = d;
        }

    }

    void OptSet::modal4_dist_by_id(unsigned query_id, const unsigned *base_id, unsigned base_id_num,
                                   std::vector<float> &dist_list) {
        assert(query_id >= 0 && query_id < query_num_);
        auto *cur_query = const_cast<float *>(query_modal4_ + query_id * dim4_);
        dist_list.resize(base_id_num);
#pragma omp parallel for num_threads(thread_num_) schedule(dynamic) default(none) shared(std::cout, cur_query, \
        base_id, base_id_num, dist_list)
        for (size_t i = 0; i < base_id_num; i++) {
            float d = dist_op_float_->calculate(cur_query, base_modal4_ + base_id[i] * dim4_,
                                                dim4_);
            dist_list[i] = d;
        }

    }

    void OptSet::norm(float *vec, const unsigned n, const unsigned d) {
        for (unsigned i = 0; i < n; i++) {
            float vector_norm = 0;
            for (unsigned j = 0; j < d; j++) {
                vector_norm += vec[i * d + j] * vec[i * d + j];
            }
            vector_norm = std::sqrt(vector_norm);
            for (unsigned j = 0; j < d; j++) {
                vec[i * d + j] /= vector_norm;
            }
        }
    }

    void OptSet::load_weight(float w1, float w2, float w3, float w4) {
        w1_ = w1;
        w2_ = w2;
        w3_ = w3;
        w4_ = w4;
    }

}  // namespace WeightLearning
