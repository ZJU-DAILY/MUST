/***************************
@Author: xxx
@Contact: xxx@xxx.com
@File: c7_routing_kgraph_modal2.h
@Time: 2022/11/1 6:54 PM
@Desc: modal2 search routing
***************************/

#ifndef GRAPHANNS_C7_ROUTING_KGRAPH_MODAL2_H
#define GRAPHANNS_C7_ROUTING_KGRAPH_MODAL2_H

#include "../c7_routing_basic.h"

class C7RoutingKGraphModal2 : public C7RoutingBasic {
public:
    DAnnFuncType prepareParam() override {
        auto *s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY);
        auto *a_param = CGRAPH_GET_GPARAM(AlgParamBasic, GA_ALG_PARAM_BASIC_KEY);
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        if (nullptr == model_ || nullptr == s_param) {
            return DAnnFuncType::ANN_PREPARE_ERROR;
        }

        num_ = model_->train_meta_modal1_.num;
        dim1_ = model_->train_meta_modal1_.dim;
        dim2_ = model_->train_meta_modal2_.dim;
        data_modal1_ = model_->train_meta_modal1_.data;
        data_modal2_ = model_->train_meta_modal2_.data;
        search_L_ = s_param->search_L;
        K_ = Params.candi_top_k_;
        query_id_modal2_ = s_param->modal2_query_id;
        query_modal1_ = model_->search_meta_modal1_.data;
        query_modal2_ = model_->search_meta_modal2_.data;
        if (Params.is_delete_id_) {
            delete_num_each_query_ = model_->delete_meta_.dim;
        }
        dist_op_.set_weight(0, Params.w2_);
        return DAnnFuncType::ANN_SEARCH;
    }

    CStatus search() override {
        auto s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY);
        if (nullptr == s_param) {
            CGRAPH_RETURN_ERROR_STATUS("C7RoutingKGraph search get param failed")
        }
        assert(search_L_ >= K_);

        std::vector<char> flags(num_, 0);
        res_modal2_.clear();

        unsigned k = 0;
        while (k < (int) search_L_) {
            unsigned nk = search_L_;

            if (s_param->sp_modal2[k].flag_) {
                s_param->sp_modal2[k].flag_ = false;
                IDType n = s_param->sp_modal2[k].id_;

                for (unsigned m = 0; m < model_->graph_m2_[n].size(); ++m) {
                    IDType id = model_->graph_m2_[n][m];

                    if (flags[id]) continue;
                    flags[id] = 1;
                    bool is_delete = false;
                    if (delete_num_each_query_) {
                        for (IDType k = 0; k < delete_num_each_query_; k++) {
                            if (id == model_->delete_meta_.data[s_param->modal2_query_id * delete_num_each_query_ + k]) {
                                is_delete = true;
                                break;
                            }
                        }
                    }
                    if (is_delete) continue;

                    DistResType dist = 0;
                    dist_op_.calculate(query_modal1_ + ((size_t)query_id_modal2_ * (size_t)dim1_),
                                       data_modal1_ + (size_t)id * (size_t)dim1_,
                                       dim1_, dim1_,
                                       query_modal2_ + ((size_t)query_id_modal2_ * (size_t)dim2_),
                                       data_modal2_ + (size_t)id * (size_t)dim2_,
                                       dim2_, dim2_, dist);

                    if (dist >= s_param->sp_modal2[search_L_ - 1].distance_) continue;
                    NeighborFlag nn(id, dist, true);
                    int r = InsertIntoPool(s_param->sp_modal2.data(), search_L_, nn);

                    if (r < nk) nk = r;
                }
            }
            nk <= k ? (k = nk) : (++k);
        }

        res_modal2_.reserve(K_);
        for (size_t i = 0; i < K_; i++) {
            res_modal2_.push_back(s_param->sp_modal2[i].id_);
        }
        return CStatus();
    }

    CStatus refreshParam() override {
        auto a_param = CGRAPH_GET_GPARAM(AlgParamBasic, GA_ALG_PARAM_BASIC_KEY);
        CGRAPH_ASSERT_NOT_NULL(a_param)

        {
            CGRAPH_PARAM_WRITE_CODE_BLOCK(a_param)
            a_param->results_modal2.push_back(res_modal2_);
        }
        return CStatus();
    }
};

#endif //GRAPHANNS_C7_ROUTING_KGRAPH_MODAL2_H
