/***************************
@Author: xxx
@Contact: xxx@xxx.com
@File: eva_merge_node.h
@Time: 2022/11/1 6:59 PM
@Desc: merge multichannel results
***************************/

#ifndef GRAPHANNS_EVA_MERGE_NODE_H
#define GRAPHANNS_EVA_MERGE_NODE_H

#include "../../elements_define.h"
#include <unordered_set>

class EvaMergeNode : public CGraph::GNode {
public:
    CStatus init() override {
        auto m_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        CGRAPH_ASSERT_NOT_NULL(m_param)
        num_ = m_param->search_meta_modal1_.num;

        return CStatus();
    }
    CStatus run() override {
        auto *s_param = CGRAPH_GET_GPARAM(AlgParamBasic, GA_ALG_PARAM_BASIC_KEY);
        if (nullptr == s_param) {
            CGRAPH_RETURN_ERROR_STATUS("EvaMergeNode run get param failed")
        }
        top_k_ = s_param->top_k;
        s_param->results.resize(num_);
        for (IDType i = 0; i < num_; i++) {
            std::unordered_set<IDType> res_set;
            for (auto each1_id : s_param->results_modal1[i]) {
                if (std::find(s_param->results_modal2[i].begin(), s_param->results_modal2[i].end(),
                              each1_id) != s_param->results_modal2[i].end()) {
                    res_set.insert(each1_id);
                }
                if (res_set.size() >= top_k_) break;
            }
            IDType j = 0;
            while (res_set.size() < top_k_ && j < top_k_) {
                res_set.insert(s_param->results_modal1[i][j++]);
            }
            s_param->results[i].insert(s_param->results[i].end(), res_set.begin(), res_set.end());
        }

        return CStatus();
    }

private:
    IDType num_;
    unsigned top_k_;
};

#endif //GRAPHANNS_EVA_MERGE_NODE_H
