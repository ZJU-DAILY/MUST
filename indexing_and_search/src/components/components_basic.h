/***************************
@Author: xxx
@Contact: xxx@xxx.com
@File: component_basic.h
@Time: 2022/4/30 14:40
@Desc: 
***************************/

#ifndef GRAPHANNS_COMPONENTS_BASIC_H
#define GRAPHANNS_COMPONENTS_BASIC_H

#include "../../CGraph/src/CGraph.h"
#include "../utils/utils.h"

using DistCalcType = BiDistanceCalculator<DistInnerProduct, DistAttributeSimilarity>;

class ComponentsBasic : public CGraph::DAnnNode {
protected:
    AnnsModelParam *model_ = nullptr;          // ann model ptr
    VecValType1 *data_modal1_ = nullptr;               // vector data
    VecValType2 *data_modal2_ = nullptr;
    size_t num_ = 0;                         // number of vector
    unsigned dim1_ = 0;                         // dimensionality of vector for modal1
    unsigned dim2_ = 0;                         // dimensionality of vector for modal2
    DistCalcType dist_op_;    // distance calculation type
};

#endif //GRAPHANNS_COMPONENTS_BASIC_H
