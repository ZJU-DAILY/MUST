/***************************
@Author: wmz
@Contact: wmengzhao@qq.com
@File: python.cpp
@Time: 2022/11/12 9:57 AM
@Desc: python interface
***************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "opt_set.h"

#define NUMPY_C_STYLE py::array::c_style | py:array::forcecast

namespace py = pybind11;

using WeightLearning::Metric;
using WeightLearning::OptSet;

using array_float = py::array_t<float, py::array::c_style | py::array::forcecast>;
using array_int = py::array_t<int, py::array::c_style | py::array::forcecast>;
using array_id = py::array_t<unsigned, py::array::c_style | py::array::forcecast>;

PYBIND11_MODULE(pymswl, m) {
    m.doc() = "This is a Python binding of C++ Maia Library for Multimodal Search Weight Learning";
    m.def("data_align", &WeightLearning::data_align, "Data alignment for instruction accelerating");

    // Metric
    py::enum_<Metric>(m, "Metric")
            .value("L2_Float", Metric::L2_Float)
            .value("IP_Float", Metric::IP_Float)
            .value("AS_Int_Skip0", Metric::AS_Int_Skip0)
            .value("AS_Int", Metric::AS_Int)
            .value("IP_Float_AS_Int_Skip0", Metric::IP_Float_AS_Int_Skip0)
            .value("IP_Float_AS_Int", Metric::IP_Float_AS_Int);

    // OptSet
    py::class_<OptSet>(m, "OptSet")
            .def(py::init([](Metric metric = Metric::IP_Float, unsigned thread_num = 1,
                             unsigned is_norm = 0) {
                auto *index = new OptSet(metric, thread_num, is_norm);
                return index;
            }), py::arg("metric") = Metric::IP_Float, py::arg("thread_num") = 1, py::arg("is_norm") = 0)

            .def("load_weight", [](OptSet &index, float w1, float w2, float w3, float w4) {
                index.load_weight(w1, w2, w3, w4);
            })
            .def("load_base_float", [](OptSet &index, array_float &base) {
                index.load_base_float(const_cast<float *>(base.data()), base.shape()[0], base.shape()[1]);
            })

            .def("load_base_int", [](OptSet &index, const array_int &base) {
                index.load_base_int(base.data(), base.shape()[0], base.shape()[1]);
            })

            .def("load_query_float", [](OptSet &index, array_float &query) {
                index.load_query_float(const_cast<float *>(query.data()), query.shape()[0], query.shape()[1]);
            })

            .def("load_query_int", [](OptSet &index, const array_int &query) {
                index.load_query_int(query.data(), query.shape()[0], query.shape()[1]);
            })

            .def("load_1float_2int", [](OptSet &index, array_float &base_modal1, array_float &query_modal1,
                                        const array_int &base_modal2, const array_int &query_modal2) {
                assert(base_modal1.shape()[0] == base_modal2.shape()[0]);
                assert(query_modal1.shape()[0] == query_modal2.shape()[0]);
                assert(base_modal1.shape()[1] == query_modal1.shape()[1]);
                assert(base_modal2.shape()[1] == query_modal2.shape()[1]);
                index.load_1float_2int(const_cast<float *>(base_modal1.data()),
                                       const_cast<float *>(query_modal1.data()),
                                       base_modal2.data(), query_modal2.data(),
                                       base_modal1.shape()[0], query_modal1.shape()[0],
                                       base_modal1.shape()[1],base_modal2.shape()[1]);
            })
            .def("load_1float_2int_345float", [](OptSet &index, array_float &base_modal1, array_float &query_modal1,
                    const array_int &base_modal2, const array_int &query_modal2,
                    array_float &base_modal3, array_float &query_modal3,
                    array_float &base_modal4, array_float &query_modal4) {
                assert(base_modal1.shape()[0] == base_modal2.shape()[0]);
                assert(query_modal1.shape()[0] == query_modal2.shape()[0]);
                assert(base_modal1.shape()[1] == query_modal1.shape()[1]);
                assert(base_modal2.shape()[1] == query_modal2.shape()[1]);
                index.load_1float_2int_345float(const_cast<float *>(base_modal1.data()),
                                       const_cast<float *>(query_modal1.data()),
                                       base_modal2.data(), query_modal2.data(),
                                       const_cast<float *>(base_modal3.data()),
                                       const_cast<float *>(query_modal3.data()),
                                       const_cast<float *>(base_modal4.data()),
                                       const_cast<float *>(query_modal4.data()),
                                       base_modal1.shape()[0], query_modal1.shape()[0],
                                       base_modal1.shape()[1], base_modal2.shape()[1], base_modal3.shape()[1],
                                       base_modal4.shape()[1]);
            })
            .def("load_1float_2float", [](OptSet &index, array_float &base_modal1, array_float &query_modal1,
                                          array_float &base_modal2, array_float &query_modal2) {
                assert(base_modal1.shape()[0] == base_modal2.shape()[0]);
                assert(query_modal1.shape()[0] == query_modal2.shape()[0]);
                assert(base_modal1.shape()[1] == query_modal1.shape()[1]);
                assert(base_modal2.shape()[1] == query_modal2.shape()[1]);
                index.load_1float_2float(const_cast<float *>(base_modal1.data()),
                                         const_cast<float *>(query_modal1.data()),
                                         const_cast<float *>(base_modal2.data()),
                                         const_cast<float *>(query_modal2.data()),
                                         base_modal1.shape()[0], query_modal1.shape()[0],
                                         base_modal1.shape()[1], base_modal2.shape()[1]);
            })

            .def("float_gen_dist", [](OptSet &index, size_t query_st, size_t query_ed)
                    -> std::vector<std::vector<float>> {
                std::vector<std::vector<float>> dist_list;
                index.float_gen_dist(query_st, query_ed, dist_list);
                return dist_list;
            })

            .def("float_dist_by_id", [](OptSet &index, size_t id_1, size_t id_2)
                   -> float {
                float dist;
                index.float_dist_by_id(id_1, id_2, dist);
                return dist;
            })
            .def("int_gen_dist", [](OptSet &index, size_t query_st, size_t query_ed)
                    -> std::vector<std::vector<float>> {
                std::vector<std::vector<float>> dist_list;
                index.int_gen_dist(query_st, query_ed, dist_list);
                return dist_list;
            })
            .def("float_int_topk_id", [](OptSet &index, size_t query_st, size_t query_ed, array_id &k)
                    -> std::vector<std::vector<unsigned>> {
                std::vector<std::vector<unsigned>> id_list;
                index.float_int_topk_id(query_st, query_ed, k.data(), id_list);
                return id_list;
            })
            .def("int_dist_by_id", [](OptSet &index, size_t id_1, size_t id_2)
                    -> float {
                float dist;
                index.int_dist_by_id(id_1, id_2, dist);
                return dist;
})
            .def("float_float_topk_id", [](OptSet &index, size_t query_st, size_t query_ed, array_id &k)
                    -> std::vector<std::vector<unsigned>> {
                std::vector<std::vector<unsigned>> id_list;
                index.float_float_topk_id(query_st, query_ed, k.data(), id_list);
                return id_list;
            })
            .def("modal1_dist_by_id", [](OptSet &index, size_t query_id, array_id &base_id)
                    -> std::vector<float> {
                std::vector<float> dist_list;
                index.modal1_dist_by_id(query_id, base_id.data(), base_id.size(), dist_list);
                return dist_list;
            })
            .def("modal2_int_dist_by_id", [](OptSet &index, size_t query_id, array_id &base_id)
                    -> std::vector<float> {
                std::vector<float> dist_list;
                index.modal2_int_dist_by_id(query_id, base_id.data(), base_id.size(), dist_list);
                return dist_list;
            })
            .def("modal2_float_dist_by_id", [](OptSet &index, size_t query_id, array_id &base_id)
                    -> std::vector<float> {
                std::vector<float> dist_list;
                index.modal2_float_dist_by_id(query_id, base_id.data(), base_id.size(), dist_list);
                return dist_list;
            })
            .def("modal3_dist_by_id", [](OptSet &index, size_t query_id, array_id &base_id)
                    -> std::vector<float> {
                std::vector<float> dist_list;
                index.modal3_dist_by_id(query_id, base_id.data(), base_id.size(), dist_list);
                return dist_list;
            })
            .def("modal4_dist_by_id", [](OptSet &index, size_t query_id, array_id &base_id)
                    -> std::vector<float> {
                std::vector<float> dist_list;
                index.modal4_dist_by_id(query_id, base_id.data(), base_id.size(), dist_list);
                return dist_list;
            });
}