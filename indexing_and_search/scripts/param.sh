#!/bin/sh

source dataset.sh

dataset

# possible options: "MultimodalSimilarityAggregationGraph"("MSAG"), "MultimodalSimilarityAggregationBruteforce"("MSAB"),
# "MultichannelAggregationByGraph"("MABG"), "MultichannelAggregationByBruteforce"("MABB")
STRATEGY="MSAG"
INDEX_PREFIX_PATH=../doc/index/${DATASET}
INDEX_PATH=${INDEX_PREFIX_PATH}/${DATASET}_${STRATEGY}.index
IS_NORM_MODAL1=1 # 0 for siftsmall; 1 for celeba
IS_NORM_MODAL2=0 # 0 for siftsmall, celeba,
IS_SKIP_NUM=1 # 1 for celeba, 0 for siftsmall
SKIP_NUM=0 # 0 for celeba
IS_MULTI_RESULT_EQUAL=1  # 1 for celeba, 0 for siftsmall
BUILD_THREAD_NUM=64
SEARCH_THREAD_NUM=1
TOPK=10
GTK=10

param_siftsmall_MSAG() {
  W1=-0.5
  W2=-0.5
  L_candidate=100
  R_neighbor=100
  C_neighbor=200
  k_init_graph=100
  nn_size=50
  rnn_size=25
  pool_size=200
  iter=8
  sample_num=100
  graph_quality_threshold=0.8
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  IS_SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=0
  IS_DELETE_ID=0

  L_search=200
}

param_celeba_MSAG() {
  #trig_encode, W1=-0.106381, W2=-0.641439;
  W1=-0.084799
  W2=-1.185460
  L_candidate=100
  R_neighbor=30
  C_neighbor=200
  k_init_graph=100
  nn_size=50
  rnn_size=25
  pool_size=200
  iter=8
  sample_num=100
  graph_quality_threshold=0.8
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  IS_SKIP_NUM=1
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=1
  IS_DELETE_ID=1

  TOPK=1
  GTK=1
  L_search=5000
}

param_siftsmall_MSAB() {
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  SEARCH_THREAD_NUM=8
  W1=-0.7
  W2=-0.1
  IS_SKIP_NUM=0
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=0
  IS_DELETE_ID=0
  TOPK=10
  GTK=10
}

param_celeba_MSAB() {
  SEARCH_THREAD_NUM=64
  W1=0
  W2=-0.9
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  IS_SKIP_NUM=1
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=1
  IS_DELETE_ID=1
  TOPK=(1)
  GTK=1
}

param_shopping100k_MSAB() {
  SEARCH_THREAD_NUM=32
  W1=-0.009221
  W2=-1.204230
#  W1=-0.026219
#  W2=-1.212432
  IS_SKIP_NUM=1
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=1
  IS_DELETE_ID=1
  TOPK=(1)
  GTK=1
}

param_fashioniq_MSAB() {
  SEARCH_THREAD_NUM=32
  W1=-0.65018
  W2=-0.18705
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=1
  IS_SKIP_NUM=0
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=1
  IS_DELETE_ID=1
  TOPK=1
  GTK=1
}

param_mitstates_MSAB() {
  SEARCH_THREAD_NUM=64
  W1=-0.9 #0.512587
  W2=-0.1 #0.020306
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=1
  IS_SKIP_NUM=0
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=1
  IS_DELETE_ID=1
  TOPK=(1)
  GTK=1
}

param_siftsmall_MABG() {
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  IS_SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=0
  IS_DELETE_ID=0
  W1=-1
  W2=-1
  L_candidate=100
  R_neighbor=30
  C_neighbor=200
  k_init_graph=100
  nn_size=50
  rnn_size=25
  pool_size=200
  iter=8
  sample_num=100
  graph_quality_threshold=0.8
  INDEX_PATH_MODAL1=${INDEX_PREFIX_PATH}/${DATASET}_${STRATEGY}_modal1.index
  INDEX_PATH_MODAL2=${INDEX_PREFIX_PATH}/${DATASET}_${STRATEGY}_modal2.index
  CANDIDATE_TOPK=200

  TOPK=10
  GTK=10
  L_search=200
}

param_celeba_MABG() {
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  IS_SKIP_NUM=1
  IS_MULTI_RESULT_EQUAL=1
  IS_DELETE_ID=1
  W1=-1
  W2=-1
  L_candidate=100
  R_neighbor=30
  C_neighbor=200
  k_init_graph=100
  nn_size=50
  rnn_size=25
  pool_size=200
  iter=8
  sample_num=100
  graph_quality_threshold=0.8
  INDEX_PATH_MODAL1=${INDEX_PREFIX_PATH}/${DATASET}_${STRATEGY}_modal1.index
  INDEX_PATH_MODAL2=${INDEX_PREFIX_PATH}/${DATASET}_${STRATEGY}_modal2.index

  CANDIDATE_TOPK=5
  TOPK=1
  GTK=1
  L_search=20
}

param_mitstates_MSAG() {
  #trig_encode, W1=-0.106381, W2=-0.641439;
  W1=-0.001177
  W2=-1.429084
  L_candidate=100
  R_neighbor=30
  C_neighbor=200
  k_init_graph=100
  nn_size=50
  rnn_size=25
  pool_size=200
  iter=8
  sample_num=100
  graph_quality_threshold=0.8
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=1
  IS_SKIP_NUM=0
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=1
  IS_DELETE_ID=1

  TOPK=1
  GTK=1
  L_search=1000
}

param_siftsmall_MABB() {
  SEARCH_THREAD_NUM=8
  W1=-0.5
  W2=-0.5
  IS_SKIP_NUM=0
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=0
  IS_DELETE_ID=0
  CANDIDATE_TOPK=50
  TOPK=10
  GTK=10
}

param_celeba_MABB() {
  SEARCH_THREAD_NUM=96
  W1=-1
  W2=-1
  IS_SKIP_NUM=1
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=1
  IS_DELETE_ID=1
  CANDIDATE_TOPK=(1400)
  TOPK=1
  GTK=1
}

param_celeba+_MABB() {
  SEARCH_THREAD_NUM=64
  W1=-1
  W2=-1
  IS_NORM_MODAL3=1
  IS_NORM_MODAL4=1
  IS_SKIP_NUM=1
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=1
  IS_DELETE_ID=1
  CANDIDATE_TOPK=(4600 4700 4800 4900)
  TOPK=1
  GTK=1
}

param_celeba+_MSAB() {
  SEARCH_THREAD_NUM=64
  W1=-0.409155
  W2=-3.136325
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  IS_NORM_MODAL3=1
  IS_NORM_MODAL4=1
  IS_SKIP_NUM=1
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=1
  IS_DELETE_ID=1
  TOPK=(1)
  GTK=1
}

param_mitstates_MABB() {
  SEARCH_THREAD_NUM=64
  W1=-1
  W2=-1
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=1
  IS_SKIP_NUM=0
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=1
  IS_DELETE_ID=1
  CANDIDATE_TOPK=(240)
  TOPK=1
  GTK=1
}

param_shopping100k_MABB() {
  SEARCH_THREAD_NUM=32
  W1=-1
  W2=-1
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=1
  IS_SKIP_NUM=0
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=1
  IS_DELETE_ID=1
  CANDIDATE_TOPK=(3)
  TOPK=1
  GTK=1
}

param_sift1m_MSAB() {
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  SEARCH_THREAD_NUM=1
  W1=-0.119856
  W2=-0.557178
  IS_SKIP_NUM=0
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=0
  IS_DELETE_ID=0
  TOPK=10
  GTK=10
}

param_sift1m_MABB() {
  SEARCH_THREAD_NUM=1
  W1=-1
  W2=-1
  IS_SKIP_NUM=0
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=0
  IS_DELETE_ID=0
  CANDIDATE_TOPK=(100 1200 1800 2500 2800 2900 3000 3100 3200 3300 3500)
  TOPK=10
  GTK=10
}

param_sift1m_MSAG() {
  W1=-0.119856
  W2=-0.557178
  L_candidate=100
  R_neighbor=20
  C_neighbor=200
  k_init_graph=100
  nn_size=50
  rnn_size=25
  pool_size=200
  iter=8
  sample_num=100
  graph_quality_threshold=0.8
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  IS_SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=0
  IS_DELETE_ID=0

  L_search=(100, 300, 500, 700, 1000, 1500, 2000, 4000)
  TOPK=10
  GTK=10
}

param_sift1m_MABG() {
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  IS_SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=0
  IS_DELETE_ID=0
  W1=-1
  W2=-1
  L_candidate=100
  R_neighbor=30
  C_neighbor=200
  k_init_graph=100
  nn_size=50
  rnn_size=25
  pool_size=200
  iter=8
  sample_num=100
  graph_quality_threshold=0.8
  INDEX_PATH_MODAL1=${INDEX_PREFIX_PATH}/${DATASET}_${STRATEGY}_modal1.index
  INDEX_PATH_MODAL2=${INDEX_PREFIX_PATH}/${DATASET}_${STRATEGY}_modal2.index
  CANDIDATE_TOPK=(1000 3000 5000 8000 10000 10500 11000 11500 12000)

  TOPK=100
  GTK=100
  L_search=11000
}

param_msong1m_MSAB() {
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  SEARCH_THREAD_NUM=32
  W1=-0.045283
  W2=-0.858904
  IS_SKIP_NUM=0
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=0
  IS_DELETE_ID=0
  TOPK=10
  GTK=10
}

param_msong1m_MSAG() {
  W1=-0.045283
  W2=-0.858904
  L_candidate=100
  R_neighbor=30
  C_neighbor=200
  k_init_graph=100
  nn_size=50
  rnn_size=25
  pool_size=200
  iter=8
  sample_num=100
  graph_quality_threshold=0.8
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  IS_SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=0
  IS_DELETE_ID=0

  L_search=3700
  TOPK=10
  GTK=10
}

param_msong1m_MABG() {
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  IS_SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=0
  IS_DELETE_ID=0
  W1=-1
  W2=-1
  L_candidate=100
  R_neighbor=30
  C_neighbor=200
  k_init_graph=100
  nn_size=50
  rnn_size=25
  pool_size=200
  iter=8
  sample_num=100
  graph_quality_threshold=0.8
  INDEX_PATH_MODAL1=${INDEX_PREFIX_PATH}/${DATASET}_${STRATEGY}_modal1.index
  INDEX_PATH_MODAL2=${INDEX_PREFIX_PATH}/${DATASET}_${STRATEGY}_modal2.index
  CANDIDATE_TOPK=(100 300 700 1200 1800 2500 3000 3300 3500)

  TOPK=10
  GTK=10
  L_search=3500
}

param_uqv1m_MSAB() {
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  SEARCH_THREAD_NUM=32
  W1=-0.310647
  W2=-0.443985
  IS_SKIP_NUM=0
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=0
  IS_DELETE_ID=0
  TOPK=10
  GTK=10
}

param_uqv1m_MSAG() {
  W1=-0.310647
  W2=-0.443985
  L_candidate=100
  R_neighbor=30
  C_neighbor=200
  k_init_graph=100
  nn_size=50
  rnn_size=25
  pool_size=200
  iter=8
  sample_num=100
  graph_quality_threshold=0.8
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  IS_SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=0
  IS_DELETE_ID=0

  L_search=5000
  TOPK=10
  GTK=10
}

param_uqv1m_MABG() {
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  IS_SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=0
  IS_DELETE_ID=0
  W1=-1
  W2=-1
  L_candidate=100
  R_neighbor=30
  C_neighbor=200
  k_init_graph=100
  nn_size=50
  rnn_size=25
  pool_size=200
  iter=8
  sample_num=100
  graph_quality_threshold=0.8
  INDEX_PATH_MODAL1=${INDEX_PREFIX_PATH}/${DATASET}_${STRATEGY}_modal1.index
  INDEX_PATH_MODAL2=${INDEX_PREFIX_PATH}/${DATASET}_${STRATEGY}_modal2.index
  CANDIDATE_TOPK=(100 300 700 1200 1800 2500 3000 3300 3500)

  TOPK=10
  GTK=10
  L_search=3500
}

param_ImageText1M_MABB() {
  SEARCH_THREAD_NUM=48
  W1=-1
  W2=-1
  IS_SKIP_NUM=0
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=0
  IS_DELETE_ID=0
  CANDIDATE_TOPK=3500
  TOPK=10
  GTK=10
}

param_deep16m_MSAB() {
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  SEARCH_THREAD_NUM=1
  W1=-0.112320
  W2=-0.874206
  IS_SKIP_NUM=0
  SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=0
  IS_DELETE_ID=0
  TOPK=10
  GTK=10
}

param_deep16m_MSAG() {
  W1=-0.112320
  W2=-0.874206

  L_candidate=100
  R_neighbor=48
  C_neighbor=200
  k_init_graph=100
  nn_size=50
  rnn_size=25
  pool_size=200
  iter=8
  sample_num=100
  graph_quality_threshold=0.8
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  IS_SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=0
  IS_DELETE_ID=0

  L_search=(4000 5000 6000 7000 8000 9000 10000)
  TOPK=10
  GTK=10
}

param_deep16m_MABG() {
  IS_NORM_MODAL1=1
  IS_NORM_MODAL2=0
  IS_SKIP_NUM=0
  IS_MULTI_RESULT_EQUAL=0
  IS_DELETE_ID=0
  W1=-1
  W2=-1
  L_candidate=100
  R_neighbor=48
  C_neighbor=200
  k_init_graph=100
  nn_size=50
  rnn_size=25
  pool_size=200
  iter=8
  sample_num=100
  graph_quality_threshold=0.8
  INDEX_PATH_MODAL1=${INDEX_PREFIX_PATH}/${DATASET}_${STRATEGY}_modal1.index
  INDEX_PATH_MODAL2=${INDEX_PREFIX_PATH}/${DATASET}_${STRATEGY}_modal2.index
  CANDIDATE_TOPK=(1000 3000 5000 8000 10000 10500 11000 11500 12000)

  TOPK=100
  GTK=100
  L_search=11000
}