cmake_minimum_required(VERSION 3.2.5)
project(GraphANNS)

set(CMAKE_CXX_STANDARD 17)


#### [required] add CGraph dependence ####
include(cmake/GA-CGraph-depend.cmake)

#### [optional] add Eigen3 dependence ####
include(cmake/GA-Eigen3-depend.cmake)

#### [optional] add OpenMP dependence ####
include(cmake/GA-OpenMP-depend.cmake)


set(TUTORIAL_LIST
        T-MSAG
        S-MSAG
        S-MSAB
        S-MABG
        S-MABB
        )

foreach(tut ${TUTORIAL_LIST})
    add_executable(${tut} ${tut}.cpp ${CGRAPH_SRC_LIST})
endforeach()


#### [optional] add 3rd-party anns pipeline ####
# add_subdirectory(anns-pipeline/3rd-annoy)
# add_subdirectory(anns-pipeline/3rd-hnsw)
