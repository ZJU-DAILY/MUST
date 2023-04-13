set(CMAKE_CXX_STANDARD 11)

# file(GLOB_RECURSE CPP_SOURCES *.cpp)
list(
    APPEND CPP_SOURCES
        opt_set.cpp
        util.cpp
)

add_library(${PROJECT_NAME} ${CPP_SOURCES})
add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
pybind11_add_module(pymswl python.cpp ${CPP_SOURCES})

#install()