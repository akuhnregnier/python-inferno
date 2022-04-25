cmake_minimum_required (VERSION 3.20)
project(python_inferno)

add_subdirectory(gpu_inferno)

install(
    TARGETS py_gpu_inferno
    LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/src/python_inferno/py_gpu_inferno/
)
