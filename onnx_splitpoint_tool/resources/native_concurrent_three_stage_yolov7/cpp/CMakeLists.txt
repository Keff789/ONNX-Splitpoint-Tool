cmake_minimum_required(VERSION 3.16)
project(onnx_splitpoint_yolov7_three_stage_canary LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(HailoRT REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu /usr/include /usr/src/tensorrt/include /usr/local/TensorRT/include)
find_library(NVINFER_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/TensorRT/lib)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/TensorRT/lib)
find_library(CUDART_LIB cudart HINTS /usr/local/cuda/lib64 /usr/lib/aarch64-linux-gnu /usr/lib)
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h HINTS /usr/local/cuda/include /usr/include)

if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB OR NOT NVINFER_PLUGIN_LIB OR NOT CUDART_LIB OR NOT CUDA_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find TensorRT/CUDA runtime libraries")
endif()

add_library(onnx_splitpoint_three_stage_canary SHARED three_stage_canary.cpp)
target_include_directories(onnx_splitpoint_three_stage_canary PRIVATE
    ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(onnx_splitpoint_three_stage_canary PRIVATE
    HailoRT::libhailort Threads::Threads ${NVINFER_LIB} ${NVINFER_PLUGIN_LIB}
    ${CUDART_LIB} ${OpenCV_LIBS})
target_compile_options(onnx_splitpoint_three_stage_canary PRIVATE
    -O3 -DNDEBUG -Wno-deprecated-declarations)
set_target_properties(onnx_splitpoint_three_stage_canary PROPERTIES
    OUTPUT_NAME "onnx_splitpoint_three_stage_canary")
