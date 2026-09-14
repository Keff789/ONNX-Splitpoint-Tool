cmake_minimum_required(VERSION 3.16)
project(splitpoint_native_hailo_trt_fifo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV QUIET)

find_library(NVINFER_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)
find_library(HAILORT_LIB hailort HINTS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu /usr/local/hailo/lib)

if(NOT NVINFER_LIB)
  message(FATAL_ERROR "TensorRT nvinfer library not found")
endif()
if(NOT HAILORT_LIB)
  message(FATAL_ERROR "HailoRT library not found")
endif()

add_executable(splitpoint_hailo_trt_fifo hailo_trt_fifo_runner.cpp)

target_include_directories(splitpoint_hailo_trt_fifo PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}
  /usr/include
  /usr/local/include
  /usr/local/hailo/include
)

target_link_libraries(splitpoint_hailo_trt_fifo PRIVATE
  CUDA::cudart
  ${NVINFER_LIB}
  ${HAILORT_LIB}
)

if(NVINFER_PLUGIN_LIB)
  target_link_libraries(splitpoint_hailo_trt_fifo PRIVATE ${NVINFER_PLUGIN_LIB})
endif()

if(OpenCV_FOUND)
  target_compile_definitions(splitpoint_hailo_trt_fifo PRIVATE HAVE_OPENCV=1)
  target_link_libraries(splitpoint_hailo_trt_fifo PRIVATE ${OpenCV_LIBS})
else()
  message(WARNING "OpenCV not found: image-dir mode is disabled; synthetic input still works")
endif()

target_compile_options(splitpoint_hailo_trt_fifo PRIVATE -O3 -DNDEBUG)
