cmake_minimum_required(VERSION 3.16)
project(splitpoint_native_hailo_trt_fifo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(SPLITPOINT_WITH_OPENCV "Build with OpenCV preprocessing support" ON)

find_package(CUDA QUIET)
find_library(CUDART_LIB cudart HINTS /usr/local/cuda/lib64 /usr/lib/aarch64-linux-gnu)
find_library(NVINFER_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64)
find_library(HAILORT_LIB hailort HINTS /usr/lib/aarch64-linux-gnu /usr/local/lib)

if(SPLITPOINT_WITH_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
endif()

add_executable(splitpoint_native_hailo_trt_fifo native_hailo_trt_fifo.cpp)

target_include_directories(splitpoint_native_hailo_trt_fifo PRIVATE
  /usr/local/cuda/include
  /usr/include/hailo
  /usr/include/hailort
  /usr/local/include
)

if(OpenCV_FOUND)
  target_compile_definitions(splitpoint_native_hailo_trt_fifo PRIVATE SPLITPOINT_WITH_OPENCV=1)
  target_include_directories(splitpoint_native_hailo_trt_fifo PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(splitpoint_native_hailo_trt_fifo PRIVATE ${OpenCV_LIBS})
else()
  target_compile_definitions(splitpoint_native_hailo_trt_fifo PRIVATE SPLITPOINT_WITH_OPENCV=0)
endif()

if(NOT CUDART_LIB)
  message(FATAL_ERROR "Could not find libcudart")
endif()
if(NOT NVINFER_LIB)
  message(FATAL_ERROR "Could not find libnvinfer")
endif()
if(NOT NVINFER_PLUGIN_LIB)
  message(WARNING "Could not find libnvinfer_plugin; continuing without explicit plugin link")
endif()
if(NOT HAILORT_LIB)
  message(FATAL_ERROR "Could not find libhailort")
endif()

target_link_libraries(splitpoint_native_hailo_trt_fifo PRIVATE
  ${CUDART_LIB}
  ${NVINFER_LIB}
  ${HAILORT_LIB}
  pthread
  dl
)
if(NVINFER_PLUGIN_LIB)
  target_link_libraries(splitpoint_native_hailo_trt_fifo PRIVATE ${NVINFER_PLUGIN_LIB})
endif()
