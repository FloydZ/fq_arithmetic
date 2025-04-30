cmake_minimum_required(VERSION 3.20.1)
project(cuda_fp_arithmetic)

enable_language(CUDA)

# enable languages
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED True)

# verbose output and create `compile_commands.json`
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

# global include path for dependencies
# maybe a little ugly, but the easiest way to include `cuda_runtime.h`
# as far as i know this is only needed for arch linux.
include_directories(/opt/cuda/samples/common/inc)
# again only for arch... fuck NVIDIA
include_directories(/opt/cuda/samples/cuda-samples/Common)
include_directories(/usr/local/cuda/samples/cuda-samples/Common)
include_directories(/opt/cuda/include)
include_directories(deps/cudacommon)
include_directories(deps/cuCollections/include)
include_directories(/usr/local/cuda/samples/common/inc)



add_executable(test_limb2 test/limb2/test.cu)
add_executable(test_limb3 test/limb3/test.cu)
add_executable(test_limb4 test/limb4/test.cu)
add_executable(test_limb5 test/limb5/test.cu)
add_executable(test_limb6 test/limb6/test.cu)
add_executable(test_limb7 test/limb7/test.cu)
add_executable(test_limb8 test/limb8/test.cu)

add_executable(bench bench/clock.cu)
