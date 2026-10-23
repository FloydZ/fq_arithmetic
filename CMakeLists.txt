cmake_minimum_required(VERSION 3.11)
project(fq_arithmetic LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(default_build_type "Release")
enable_testing()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# needed for nvim autocompletion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES 
      ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

# only needed for souper
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# import needed  
include(FetchContent)
include(ExternalProject)
include(CheckCXXCompilerFlag) 	# check for compiler optimization flags
include(CheckIPOSupported)  	# check for -flto
include(CheckLanguage) 			# needed for `check_language(CUDA)`
find_package(benchmark QUIET)

# optional CUDA implementations (src/*/arith.cu), tested with googletest and
# benchmarked with nvbench. Both dependencies are fetched by CMake. This block
# runs before SIMD_FLAGS (incl. -fsanitize=address) are appended to the global
# C/CXX flags, so googletest and nvbench are built with clean flags.
option(FQ_ARITHMETIC_CUDA "build the CUDA implementations, tests and benchmarks" ON)
if(FQ_ARITHMETIC_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            # requires CMake >= 3.24; override with -DCMAKE_CUDA_ARCHITECTURES=86 etc.
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.15.2
            GIT_SHALLOW    TRUE)

        set(NVBENCH_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(NVBENCH_ENABLE_TESTING  OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(nvbench
            GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
            GIT_TAG        main
            GIT_SHALLOW    TRUE)

        FetchContent_MakeAvailable(googletest nvbench)
    else()
        message(STATUS "No CUDA compiler found, skipping the CUDA targets")
    endif()
endif()

IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64") 
	message(STATUS "ARM CPU detected")
    set(SIMD_FLAGS "-DUSE_NEON -flax-vector-conversions -march=native")
    # set(SIMD_FLAGS "-DUSE_M4 -mcpu=cortex-m4 -mthumb -g")
else()
    set(SIMD_FLAGS "-DUSE_AVX2 -mavx2 -mavx -mbmi2 -mpclmul -march=native -fsanitize=address -Wall -Wno-unused-function")
endif()


#set(SIMD_FLAGS " ${SIMD_FLAGS} -DUSE_AVX512 -mavx512f -mavx512bw -mavx512vl -mavx512vbmi -mvpclmulqdq -mgfni")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 ${SIMD_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${SIMD_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0 ${SIMD_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${SIMD_FLAGS}")

MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()


add_subdirectory(src)
