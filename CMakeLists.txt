cmake_minimum_required(VERSION 3.11)
project(fq_arithmetic LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(default_build_type "Release")

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

# import needed  
include(FetchContent)
include(ExternalProject)
include(CheckCXXCompilerFlag) 	# check for compiler optimization flags
include(CheckIPOSupported)  	# check for -flto
include(CheckLanguage) 			# needed for `check_language(CUDA)`
find_package(benchmark REQUIRED)

FetchContent_Declare(
  cmake_optimize_for_architecture
  GIT_REPOSITORY https://github.com/FloydZ/cmake_optimize_for_architecture
  GIT_TAG     	 704bd3e39fabf5533ac00f4cab41281fd15fedbc
)
#FetchContent_MakeAvailable(cmake_optimize_for_architecture)
set(AVX2_FLAGS "-DUSE_AVX2 -mavx2 -mavx -mbmi2 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 ${AVX2_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${AVX2_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0 ${AVX2_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${AVX2_FLAGS}")

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
