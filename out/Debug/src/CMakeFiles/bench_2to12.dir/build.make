# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /nix/store/z9w9jh0pq94vlfzsiyb3g6lrai8ba2px-cmake-3.29.2/bin/cmake

# The command to remove a file.
RM = /nix/store/z9w9jh0pq94vlfzsiyb3g6lrai8ba2px-cmake-3.29.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/duda/Downloads/crypto/lib/fq_arithmetic

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug

# Include any dependencies generated for this target.
include src/CMakeFiles/bench_2to12.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/bench_2to12.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/bench_2to12.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/bench_2to12.dir/flags.make

src/CMakeFiles/bench_2to12.dir/2to12/bench.cpp.o: src/CMakeFiles/bench_2to12.dir/flags.make
src/CMakeFiles/bench_2to12.dir/2to12/bench.cpp.o: /home/duda/Downloads/crypto/lib/fq_arithmetic/src/2to12/bench.cpp
src/CMakeFiles/bench_2to12.dir/2to12/bench.cpp.o: src/CMakeFiles/bench_2to12.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/bench_2to12.dir/2to12/bench.cpp.o"
	cd /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug/src && /nix/store/wk45q17zzwwpsgary6hfc368567kmwiy-clang-wrapper-17.0.6/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/bench_2to12.dir/2to12/bench.cpp.o -MF CMakeFiles/bench_2to12.dir/2to12/bench.cpp.o.d -o CMakeFiles/bench_2to12.dir/2to12/bench.cpp.o -c /home/duda/Downloads/crypto/lib/fq_arithmetic/src/2to12/bench.cpp

src/CMakeFiles/bench_2to12.dir/2to12/bench.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/bench_2to12.dir/2to12/bench.cpp.i"
	cd /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug/src && /nix/store/wk45q17zzwwpsgary6hfc368567kmwiy-clang-wrapper-17.0.6/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/duda/Downloads/crypto/lib/fq_arithmetic/src/2to12/bench.cpp > CMakeFiles/bench_2to12.dir/2to12/bench.cpp.i

src/CMakeFiles/bench_2to12.dir/2to12/bench.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/bench_2to12.dir/2to12/bench.cpp.s"
	cd /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug/src && /nix/store/wk45q17zzwwpsgary6hfc368567kmwiy-clang-wrapper-17.0.6/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/duda/Downloads/crypto/lib/fq_arithmetic/src/2to12/bench.cpp -o CMakeFiles/bench_2to12.dir/2to12/bench.cpp.s

# Object files for target bench_2to12
bench_2to12_OBJECTS = \
"CMakeFiles/bench_2to12.dir/2to12/bench.cpp.o"

# External object files for target bench_2to12
bench_2to12_EXTERNAL_OBJECTS =

src/bench_2to12: src/CMakeFiles/bench_2to12.dir/2to12/bench.cpp.o
src/bench_2to12: src/CMakeFiles/bench_2to12.dir/build.make
src/bench_2to12: /nix/store/1z9cg2zfpqj4fjpl12ign18yqn0r8y6g-gbenchmark-1.8.3/lib/libbenchmark.a
src/bench_2to12: src/CMakeFiles/bench_2to12.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bench_2to12"
	cd /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bench_2to12.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/bench_2to12.dir/build: src/bench_2to12
.PHONY : src/CMakeFiles/bench_2to12.dir/build

src/CMakeFiles/bench_2to12.dir/clean:
	cd /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug/src && $(CMAKE_COMMAND) -P CMakeFiles/bench_2to12.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/bench_2to12.dir/clean

src/CMakeFiles/bench_2to12.dir/depend:
	cd /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/duda/Downloads/crypto/lib/fq_arithmetic /home/duda/Downloads/crypto/lib/fq_arithmetic/src /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug/src /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug/src/CMakeFiles/bench_2to12.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/CMakeFiles/bench_2to12.dir/depend

