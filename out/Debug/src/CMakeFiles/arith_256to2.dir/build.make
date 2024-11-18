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
include src/CMakeFiles/arith_256to2.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/arith_256to2.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/arith_256to2.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/arith_256to2.dir/flags.make

src/CMakeFiles/arith_256to2.dir/256to2/arith.c.o: src/CMakeFiles/arith_256to2.dir/flags.make
src/CMakeFiles/arith_256to2.dir/256to2/arith.c.o: /home/duda/Downloads/crypto/lib/fq_arithmetic/src/256to2/arith.c
src/CMakeFiles/arith_256to2.dir/256to2/arith.c.o: src/CMakeFiles/arith_256to2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/CMakeFiles/arith_256to2.dir/256to2/arith.c.o"
	cd /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug/src && /nix/store/wk45q17zzwwpsgary6hfc368567kmwiy-clang-wrapper-17.0.6/bin/clang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/CMakeFiles/arith_256to2.dir/256to2/arith.c.o -MF CMakeFiles/arith_256to2.dir/256to2/arith.c.o.d -o CMakeFiles/arith_256to2.dir/256to2/arith.c.o -c /home/duda/Downloads/crypto/lib/fq_arithmetic/src/256to2/arith.c

src/CMakeFiles/arith_256to2.dir/256to2/arith.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/arith_256to2.dir/256to2/arith.c.i"
	cd /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug/src && /nix/store/wk45q17zzwwpsgary6hfc368567kmwiy-clang-wrapper-17.0.6/bin/clang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/duda/Downloads/crypto/lib/fq_arithmetic/src/256to2/arith.c > CMakeFiles/arith_256to2.dir/256to2/arith.c.i

src/CMakeFiles/arith_256to2.dir/256to2/arith.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/arith_256to2.dir/256to2/arith.c.s"
	cd /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug/src && /nix/store/wk45q17zzwwpsgary6hfc368567kmwiy-clang-wrapper-17.0.6/bin/clang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/duda/Downloads/crypto/lib/fq_arithmetic/src/256to2/arith.c -o CMakeFiles/arith_256to2.dir/256to2/arith.c.s

# Object files for target arith_256to2
arith_256to2_OBJECTS = \
"CMakeFiles/arith_256to2.dir/256to2/arith.c.o"

# External object files for target arith_256to2
arith_256to2_EXTERNAL_OBJECTS =

src/arith_256to2: src/CMakeFiles/arith_256to2.dir/256to2/arith.c.o
src/arith_256to2: src/CMakeFiles/arith_256to2.dir/build.make
src/arith_256to2: src/CMakeFiles/arith_256to2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable arith_256to2"
	cd /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/arith_256to2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/arith_256to2.dir/build: src/arith_256to2
.PHONY : src/CMakeFiles/arith_256to2.dir/build

src/CMakeFiles/arith_256to2.dir/clean:
	cd /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug/src && $(CMAKE_COMMAND) -P CMakeFiles/arith_256to2.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/arith_256to2.dir/clean

src/CMakeFiles/arith_256to2.dir/depend:
	cd /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/duda/Downloads/crypto/lib/fq_arithmetic /home/duda/Downloads/crypto/lib/fq_arithmetic/src /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug/src /home/duda/Downloads/crypto/lib/fq_arithmetic/out/Debug/src/CMakeFiles/arith_256to2.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/CMakeFiles/arith_256to2.dir/depend

