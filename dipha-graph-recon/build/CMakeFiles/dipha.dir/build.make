# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build

# Include any dependencies generated for this target.
include CMakeFiles/dipha.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dipha.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dipha.dir/flags.make

CMakeFiles/dipha.dir/src/dipha.cpp.o: CMakeFiles/dipha.dir/flags.make
CMakeFiles/dipha.dir/src/dipha.cpp.o: ../src/dipha.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dipha.dir/src/dipha.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dipha.dir/src/dipha.cpp.o -c /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/src/dipha.cpp

CMakeFiles/dipha.dir/src/dipha.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dipha.dir/src/dipha.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/src/dipha.cpp > CMakeFiles/dipha.dir/src/dipha.cpp.i

CMakeFiles/dipha.dir/src/dipha.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dipha.dir/src/dipha.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/src/dipha.cpp -o CMakeFiles/dipha.dir/src/dipha.cpp.s

CMakeFiles/dipha.dir/src/dipha.cpp.o.requires:

.PHONY : CMakeFiles/dipha.dir/src/dipha.cpp.o.requires

CMakeFiles/dipha.dir/src/dipha.cpp.o.provides: CMakeFiles/dipha.dir/src/dipha.cpp.o.requires
	$(MAKE) -f CMakeFiles/dipha.dir/build.make CMakeFiles/dipha.dir/src/dipha.cpp.o.provides.build
.PHONY : CMakeFiles/dipha.dir/src/dipha.cpp.o.provides

CMakeFiles/dipha.dir/src/dipha.cpp.o.provides.build: CMakeFiles/dipha.dir/src/dipha.cpp.o


# Object files for target dipha
dipha_OBJECTS = \
"CMakeFiles/dipha.dir/src/dipha.cpp.o"

# External object files for target dipha
dipha_EXTERNAL_OBJECTS =

dipha: CMakeFiles/dipha.dir/src/dipha.cpp.o
dipha: CMakeFiles/dipha.dir/build.make
dipha: /usr/lib/x86_64-linux-gnu/libmpichcxx.so
dipha: /usr/lib/x86_64-linux-gnu/libmpich.so
dipha: CMakeFiles/dipha.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable dipha"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dipha.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dipha.dir/build: dipha

.PHONY : CMakeFiles/dipha.dir/build

CMakeFiles/dipha.dir/requires: CMakeFiles/dipha.dir/src/dipha.cpp.o.requires

.PHONY : CMakeFiles/dipha.dir/requires

CMakeFiles/dipha.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dipha.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dipha.dir/clean

CMakeFiles/dipha.dir/depend:
	cd /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build/CMakeFiles/dipha.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dipha.dir/depend

