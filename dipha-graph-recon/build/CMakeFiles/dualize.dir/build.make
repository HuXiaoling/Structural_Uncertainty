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
include CMakeFiles/dualize.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dualize.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dualize.dir/flags.make

CMakeFiles/dualize.dir/src/dualize.cpp.o: CMakeFiles/dualize.dir/flags.make
CMakeFiles/dualize.dir/src/dualize.cpp.o: ../src/dualize.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dualize.dir/src/dualize.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dualize.dir/src/dualize.cpp.o -c /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/src/dualize.cpp

CMakeFiles/dualize.dir/src/dualize.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dualize.dir/src/dualize.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/src/dualize.cpp > CMakeFiles/dualize.dir/src/dualize.cpp.i

CMakeFiles/dualize.dir/src/dualize.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dualize.dir/src/dualize.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/src/dualize.cpp -o CMakeFiles/dualize.dir/src/dualize.cpp.s

CMakeFiles/dualize.dir/src/dualize.cpp.o.requires:

.PHONY : CMakeFiles/dualize.dir/src/dualize.cpp.o.requires

CMakeFiles/dualize.dir/src/dualize.cpp.o.provides: CMakeFiles/dualize.dir/src/dualize.cpp.o.requires
	$(MAKE) -f CMakeFiles/dualize.dir/build.make CMakeFiles/dualize.dir/src/dualize.cpp.o.provides.build
.PHONY : CMakeFiles/dualize.dir/src/dualize.cpp.o.provides

CMakeFiles/dualize.dir/src/dualize.cpp.o.provides.build: CMakeFiles/dualize.dir/src/dualize.cpp.o


# Object files for target dualize
dualize_OBJECTS = \
"CMakeFiles/dualize.dir/src/dualize.cpp.o"

# External object files for target dualize
dualize_EXTERNAL_OBJECTS =

dualize: CMakeFiles/dualize.dir/src/dualize.cpp.o
dualize: CMakeFiles/dualize.dir/build.make
dualize: /usr/lib/x86_64-linux-gnu/libmpichcxx.so
dualize: /usr/lib/x86_64-linux-gnu/libmpich.so
dualize: CMakeFiles/dualize.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable dualize"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dualize.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dualize.dir/build: dualize

.PHONY : CMakeFiles/dualize.dir/build

CMakeFiles/dualize.dir/requires: CMakeFiles/dualize.dir/src/dualize.cpp.o.requires

.PHONY : CMakeFiles/dualize.dir/requires

CMakeFiles/dualize.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dualize.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dualize.dir/clean

CMakeFiles/dualize.dir/depend:
	cd /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build/CMakeFiles/dualize.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dualize.dir/depend

