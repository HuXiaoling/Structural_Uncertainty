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
include CMakeFiles/create_phat_filtration.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/create_phat_filtration.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/create_phat_filtration.dir/flags.make

CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.o: CMakeFiles/create_phat_filtration.dir/flags.make
CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.o: ../src/create_phat_filtration.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.o -c /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/src/create_phat_filtration.cpp

CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/src/create_phat_filtration.cpp > CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.i

CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/src/create_phat_filtration.cpp -o CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.s

CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.o.requires:

.PHONY : CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.o.requires

CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.o.provides: CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.o.requires
	$(MAKE) -f CMakeFiles/create_phat_filtration.dir/build.make CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.o.provides.build
.PHONY : CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.o.provides

CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.o.provides.build: CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.o


# Object files for target create_phat_filtration
create_phat_filtration_OBJECTS = \
"CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.o"

# External object files for target create_phat_filtration
create_phat_filtration_EXTERNAL_OBJECTS =

create_phat_filtration: CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.o
create_phat_filtration: CMakeFiles/create_phat_filtration.dir/build.make
create_phat_filtration: /usr/lib/x86_64-linux-gnu/libmpichcxx.so
create_phat_filtration: /usr/lib/x86_64-linux-gnu/libmpich.so
create_phat_filtration: CMakeFiles/create_phat_filtration.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable create_phat_filtration"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/create_phat_filtration.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/create_phat_filtration.dir/build: create_phat_filtration

.PHONY : CMakeFiles/create_phat_filtration.dir/build

CMakeFiles/create_phat_filtration.dir/requires: CMakeFiles/create_phat_filtration.dir/src/create_phat_filtration.cpp.o.requires

.PHONY : CMakeFiles/create_phat_filtration.dir/requires

CMakeFiles/create_phat_filtration.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/create_phat_filtration.dir/cmake_clean.cmake
.PHONY : CMakeFiles/create_phat_filtration.dir/clean

CMakeFiles/create_phat_filtration.dir/depend:
	cd /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build /home/xhu/projects/uncertainty/morse_DMT/dipha-graph-recon/build/CMakeFiles/create_phat_filtration.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/create_phat_filtration.dir/depend

