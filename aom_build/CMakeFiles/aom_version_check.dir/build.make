# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/chenjs/a5/aom_190109/aom

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chenjs/a5/aom_190109/aom_build

# Utility rule file for aom_version_check.

# Include the progress variables for this target.
include CMakeFiles/aom_version_check.dir/progress.make

CMakeFiles/aom_version_check:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/chenjs/a5/aom_190109/aom_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Updating version info if necessary."
	/usr/bin/cmake -DAOM_CONFIG_DIR=/home/chenjs/a5/aom_190109/aom_build -DAOM_ROOT=/home/chenjs/a5/aom_190109/aom -DGIT_EXECUTABLE=/usr/bin/git -DPERL_EXECUTABLE=/usr/bin/perl -P /home/chenjs/a5/aom_190109/aom/build/cmake/version.cmake

aom_version_check: CMakeFiles/aom_version_check
aom_version_check: CMakeFiles/aom_version_check.dir/build.make

.PHONY : aom_version_check

# Rule to build all files generated by this target.
CMakeFiles/aom_version_check.dir/build: aom_version_check

.PHONY : CMakeFiles/aom_version_check.dir/build

CMakeFiles/aom_version_check.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/aom_version_check.dir/cmake_clean.cmake
.PHONY : CMakeFiles/aom_version_check.dir/clean

CMakeFiles/aom_version_check.dir/depend:
	cd /home/chenjs/a5/aom_190109/aom_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chenjs/a5/aom_190109/aom /home/chenjs/a5/aom_190109/aom /home/chenjs/a5/aom_190109/aom_build /home/chenjs/a5/aom_190109/aom_build /home/chenjs/a5/aom_190109/aom_build/CMakeFiles/aom_version_check.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/aom_version_check.dir/depend

