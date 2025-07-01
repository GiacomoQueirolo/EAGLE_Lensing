cmake_minimum_required(VERSION 2.8)

# Adapted for the test

####
# project settings
####

project(Example)

set(sample_VERSION_MAJOR 1)
set(sample_VERSION_MINOR 0)


####
# headers
####

set(HEADERS
)


####
# sources
####

set(SOURCES
	example_image.cpp
)


####
# dependencies
####
find_package(GLAMER NO_MODULE REQUIRED)

include_directories(${GLAMER_INCLUDE_DIRS})


####
# build
####

add_executable(Example ${HEADERS} ${SOURCES})

target_link_libraries(Example ${GLAMER_LIBRARIES})
