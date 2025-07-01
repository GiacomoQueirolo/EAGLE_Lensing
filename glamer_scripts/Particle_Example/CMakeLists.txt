cmake_minimum_required(VERSION 3.0...3.8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

####
# project settings
####

project(TestParicles)

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
	testparticles.cpp
)


####
# dependencies
####

find_package(GLAMER NO_MODULE REQUIRED)

include_directories(${GLAMER_INCLUDE_DIRS})


####
# build
####

add_executable(TestParticles ${HEADERS} ${SOURCES})

target_link_libraries(TestParticles ${GLAMER_LIBRARIES})