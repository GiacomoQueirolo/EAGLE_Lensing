cmake_minimum_required(VERSION 3.0...3.8)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

####
# project settings
####

project(test_new)

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
    particles.cpp
)

####
# dependencies
####

find_library(CERF_LIBRARY cerf HINTS /pbs/home/g/gqueirolo/.local/lib64)
#message(CERF_LIBRARY="${CERF_LIBRARY}")
find_path(CERF_INCLUDE_DIR cerf.h HINTS /pbs/home/g/gqueirolo/.local/include)
#message(CERF_INCLUDE_DIR="${CERF_INCLUDE_DIR}")
if(NOT CERF_LIBRARY OR NOT CERF_INCLUDE_DIR)
    message(FATAL_ERROR "libcerf not found. Set CERF_LIBRARY and CERF_INCLUDE_DIR correctly.")
endif()

find_package(GLAMER NO_MODULE REQUIRED)

include_directories(${GLAMER_INCLUDE_DIRS})
include_directories(${CERF_INCLUDE_DIR})
message(GLAMER_LIBRARIES="${GLAMER_LIBRARIES}")
####
# build
####

add_executable(test_new ${HEADERS} ${SOURCES})
target_include_directories(test_new PRIVATE ${CERF_INCLUDE_DIR})
target_link_libraries(test_new ${GLAMER_LIBRARIES} ${CERF_LIBRARY})

