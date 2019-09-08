cmake_minimum_required(VERSION 3.10)
project(edgedetector)

set(CMAKE_CXX_STANDARD 14)

include(CMakeLists.local.txt OPTIONAL)

find_package(OpenCV REQUIRED)

find_package(OpenMP REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(edgedetector main.cpp Sobel/Sobel.cpp Sobel/Sobel.h Sobel/ompSobel.cpp Sobel/ompSobel.h Canny/Canny.cpp Canny/Canny.h Canny/ompCanny.cpp Canny/ompCanny.h)
target_link_libraries(edgedetector ${OpenCV_LIBS})