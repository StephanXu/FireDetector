cmake_minimum_required(VERSION 3.2)

# Set libraries path
set(OpenCV_DIR "/home/lorime/stephanxu/opencv/opencv-3.4.3-cxx/build")
# set(Caffe_DIR "/home/lorime/liuyang/ssd-caffe/build")
set(Caffe_DIR "/home/lorime/caffe/build")
set(Caffe_INCLUDE_DIRS "/home/lorime/caffe/include")

# set project name
project(FireDetector)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb") 
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# test CXX11 support
add_definitions("-std=c++11")
# gcc 5.4 disabled the support of C++11 defaultly
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
     message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# find libraries
find_package(ZLIB REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Caffe REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS} ${ZLIB_INCLUDE_DIRS} ${Caffe_INCLUDE_DIRS} /home/lorime/caffe/build/include
)

aux_source_directory(
    ./src DIR_SRC
)

add_executable(${PROJECT_NAME} ${DIR_SRC})

target_link_libraries(${PROJECT_NAME} ${ZLIB_LIBRARIES} ${OpenCV_LIBS} ${Caffe_LIBRARIES} libtiff.so libuuid.so)

