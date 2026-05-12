cmake_minimum_required(VERSION 3.10) # Minimum CMake version required
# Set the project name and version
project(cv) # Project name is 'cv'

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS True)

# Find all package that are required
find_package( OpenCV REQUIRED ) # Find OpenCV package
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(MVS REQUIRED)
find_package(PkgConfig REQUIRED)
# 找到 OpenVINO 包
# 需要先 source openvino/setupvars.sh
find_package(OpenVINO REQUIRED)

# find_package(LibUSB 1.0 REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(RM_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../rm-src")
add_subdirectory(${RM_SRC_DIR}/communicate ${CMAKE_CURRENT_BINARY_DIR}/communicate)

add_executable(data_autoSDK 
    src/HikCamera.cpp
    src/logger.cpp
    src/data_autoSDK.cpp
)
add_executable(Calibration src/Calibration.cpp)

target_include_directories(data_autoSDK PUBLIC
    ${MVS_INCLUDE_DIRS}
    ${RM_SRC_DIR}/communicate
)
target_link_libraries(data_autoSDK
    ${OpenCV_LIBS}
    fmt::fmt
    spdlog::spdlog
    ${MVS_LIBS}
    rm_communicate
    Eigen3::Eigen
    yaml-cpp
)
# add the executable target

#link the OpenCV libraries to the executable
target_link_libraries(Calibration ${OpenCV_LIBS}) # Link OpenCV libraries to the cv executable







