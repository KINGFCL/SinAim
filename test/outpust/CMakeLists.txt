cmake_minimum_required(VERSION 3.16)
project(OutPustTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Visualisation/rerun_sdk_install")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../../Visualisation/rerun_viewer/rerun"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
find_package(rerun_sdk REQUIRED)

set(RM_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../rm-src")

add_subdirectory(${RM_SRC_DIR}/identify ${CMAKE_CURRENT_BINARY_DIR}/identify)
add_subdirectory(${RM_SRC_DIR}/communicate ${CMAKE_CURRENT_BINARY_DIR}/communicate)
add_subdirectory(${RM_SRC_DIR}/aim ${CMAKE_CURRENT_BINARY_DIR}/aim)

add_executable(outpust_test test.cpp)

target_include_directories(outpust_test PRIVATE
    ${RM_SRC_DIR}
    ${RM_SRC_DIR}/aim/include
    ${RM_SRC_DIR}/identify/include
    ${RM_SRC_DIR}/communicate
)

target_link_libraries(outpust_test PRIVATE
    rm_aim
    rm_identify
    rm_communicate
    rerun_sdk
    Eigen3::Eigen
    yaml-cpp
    fmt::fmt
    spdlog::spdlog
    openvino::runtime
    ${OpenCV_LIBS}
)
