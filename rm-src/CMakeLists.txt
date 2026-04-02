cmake_minimum_required(VERSION 3.10)
project(RoboMaster_AutoAim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-O3 -march=native)

# 查找依赖包
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenVINO REQUIRED)

# 添加子模块
add_subdirectory(communicate)
add_subdirectory(HikCamera)
add_subdirectory(identify)
add_subdirectory(aim)

# 可选：编译示例程序
option(BUILD_EXAMPLE "Build example program" OFF)
if(BUILD_EXAMPLE)
    add_executable(rm_example example_main.cpp)
    target_link_libraries(rm_example
        rm_aim
        rm_identify
        rm_communicate
        rm_hikcamera
    )
endif()
