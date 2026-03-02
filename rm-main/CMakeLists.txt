cmake_minimum_required(VERSION 3.10)
project(MyProject VERSION 1.0 LANGUAGES CXX)

# add_definitions(-Wall -Werror)

add_compile_options(-march=native -O3)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(OpenCV REQUIRED )
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(MVS REQUIRED)
find_package(PkgConfig REQUIRED)
# 找到 OpenVINO 包
# 需要先 source openvino/setupvars.sh
find_package(OpenVINO REQUIRED)

# find_package(LibUSB 1.0 REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_executable(main 
    main.cpp
    src/HikCamera.cpp
    src/logger.cpp
    src/Detecor.cpp
    src/NumClassifier.cpp
    src/Solver.cpp
    src/Target.cpp
    src/EKFKalman.cpp
    ./Function.cpp
)
target_include_directories(main PUBLIC ${MVS_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS} fmt::fmt spdlog::spdlog ${MVS_LIBS} openvino::runtime)






