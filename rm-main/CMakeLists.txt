cmake_minimum_required(VERSION 3.10)
project(RoboMaster_AutoAim VERSION 1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-march=native -O3)

# ==========================================
# 依赖包
# ==========================================
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(MVS REQUIRED)
find_package(PkgConfig REQUIRED)

# Rerun SDK
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../Visualisation/rerun_sdk_install")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../Visualisation/rerun_viewer/rerun"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
find_package(rerun_sdk REQUIRED)

# ==========================================
# rm-src 子模块（库）
# ==========================================
set(RM_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../rm-src")

add_subdirectory(${RM_SRC_DIR}/communicate ${CMAKE_CURRENT_BINARY_DIR}/communicate)
add_subdirectory(${RM_SRC_DIR}/HikCamera   ${CMAKE_CURRENT_BINARY_DIR}/HikCamera)
add_subdirectory(${RM_SRC_DIR}/identify    ${CMAKE_CURRENT_BINARY_DIR}/identify)
add_subdirectory(${RM_SRC_DIR}/aim         ${CMAKE_CURRENT_BINARY_DIR}/aim)

# ==========================================
# 主程序：只保留三种识别器版本
# ==========================================
set(AUTOAIM_MAIN_LIBS
    rm_aim
    rm_identify
    rm_communicate
    rm_hikcamera
    yaml-cpp
    fmt::fmt
    spdlog::spdlog
    openvino::runtime
    rerun_sdk
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

function(add_autoaim_main target source)
    add_executable(${target} ${source} Function.cpp)

    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${RM_SRC_DIR}
        ${MVS_INCLUDE_DIRS}
    )

    target_link_libraries(${target} PRIVATE ${AUTOAIM_MAIN_LIBS})
endfunction()

add_autoaim_main(mlpmain mlpmain.cpp)
add_autoaim_main(resetmain resetmain.cpp)
add_autoaim_main(yolomain yolomain.cpp)
add_autoaim_main(demomain demompcmain.cpp)
