cmake_minimum_required(VERSION 3.10)
project(MyProject VERSION 1.0 LANGUAGES CXX)

#Release 模式 (这会带来 -O3 和 -DNDEBUG)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

#在 Release 的基础上，加上硬件特化指令（-march=native）
add_compile_options(-march=native -O3)

# 指定 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 3. 第三步：现在可以安心引入子模块了，planner 会完美继承上面的所有暴力优化
add_subdirectory(planner)



list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# ==========================================
# 1. 链接预编译好的 Rerun 静态库
# ==========================================
# 指向我们刚刚 install 出来的那个文件夹
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../Visualisation/rerun_sdk_install")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../Visualisation/rerun_viewer/rerun" 
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")  # 将rerun渲染文程序复制到构建目录

# 现在 CMake 瞬间就能找到它，不需要再编译了！
find_package(rerun_sdk REQUIRED)


find_package(OpenCV REQUIRED )
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(MVS REQUIRED)
find_package(PkgConfig REQUIRED)
# 找到 OpenVINO 包
# 需要先 source openvino/setupvars.sh
find_package(OpenVINO REQUIRED)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

target_include_directories(main PUBLIC ${MVS_INCLUDE_DIRS})
target_link_libraries(main planner_lib ${OpenCV_LIBS} fmt::fmt spdlog::spdlog ${MVS_LIBS} openvino::runtime rerun_sdk)

target_include_directories(smallmain PUBLIC ${MVS_INCLUDE_DIRS})
target_link_libraries(smallmain planner_lib ${OpenCV_LIBS} fmt::fmt spdlog::spdlog ${MVS_LIBS} rerun_sdk)

target_include_directories(mpcmain PUBLIC ${MVS_INCLUDE_DIRS})
target_link_libraries(mpcmain PUBLIC planner_lib ${OpenCV_LIBS} fmt::fmt spdlog::spdlog ${MVS_LIBS} rerun_sdk)

#target_include_directories(Yolomain PUBLIC ${MVS_INCLUDE_DIRS})
#target_link_libraries(Yolomain ${OpenCV_LIBS} fmt::fmt spdlog::spdlog ${MVS_LIBS} openvino::runtime rerun_sdk)






