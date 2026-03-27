cmake_minimum_required(VERSION 3.16)
project(MPCTest)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 解决 "relocation against ... in read-only section" 的关键：强制启用 -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 1. 寻找必要的依赖库
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
# 新增：寻找日志相关的库
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)

# 2. 添加头文件搜索路径
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rm-main/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rm-main/planner
)

# 3. 引入 planner 的构建逻辑
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rm-main/planner 
    ${CMAKE_CURRENT_BINARY_DIR}/planner_build
)

# 4. 生成可执行文件
# 新增：把报错缺少的 logger.cpp 也一起编译进去
add_executable(MPCTest 
    test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rm-main/src/logger.cpp
)

# 5. 链接相关库
# 新增：链接 spdlog 和 fmt
target_link_libraries(MPCTest
    PRIVATE
    planner_lib
    Eigen3::Eigen
    ${OpenCV_LIBS}
    spdlog::spdlog
    fmt::fmt
)