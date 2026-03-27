cmake_minimum_required(VERSION 3.10)
project(MPC_Planner)

# 指定 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED )
find_package(yaml-cpp REQUIRED)
#引入 TinyMPC 子模块
add_subdirectory(tinympc) 

#源文件打包成一个静态库
set(PLANNER_SOURCES
    planner.cpp
    ../src/TargetState.cpp
    ../src/Target.cpp
)

add_library(planner_lib STATIC ${PLANNER_SOURCES} )

# 5. 指定头文件包含路径
target_include_directories(planner_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}             # 包含 planner.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../include  # 包含 yaml.hpp 和 TargetState.hpp
)

# 6. 【关键补充】将 Eigen3 和 TinyMPC 链接到你的 planner 库上
target_link_libraries(planner_lib PUBLIC 
    tinympcstatic      # TinyMPC 静态库
    ${OpenCV_LIBS}
    yaml-cpp
)