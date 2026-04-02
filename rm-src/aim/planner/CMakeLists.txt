add_subdirectory(tinympc)

add_library(rm_planner STATIC
    planner.cpp
)

target_include_directories(rm_planner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../identify/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../communicate
)

target_link_libraries(rm_planner PUBLIC
    tinympcstatic
    ${OpenCV_LIBS}
    yaml-cpp
    Eigen3::Eigen
)
