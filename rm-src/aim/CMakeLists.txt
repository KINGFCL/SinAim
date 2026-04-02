add_subdirectory(planner)

add_library(rm_aim STATIC
    src/EKFKalman.cpp
    src/Solver.cpp
    src/Target.cpp
    src/TargetState.cpp
    src/Tracker.cpp
)

target_include_directories(rm_aim PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../identify/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../communicate
)

target_link_libraries(rm_aim PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    rm_planner
    rm_identify
)
