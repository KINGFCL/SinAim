set(LIBXR_NO_EIGEN ON CACHE BOOL "Use project Eigen instead of libxr bundled Eigen" FORCE)

add_subdirectory(libxr)
add_subdirectory(SharedTopic)
add_subdirectory(SharedTopicClient)

add_library(rm_communicate STATIC
    logger.cpp
)

target_include_directories(rm_communicate PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/SharedTopic
    ${CMAKE_CURRENT_SOURCE_DIR}/SharedTopicClient
    ${CMAKE_CURRENT_SOURCE_DIR}/../aim/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../identify/include
)

target_link_libraries(rm_communicate PUBLIC
    xr
    Eigen3::Eigen
    ${OpenCV_LIBS}
    yaml-cpp
)
