# Rerun SDK
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../Visualisation/rerun_sdk_install")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../../Visualisation/rerun_viewer/rerun"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
find_package(rerun_sdk REQUIRED)

add_library(rm_communicate STATIC
    logger.cpp
)

target_include_directories(rm_communicate PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../aim/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../identify/include
)

target_link_libraries(rm_communicate PUBLIC
    ${OpenCV_LIBS}
    yaml-cpp
    rerun_sdk
)
