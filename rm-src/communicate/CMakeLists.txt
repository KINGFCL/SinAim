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
)
