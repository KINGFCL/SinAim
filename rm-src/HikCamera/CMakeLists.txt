list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(MVS REQUIRED)

add_library(rm_hikcamera STATIC
    HikCamera.cpp
)

target_include_directories(rm_hikcamera PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${MVS_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../communicate
)

target_link_libraries(rm_hikcamera PUBLIC
    ${OpenCV_LIBS}
    ${MVS_LIBS}
)
