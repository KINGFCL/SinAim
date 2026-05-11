add_library(rm_communicate STATIC
    logger.cpp
)

if(TARGET xr)
    target_sources(rm_communicate PRIVATE
        LibXRRobotLink.cpp
    )
    target_include_directories(rm_communicate BEFORE PRIVATE
        "${AUTOAIM_BSP_DIR}/libxr/src/middleware"
    )
endif()

target_include_directories(rm_communicate PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../aim/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../identify/include
)

target_link_libraries(rm_communicate PUBLIC
    ${OpenCV_LIBS}
    yaml-cpp
)

if(TARGET xr)
    target_link_libraries(rm_communicate PRIVATE
        xr
    )
endif()
