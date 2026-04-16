add_library(rm_identify STATIC
    src/CVDetector.cpp
    src/MlpNumClassifier.cpp
    src/ResNetNumClassifier.cpp
    src/YoloDetector.cpp
)

target_include_directories(rm_identify PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../communicate/include
)

target_link_libraries(rm_identify PUBLIC
    ${OpenCV_LIBS}
    openvino::runtime
)
