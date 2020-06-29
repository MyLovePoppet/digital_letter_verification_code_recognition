cmake_minimum_required(VERSION 3.10)
project(digital_letter_verification_code_recognition)

set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR "D:\\shuqy-package\\OpenCV\\mingw_build\\install")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(digital_letter_verification_code_recognition BPNN.cpp)
# add libs you need
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_ml)
# linking
target_link_libraries(digital_letter_verification_code_recognition ${OpenCV_LIBS})