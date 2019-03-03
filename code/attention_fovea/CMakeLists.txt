cmake_minimum_required(VERSION 2.8)
project(yolt)

set(OpenCV_DIR /home/pedro/opencv/build/)
set(CLASSIFICATION OFF)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/")

# Include package config, boost-python
find_package(PkgConfig REQUIRED)
include(cmake/boost-python.cmake)
#include(cmake/Modules/FindCaffe.cmake)
#  add_definitions(-DOPENCV_3)

if(${CLASSIFICATION})
  find_package(Caffe REQUIRED)
  find_package(CUDA REQUIRED)
endif(${CLASSIFICATION})
find_package(Boost COMPONENTS system filesystem regex python REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(NumPy REQUIRED)
find_package(PythonLibs REQUIRED)


option(-std "std" c++11)

include_directories(
	include
#	${Caffe_INCLUDE_DIRS}
	${PYTHON_INCLUDE_DIR}
	${PYTHON_NUMPY_INCLUDE_DIR}
)

#message("Caffe_INCLUDE_DIRS is ${Caffe_INCLUDE_DIRS}") 
#message("PYTHON_INCLUDE_DIR is ${PYTHON_INCLUDE_DIR}")
#message("PYTHON_NUMPY_INCLUDE_DIR is ${PYTHON_NUMPY_INCLUDE_DIR}")

#add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY
add_definitions(-DCPU_ONLY)

# ADD OUR OWN EXECUTABLE FILES
set(GLOG_LIB glog)

if(${CLASSIFICATION})
  add_executable(
	yolt 
	src/yolt.cpp
	src/laplacian_foveation.cpp
	src/network_classes.cpp
  )  

  target_link_libraries(
	yolt
	${Boost_LIBRARIES} 
	${Caffe_LIBRARIES} 
	${PYTHON_LIBRARIES}
	${GLOG_LIB}  
	${OpenCV_LIBS}
  )
endif(${CLASSIFICATION})


# FOVEATION PYTHON BINDINGS

boost_python_module(np_opencv_converter src/python_bindings/np_opencv_converter.cpp src/python_bindings/conversion.cpp)
target_link_libraries(np_opencv_converter boost_system boost_python ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARIES})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
boost_python_module(np_opencv_module src/python_bindings/np_opencv_module.cpp)
target_link_libraries(np_opencv_module boost_system boost_python np_opencv_converter ${OpenCV_LIBS})

add_library(yolt_python SHARED src/python_bindings/yolt_python.cpp src/laplacian_foveation.cpp)
target_link_libraries(yolt_python -lboost_python -lboost_system ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARIES})

set_target_properties(yolt_python PROPERTIES SUFFIX .so)
set_target_properties(yolt_python PROPERTIES PREFIX "")


