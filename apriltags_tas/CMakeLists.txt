cmake_minimum_required(VERSION 3.9)
project(apriltags_tas)

find_package(catkin_simple REQUIRED)
find_package(Ceres REQUIRED)

catkin_simple()

cs_add_executable(apriltag_detector
    src/apriltag_detector_node.cpp
    src/apriltag_detector.cpp
    src/common_functions.cpp
    )
target_link_libraries(apriltag_detector
    ceres
    glog
    )

add_dependencies(apriltag_detector ${PROJECT_NAME}_gencfg)

cs_install()

cs_export()
