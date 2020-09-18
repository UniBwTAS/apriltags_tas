cmake_minimum_required(VERSION 3.9)
project(apriltags_tas)

find_package(catkin REQUIRED
        COMPONENTS
        roscpp
        cmake_modules
        dynamic_reconfigure
        apriltags
        apriltags_msgs
        xmlrpcpp
        tf2_ros
        cv_bridge
        image_transport
        image_geometry
        tf
        )

find_package(OpenCV 3 REQUIRED PATHS "/opt/opencv/" "/usr" NO_DEFAULT_PATH) # QUIET
include_directories(BEFORE SYSTEM ${OpenCV_INCLUDE_DIRS})

include_directories(include include/apriltag_ros)
include_directories(include include/apriltag_tas)
include_directories(${catkin_INCLUDE_DIRS})

generate_dynamic_reconfigure_options(cfg/AprilTagDetector.cfg)

add_executable(apriltag_detector
        src/apriltag_detector_node.cpp
        src/apriltag_detector.cpp
        src/common_functions.cpp
        )
target_link_libraries(apriltag_detector
        ${OpenCV_LIBS}
        ceres
        glog
        ${catkin_LIBRARIES}
        )

add_dependencies(apriltag_detector ${${PROJECT_NAME}_EXPORTED_TARGETS})