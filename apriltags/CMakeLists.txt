cmake_minimum_required(VERSION 3.8)
project(apriltags)

set(APRILTAGS_SOURCES
    src/Edge.cc
    src/GLine2D.cc
    src/GLineSegment2D.cc
    src/GrayModel.cc
    src/Homography33.cc
    src/MathUtil.cc
    src/Quad.cc
    src/Segment.cc
    src/TagDetection.cc
    src/TagDetectionUtils.cc
    src/TagDetector.cc
    src/TagFamily.cc
    src/UnionFindSimple.cc
)

find_package(OpenCV REQUIRED)

if(DEFINED ENV{ROS_VERSION} AND "$ENV{ROS_VERSION}" STREQUAL "1")
    find_package(catkin REQUIRED COMPONENTS cmake_modules)
    find_package(Eigen REQUIRED)

    catkin_package(
        INCLUDE_DIRS include
        LIBRARIES apriltags
        DEPENDS Eigen OpenCV
    )

    include_directories(
        include
        include/AprilTags
        ${Eigen_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
    )

    add_library(apriltags ${APRILTAGS_SOURCES})
    target_link_libraries(apriltags
        ${Eigen_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
    )

    install(TARGETS apriltags
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )

    install(DIRECTORY include/
        DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h"
    )
elseif(DEFINED ENV{ROS_VERSION} AND "$ENV{ROS_VERSION}" STREQUAL "2")
    find_package(ament_cmake REQUIRED)
    find_package(Eigen3 REQUIRED)

    add_library(apriltags SHARED ${APRILTAGS_SOURCES})
    target_include_directories(apriltags PUBLIC
        include
        include/AprilTags
        ${EIGEN3_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(apriltags
        ${OpenCV_LIBRARIES}
    )

    install(TARGETS apriltags
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
    install(DIRECTORY include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
    )

    ament_export_include_directories(include)
    ament_export_libraries(apriltags)
    ament_package()
else()
    message(FATAL_ERROR "ROS_VERSION not set to 1 or 2. Please source a ROS environment first.")
endif()
