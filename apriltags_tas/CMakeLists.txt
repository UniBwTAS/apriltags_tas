cmake_minimum_required(VERSION 3.9)
project(apriltags_tas)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

if(DEFINED ENV{ROS_VERSION} AND "$ENV{ROS_VERSION}" STREQUAL "1")
  find_package(catkin REQUIRED COMPONENTS
    roscpp
    dynamic_reconfigure
    std_msgs
    sensor_msgs
    image_geometry
    image_transport
    tf2_ros
    tf
    cv_bridge
    apriltags
    apriltags_msgs)

  generate_dynamic_reconfigure_options(cfg/AprilTagDetector.cfg)

  catkin_package(CATKIN_DEPENDS
    roscpp
    dynamic_reconfigure
    std_msgs
    sensor_msgs
    image_geometry
    image_transport
    tf2_ros
    tf
    cv_bridge
    apriltags
    apriltags_msgs)

  include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})

  add_executable(apriltag_detector
    src/apriltag_detector_node_ros1.cpp
    src/ros1/apriltag_detector_ros.cpp
    src/ros1/common_functions.cpp
    src/core/apriltag_detector.cpp)

  add_dependencies(apriltag_detector ${PROJECT_NAME}_gencfg)

  target_link_libraries(apriltag_detector
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${CERES_LIBRARIES}
    glog)

  install(TARGETS apriltag_detector
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
elseif(DEFINED ENV{ROS_VERSION} AND "$ENV{ROS_VERSION}" STREQUAL "2")
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(tf2_ros REQUIRED)
  find_package(image_transport REQUIRED)
  find_package(image_geometry REQUIRED)
  find_package(cv_bridge REQUIRED)
  find_package(apriltags_msgs REQUIRED)
  find_package(apriltags REQUIRED)
  find_package(rcl_interfaces REQUIRED)
  find_package(yaml-cpp REQUIRED)

  add_executable(apriltag_detector
    src/apriltag_detector_node_ros2.cpp
    src/ros2/apriltag_detector_ros.cpp
    src/ros2/common_functions.cpp
    src/core/apriltag_detector.cpp)

  target_include_directories(apriltag_detector PRIVATE include)
  target_compile_features(apriltag_detector PRIVATE cxx_std_17)

  ament_target_dependencies(apriltag_detector
    rclcpp
    sensor_msgs
    std_msgs
    geometry_msgs
    tf2_ros
    image_transport
    image_geometry
    cv_bridge
    apriltags_msgs
    apriltags
    rcl_interfaces)

  target_link_libraries(apriltag_detector
    ${OpenCV_LIBRARIES}
    ${CERES_LIBRARIES}
    yaml-cpp)

  install(TARGETS apriltag_detector DESTINATION lib/${PROJECT_NAME})
  install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
  install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
  install(DIRECTORY example_images DESTINATION share/${PROJECT_NAME})

  ament_package()
else()
  message(FATAL_ERROR "ROS_VERSION not set to 1 or 2. Please source a ROS environment first.")
endif()
