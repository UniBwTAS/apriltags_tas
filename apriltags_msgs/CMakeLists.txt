cmake_minimum_required(VERSION 3.8)
project(apriltags_msgs)

if(DEFINED ENV{ROS_VERSION} AND "$ENV{ROS_VERSION}" STREQUAL "1")
  find_package(catkin REQUIRED COMPONENTS
    std_msgs
    geometry_msgs
    sensor_msgs
    message_generation
  )

  add_message_files(
    FILES
    AprilTagDetections.msg
    AprilTag.msg
  )

  generate_messages(
    DEPENDENCIES
    std_msgs
    geometry_msgs
    sensor_msgs
  )

  catkin_package(
      CATKIN_DEPENDS message_runtime
  )
elseif(DEFINED ENV{ROS_VERSION} AND "$ENV{ROS_VERSION}" STREQUAL "2")
  find_package(ament_cmake REQUIRED)
  find_package(rosidl_default_generators REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)

  rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/AprilTag.msg"
    "msg/AprilTagDetections.msg"
    DEPENDENCIES std_msgs geometry_msgs sensor_msgs
  )

  ament_export_dependencies(rosidl_default_runtime)
  ament_package()
else()
  message(FATAL_ERROR "ROS_VERSION not set to 1 or 2. Please source a ROS environment first.")
endif()
