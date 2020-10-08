cmake_minimum_required(VERSION 2.8.3)
project(apriltags_msgs)

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
