#pragma once

#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <apriltags_msgs/msg/april_tag_detections.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <opencv2/opencv.hpp>
#if __has_include(<image_geometry/pinhole_camera_model.hpp>)
#include <image_geometry/pinhole_camera_model.hpp>
#else
#include <image_geometry/pinhole_camera_model.h>
#endif
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

#include <apriltags/TagDetector.h>
#include <apriltag_ros/ros2/common_functions.h>
#include <apriltags_tas/apriltag_detector.h>

class AprilTagDetectorROS2
{
  public:
    AprilTagDetectorROS2(const rclcpp::Node::SharedPtr& node,
                     const bool use_test_image,
                     const sensor_msgs::msg::CameraInfo::SharedPtr& camera_info,
                     apriltag_ros::TagDetector& tag_config);

    void reconfigure(Config& config, uint32_t level);
    rcl_interfaces::msg::SetParametersResult reconfigure(const std::vector<rclcpp::Parameter>& parameters);
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void republishLast();

    rclcpp::Publisher<apriltags_msgs::msg::AprilTagDetections>::SharedPtr detections_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

  private:
    void process(const cv::Mat& image);

    bool getPose(AprilTags::TagDetection& tag, geometry_msgs::msg::Pose& pose) noexcept;

    void publishTagDetections(std::vector<AprilTags::TagDetection>& tag_detections,
                              std_msgs::msg::Header header) noexcept;
    void publishTfTransform(std::vector<AprilTags::TagDetection>& tag_detections,
                            std_msgs::msg::Header header) noexcept;

    rclcpp::Node::SharedPtr node_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    const bool use_test_image_;
    image_geometry::PinholeCameraModel camera_model_;
    cv::Mat image_;
    std::string image_encoding_{"bgr8"};
    std_msgs::msg::Header img_header_;

    apriltag_ros::TagDetector tag_config_;
    Config config_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;

    sensor_msgs::msg::Image::SharedPtr last_image_msg_;
    std::shared_ptr<apriltags_msgs::msg::AprilTagDetections> last_detections_msg_;

    AprilTagDetector core_;
};
