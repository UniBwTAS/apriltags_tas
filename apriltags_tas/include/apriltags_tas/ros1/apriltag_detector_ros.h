#pragma once

#include <image_geometry/pinhole_camera_model.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <apriltags/TagDetector.h>

#include <apriltag_ros/ros1/common_functions.h>
#include <apriltags_tas/AprilTagDetectorConfig.h>
#include <apriltags_tas/apriltag_detector.h>

class AprilTagDetectorROS
{
  public:
    AprilTagDetectorROS(const bool use_test_image,
                        const sensor_msgs::CameraInfo::ConstPtr& camera_info,
                        apriltag_ros::TagDetector& tag_config);

    void reconfigure(apriltags_tas::AprilTagDetectorConfig& config, uint32_t level);
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);

    ros::Publisher detections_pub_;
    ros::Publisher image_pub_;

  private:
    void process(const cv::Mat& image);

    bool getPose(AprilTags::TagDetection& tag, geometry_msgs::Pose& pose) noexcept;

    void publishTagDetections(std::vector<AprilTags::TagDetection>& tag_detections, std_msgs::Header header) noexcept;
    void publishTfTransform(std::vector<AprilTags::TagDetection>& tag_detections, std_msgs::Header header) noexcept;

    const bool use_test_image_;
    image_geometry::PinholeCameraModel camera_model_;
    cv::Mat image_;
    std_msgs::Header img_header_;

    apriltag_ros::TagDetector tag_config_;
    Config config_;

    AprilTagDetector core_;
};
