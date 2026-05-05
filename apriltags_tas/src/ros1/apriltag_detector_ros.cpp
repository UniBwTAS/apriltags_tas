#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_broadcaster.h>

#include <apriltags_msgs/AprilTag.h>
#include <apriltags_msgs/AprilTagDetections.h>

#include <apriltags_tas/apriltag_detector.h>
#include <apriltags_tas/ros1/apriltag_detector_ros.h>

AprilTagDetectorROS::AprilTagDetectorROS(const bool use_test_image,
                                         const sensor_msgs::CameraInfo::ConstPtr& camera_info,
                                         apriltag_ros::TagDetector& tag_config)
    : use_test_image_(use_test_image),
      tag_config_(tag_config),
      core_([](const std::string& msg) { ROS_INFO_STREAM(msg); }, [](const std::string& msg) { ROS_WARN_STREAM(msg); })
{
    if (camera_info)
    {
        camera_model_.fromCameraInfo(camera_info);
    }
}

void AprilTagDetectorROS::reconfigure(apriltags_tas::AprilTagDetectorConfig& config, uint32_t level)
{
    if (config.publish_tf && !camera_model_.initialized())
    {
        config.publish_tf = false;
        ROS_WARN("No camera info available. Publishing tf is disabled.");
    }

    config_.tag_family = static_cast<TagFamily>(config.tag_family);
    config_.refinement_method = static_cast<RefinementMethod>(config.refinement_method);
    config_.only_known_tags = config.only_known_tags;
    config_.filter_cross_corners = config.filter_cross_corners;
    config_.filter_cross_corners_radius_percent = config.filter_cross_corners_radius_percent;
    config_.publish_tf = config.publish_tf;
    config_.draw_image = config.draw_image;

    core_.reconfigure(config_, level);

    if (!image_.empty())
    {
        process(image_);
    }
}

void AprilTagDetectorROS::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    image_ = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
    img_header_ = msg->header;

    process(image_);
}

void AprilTagDetectorROS::process(const cv::Mat& image)
{
    if (!use_test_image_ && detections_pub_.getNumSubscribers() == 0 && image_pub_.getNumSubscribers() == 0)
    {
        ROS_WARN_STREAM("No subscribers => Do not detect tags!");
        return;
    }

    std::vector<AprilTags::TagDetection> tag_detections =
        core_.process(image,
                      [this](int id)
                      {
                          apriltag_ros::TagDescription* tag_description;
                          return tag_config_.findTagDescription(id, tag_description, false);
                      });

    publishTagDetections(tag_detections, img_header_);

    if (config_.publish_tf)
    {
        publishTfTransform(tag_detections, img_header_);
        tag_config_.processBundles(tag_detections, camera_model_, img_header_);
    }

    if (config_.draw_image)
    {
        cv::Mat output_img = image.clone();
        core_.drawTagDetections(output_img, tag_detections);

        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", output_img).toImageMsg();
        image_pub_.publish(msg);
    }
}

bool AprilTagDetectorROS::getPose(AprilTags::TagDetection& tag, geometry_msgs::Pose& pose) noexcept
{
    apriltag_ros::TagDescription* tag_description;
    if (tag_config_.findTagDescription(tag.id, tag_description))
    {
        Eigen::Matrix4d htm = tag.getRelativeTransform(
            tag_description->size, camera_model_.fx(), camera_model_.fy(), camera_model_.cx(), camera_model_.cy());
        Eigen::Matrix3d rot = htm.block(0, 0, 3, 3);
        Eigen::Quaternion<double> rot_quaternion(rot);

        pose.position.x = htm(0, 3);
        pose.position.y = htm(1, 3);
        pose.position.z = htm(2, 3);
        pose.orientation.x = rot_quaternion.x();
        pose.orientation.y = rot_quaternion.y();
        pose.orientation.z = rot_quaternion.z();
        pose.orientation.w = rot_quaternion.w();

        return true;
    }
    else
    {
        return false;
    }
}

void AprilTagDetectorROS::publishTagDetections(std::vector<AprilTags::TagDetection>& tag_detections,
                                               std_msgs::Header header) noexcept
{
    apriltags_msgs::AprilTagDetections detections_msg;
    detections_msg.header = header;
    for (AprilTags::TagDetection& tag : tag_detections)
    {
        apriltags_msgs::AprilTag tag_msg;
        geometry_msgs::Point p;
        p.z = 0;
        for (int i = 0; i < 4; i++)
        {
            p.x = tag.p[i].first;
            p.y = tag.p[i].second;
            tag_msg.corners_px.push_back(p);
        }
        tag_msg.id = std::to_string(tag.id);
        tag_msg.pose_valid = getPose(tag, tag_msg.pose_3d);
        detections_msg.detections.push_back(tag_msg);
    }

    detections_msg.input_image = *cv_bridge::CvImage(header, "rgb8", image_).toImageMsg();
    detections_pub_.publish(detections_msg);
}

void AprilTagDetectorROS::publishTfTransform(std::vector<AprilTags::TagDetection>& tag_detections,
                                             std_msgs::Header header) noexcept
{
    static tf2_ros::TransformBroadcaster tf_broadcaster;

    for (AprilTags::TagDetection& tag : tag_detections)
    {
        apriltag_ros::TagDescription* tag_description;
        if (tag_config_.findTagDescription(tag.id, tag_description) && tag_description->frame_name != "")
        {
            geometry_msgs::Pose pose;
            bool pose_valid = getPose(tag, pose);

            if (pose_valid)
            {
                geometry_msgs::TransformStamped transformStamped;

                transformStamped.header = header;
                transformStamped.child_frame_id = tag_description->frame_name;
                transformStamped.transform.translation.x = pose.position.x;
                transformStamped.transform.translation.y = pose.position.y;
                transformStamped.transform.translation.z = pose.position.z;
                transformStamped.transform.rotation.x = pose.orientation.x;
                transformStamped.transform.rotation.y = pose.orientation.y;
                transformStamped.transform.rotation.z = pose.orientation.z;
                transformStamped.transform.rotation.w = pose.orientation.w;

                tf_broadcaster.sendTransform(transformStamped);
            }
        }
    }
}
