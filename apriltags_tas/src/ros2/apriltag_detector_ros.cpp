#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <geometry_msgs/msg/transform_stamped.hpp>

#include <apriltags_msgs/msg/april_tag.hpp>
#include <apriltags_msgs/msg/april_tag_detections.hpp>

#include <apriltags_tas/apriltag_detector.h>
#include <apriltags_tas/ros2/apriltag_detector_ros.h>

AprilTagDetectorROS2::AprilTagDetectorROS2(const rclcpp::Node::SharedPtr& node,
                                   const bool use_test_image,
                                   const sensor_msgs::msg::CameraInfo::SharedPtr& camera_info,
                                   apriltag_ros::TagDetector& tag_config)
    : node_(node),
      use_test_image_(use_test_image),
      tf_broadcaster_(node_),
      tag_config_(tag_config),
      core_([this](const std::string& msg) { RCLCPP_INFO_STREAM(node_->get_logger(), msg); },
            [this](const std::string& msg) { RCLCPP_WARN_STREAM(node_->get_logger(), msg); })
{
    if (camera_info)
    {
        camera_model_.fromCameraInfo(camera_info);
    }

    parameter_callback_handle_ = node_->add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter>& parameters) { return reconfigure(parameters); });
}

void AprilTagDetectorROS2::reconfigure(Config& config, uint32_t level)
{
    if (config.publish_tf && !camera_model_.initialized())
    {
        config.publish_tf = false;
        RCLCPP_WARN(node_->get_logger(), "No camera info available. Publishing tf is disabled.");
    }

    config_ = config;

    core_.reconfigure(config_, level);

    if (!image_.empty())
    {
        process(image_);
    }
}

rcl_interfaces::msg::SetParametersResult
AprilTagDetectorROS2::reconfigure(const std::vector<rclcpp::Parameter>& parameters)
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    Config updated_config = config_;
    uint32_t level = 0;

    for (const rclcpp::Parameter& parameter : parameters)
    {
        const std::string& name = parameter.get_name();
        try
        {
            if (name == "tag_family")
            {
                TagFamily parsed_tag_family = updated_config.tag_family;
                bool valid = false;
                if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
                {
                    const std::string value = parameter.as_string();
                    if (value == "16h5")
                    {
                        parsed_tag_family = TagFamily::tag16h5;
                        valid = true;
                    }
                    else if (value == "25h7")
                    {
                        parsed_tag_family = TagFamily::tag25h7;
                        valid = true;
                    }
                    else if (value == "25h9")
                    {
                        parsed_tag_family = TagFamily::tag25h9;
                        valid = true;
                    }
                    else if (value == "36h9")
                    {
                        parsed_tag_family = TagFamily::tag36h9;
                        valid = true;
                    }
                    else if (value == "36h11")
                    {
                        parsed_tag_family = TagFamily::tag36h11;
                        valid = true;
                    }
                }
                else if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
                {
                    const int value = static_cast<int>(parameter.as_int());
                    if (value >= static_cast<int>(TagFamily::tag16h5) &&
                        value <= static_cast<int>(TagFamily::tag36h11))
                    {
                        parsed_tag_family = static_cast<TagFamily>(value);
                        valid = true;
                    }
                }
                if (!valid)
                {
                    result.successful = false;
                    result.reason = "'tag_family' must be one of: 16h5, 25h7, 25h9, 36h9, 36h11.";
                    return result;
                }
                if (parsed_tag_family != updated_config.tag_family)
                {
                    level |= 1;
                }
                updated_config.tag_family = parsed_tag_family;
            }
            else if (name == "refinement_method")
            {
                RefinementMethod parsed_refinement_method = updated_config.refinement_method;
                bool valid = false;
                if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
                {
                    const std::string value = parameter.as_string();
                    if (value == "NoRefinement" || value == "none")
                    {
                        parsed_refinement_method = RefinementMethod::NoRefinement;
                        valid = true;
                    }
                    else if (value == "CornerRefinement" || value == "corner")
                    {
                        parsed_refinement_method = RefinementMethod::CornerRefinement;
                        valid = true;
                    }
                    else if (value == "AdvEdgeRefinement" || value == "edge")
                    {
                        parsed_refinement_method = RefinementMethod::AdvEdgeRefinement;
                        valid = true;
                    }
                }
                else if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
                {
                    const int value = static_cast<int>(parameter.as_int());
                    if (value >= static_cast<int>(RefinementMethod::NoRefinement) &&
                        value <= static_cast<int>(RefinementMethod::AdvEdgeRefinement))
                    {
                        parsed_refinement_method = static_cast<RefinementMethod>(value);
                        valid = true;
                    }
                }
                if (!valid)
                {
                    result.successful = false;
                    result.reason =
                        "'refinement_method' must be one of: NoRefinement, CornerRefinement, AdvEdgeRefinement.";
                    return result;
                }
                updated_config.refinement_method = parsed_refinement_method;
            }
            else if (name == "only_known_tags")
            {
                updated_config.only_known_tags = parameter.as_bool();
            }
            else if (name == "filter_cross_corners")
            {
                updated_config.filter_cross_corners = parameter.as_bool();
            }
            else if (name == "filter_cross_corners_radius_percent")
            {
                const double value = parameter.as_double();
                if (value < 0.0 || value > 100.0)
                {
                    result.successful = false;
                    result.reason = "'filter_cross_corners_radius_percent' must be in range [0, 100].";
                    return result;
                }
                updated_config.filter_cross_corners_radius_percent = value;
            }
            else if (name == "publish_tf")
            {
                updated_config.publish_tf = parameter.as_bool();
            }
            else if (name == "draw_image")
            {
                updated_config.draw_image = parameter.as_bool();
            }
        }
        catch (const rclcpp::ParameterTypeException& e)
        {
            result.successful = false;
            result.reason = std::string("Parameter type mismatch for '") + name + "': " + e.what();
            return result;
        }
    }

    reconfigure(updated_config, level);
    return result;
}

void AprilTagDetectorROS2::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    image_ = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
    img_header_ = msg->header;

    process(image_);
}

void AprilTagDetectorROS2::process(const cv::Mat& image)
{
    if (!use_test_image_ && detections_pub_->get_subscription_count() == 0 && image_pub_->get_subscription_count() == 0)
    {
        RCLCPP_WARN_STREAM(node_->get_logger(), "No subscribers => Do not detect tags!");
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

        sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(img_header_, "bgr8", output_img).toImageMsg();
        last_image_msg_ = msg;
        image_pub_->publish(*msg);
    }
}

void AprilTagDetectorROS2::republishLast()
{
    if (last_detections_msg_)
    {
        detections_pub_->publish(*last_detections_msg_);
    }
    if (last_image_msg_)
    {
        image_pub_->publish(*last_image_msg_);
    }
}

bool AprilTagDetectorROS2::getPose(AprilTags::TagDetection& tag, geometry_msgs::msg::Pose& pose) noexcept
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

void AprilTagDetectorROS2::publishTagDetections(std::vector<AprilTags::TagDetection>& tag_detections,
                                            std_msgs::msg::Header header) noexcept
{
    apriltags_msgs::msg::AprilTagDetections detections_msg;
    detections_msg.header = header;
    for (AprilTags::TagDetection& tag : tag_detections)
    {
        apriltags_msgs::msg::AprilTag tag_msg;
        geometry_msgs::msg::Point p;
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
    last_detections_msg_ = std::make_shared<apriltags_msgs::msg::AprilTagDetections>(detections_msg);
    detections_pub_->publish(detections_msg);
}

void AprilTagDetectorROS2::publishTfTransform(std::vector<AprilTags::TagDetection>& tag_detections,
                                          std_msgs::msg::Header header) noexcept
{
    for (AprilTags::TagDetection& tag : tag_detections)
    {
        apriltag_ros::TagDescription* tag_description;
        if (tag_config_.findTagDescription(tag.id, tag_description) && tag_description->frame_name != "")
        {
            geometry_msgs::msg::Pose pose;
            bool pose_valid = getPose(tag, pose);

            if (pose_valid)
            {
                geometry_msgs::msg::TransformStamped transform_stamped;

                transform_stamped.header = header;
                transform_stamped.child_frame_id = tag_description->frame_name;
                transform_stamped.transform.translation.x = pose.position.x;
                transform_stamped.transform.translation.y = pose.position.y;
                transform_stamped.transform.translation.z = pose.position.z;
                transform_stamped.transform.rotation.x = pose.orientation.x;
                transform_stamped.transform.rotation.y = pose.orientation.y;
                transform_stamped.transform.rotation.z = pose.orientation.z;
                transform_stamped.transform.rotation.w = pose.orientation.w;

                tf_broadcaster_.sendTransform(transform_stamped);
            }
        }
    }
}
