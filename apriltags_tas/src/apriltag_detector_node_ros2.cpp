#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/wait_for_message.hpp>

#include <apriltag_ros/ros2/common_functions.h>

#include <apriltags_tas/ros2/apriltag_detector_ros.h>

namespace
{

bool parseTagFamilyString(const std::string& value, TagFamily& parsed_value)
{
    if (value == "16h5")
    {
        parsed_value = TagFamily::tag16h5;
        return true;
    }
    if (value == "25h7")
    {
        parsed_value = TagFamily::tag25h7;
        return true;
    }
    if (value == "25h9")
    {
        parsed_value = TagFamily::tag25h9;
        return true;
    }
    if (value == "36h9")
    {
        parsed_value = TagFamily::tag36h9;
        return true;
    }
    if (value == "36h11")
    {
        parsed_value = TagFamily::tag36h11;
        return true;
    }

    return false;
}

bool parseTagFamilyInteger(const int value, TagFamily& parsed_value)
{
    if (value < static_cast<int>(TagFamily::tag16h5) ||
        value > static_cast<int>(TagFamily::tag36h11))
    {
        return false;
    }

    parsed_value = static_cast<TagFamily>(value);
    return true;
}

bool parseRefinementMethodString(const std::string& value, RefinementMethod& parsed_value)
{
    if (value == "NoRefinement" || value == "none")
    {
        parsed_value = RefinementMethod::NoRefinement;
        return true;
    }
    if (value == "CornerRefinement" || value == "corner")
    {
        parsed_value = RefinementMethod::CornerRefinement;
        return true;
    }
    if (value == "AdvEdgeRefinement" || value == "edge")
    {
        parsed_value = RefinementMethod::AdvEdgeRefinement;
        return true;
    }

    return false;
}

bool parseRefinementMethodInteger(const int value, RefinementMethod& parsed_value)
{
    if (value < static_cast<int>(RefinementMethod::NoRefinement) ||
        value > static_cast<int>(RefinementMethod::AdvEdgeRefinement))
    {
        return false;
    }

    parsed_value = static_cast<RefinementMethod>(value);
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("apriltag_detector");

    const std::string camera_info_topic =
        node->declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");
    const std::string image_topic = node->declare_parameter<std::string>("image_topic", "/camera/image");

    const bool use_test_input_image = node->declare_parameter<bool>("use_test_input_image", false);
    const std::string test_input_image_path = node->declare_parameter<std::string>("test_input_image_path", "");

    rcl_interfaces::msg::ParameterDescriptor tag_family_descriptor;
    tag_family_descriptor.description = "AprilTag family";
    tag_family_descriptor.additional_constraints =
        "Allowed strings: 16h5, 25h7, 25h9, 36h9, 36h11 (legacy int values 0..4 are also accepted)";
    tag_family_descriptor.dynamic_typing = true;

    rcl_interfaces::msg::ParameterDescriptor refinement_method_descriptor;
    refinement_method_descriptor.description = "Corner refinement method";
    refinement_method_descriptor.additional_constraints =
        "Allowed strings: NoRefinement|none, CornerRefinement|corner, AdvEdgeRefinement|edge "
        "(legacy int values 0..2 are also accepted)";
    refinement_method_descriptor.dynamic_typing = true;

    Config current_config;

    const rclcpp::ParameterValue tag_family_value =
        node->declare_parameter("tag_family", rclcpp::ParameterValue(std::string("36h11")), tag_family_descriptor);
    const rclcpp::ParameterValue refinement_method_value = node->declare_parameter(
        "refinement_method", rclcpp::ParameterValue(std::string("NoRefinement")), refinement_method_descriptor);

    bool tag_family_ok = false;
    if (tag_family_value.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
    {
        const std::string value = tag_family_value.get<std::string>();
        tag_family_ok = parseTagFamilyString(value, current_config.tag_family);
    }
    else if (tag_family_value.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
    {
        const int value = static_cast<int>(tag_family_value.get<int64_t>());
        tag_family_ok = parseTagFamilyInteger(value, current_config.tag_family);
    }
    if (!tag_family_ok)
    {
        RCLCPP_ERROR(node->get_logger(),
                     "Invalid parameter 'tag_family'. Allowed strings: 16h5, 25h7, 25h9, 36h9, 36h11.");
        rclcpp::shutdown();
        return 1;
    }

    bool refinement_method_ok = false;
    if (refinement_method_value.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
    {
        const std::string value = refinement_method_value.get<std::string>();
        refinement_method_ok = parseRefinementMethodString(value, current_config.refinement_method);
    }
    else if (refinement_method_value.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
    {
        const int value = static_cast<int>(refinement_method_value.get<int64_t>());
        refinement_method_ok = parseRefinementMethodInteger(value, current_config.refinement_method);
    }
    if (!refinement_method_ok)
    {
        RCLCPP_ERROR(node->get_logger(),
                     "Invalid parameter 'refinement_method'. Allowed strings: NoRefinement, CornerRefinement, "
                     "AdvEdgeRefinement.");
        rclcpp::shutdown();
        return 1;
    }

    current_config.only_known_tags = node->declare_parameter<bool>("only_known_tags", false);
    current_config.filter_cross_corners = node->declare_parameter<bool>("filter_cross_corners", true);
    current_config.filter_cross_corners_radius_percent =
        node->declare_parameter<double>("filter_cross_corners_radius_percent", 5.0);
    current_config.publish_tf = node->declare_parameter<bool>("publish_tf", true);
    current_config.draw_image = node->declare_parameter<bool>("draw_image", true);
    current_config.upscale_factor = node->declare_parameter<double>("upscale_factor", 1.0);

    sensor_msgs::msg::CameraInfo::SharedPtr camera_info;

    if (!use_test_input_image)
    {
        sensor_msgs::msg::CameraInfo camera_info_msg;
        RCLCPP_INFO(node->get_logger(), "Waiting for camera info on topic '%s'", camera_info_topic.c_str());

        if (rclcpp::wait_for_message(camera_info_msg, node, camera_info_topic, std::chrono::seconds(5)))
        {
            camera_info = std::make_shared<sensor_msgs::msg::CameraInfo>(camera_info_msg);
        }
    }

    apriltag_ros::TagDetector tag_config(node);

    AprilTagDetectorROS2 apriltag_detector(node, use_test_input_image, camera_info, tag_config);
    apriltag_detector.reconfigure(current_config, 1);

    apriltag_detector.detections_pub_ =
        node->create_publisher<apriltags_msgs::msg::AprilTagDetections>("~/detections", 1);
    apriltag_detector.image_pub_ =
        node->create_publisher<sensor_msgs::msg::Image>("~/image", rclcpp::QoS(1));

    [[maybe_unused]] rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber;
    [[maybe_unused]] rclcpp::TimerBase::SharedPtr test_image_timer;

    if (!use_test_input_image)
    {
        auto image_callback = std::bind(&AprilTagDetectorROS2::imageCallback, &apriltag_detector, std::placeholders::_1);

        image_subscriber = node->create_subscription<sensor_msgs::msg::Image>(
            image_topic, rclcpp::SensorDataQoS().keep_last(1), image_callback);
    }
    else
    {
        cv::Mat img = cv::imread(test_input_image_path);
        if (img.empty())
        {
            RCLCPP_ERROR(node->get_logger(), "Could not load test input image: '%s'", test_input_image_path.c_str());
            rclcpp::shutdown();
            return 1;
        }

        sensor_msgs::msg::Image::SharedPtr image_msg =
            cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img).toImageMsg();
        apriltag_detector.imageCallback(image_msg);

        test_image_timer = node->create_wall_timer(std::chrono::seconds(1), [&]() {
            apriltag_detector.republishLast();
        });
    }
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
