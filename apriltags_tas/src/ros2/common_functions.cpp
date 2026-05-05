/**
 * Copyright (c) 2017, California Institute of Technology.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those of the authors and should not be interpreted as representing official
 * policies, either expressed or implied, of the California Institute of
 * Technology.
 */

#include <apriltag_ros/ros2/common_functions.h>

#include <algorithm>
#include <exception>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>

namespace apriltag_ros
{

TagDetector::TagDetector(const rclcpp::Node::SharedPtr& node) : node_(node), tf_pub_(node_)
{
    const std::string tags_config_file = node_->declare_parameter<std::string>("tags_config_file", "");
    if (tags_config_file.empty())
    {
        RCLCPP_WARN(node_->get_logger(), "No tags config file provided (parameter 'tags_config_file').");

        return;
    }

    YAML::Node tags_config;
    try
    {
        tags_config = YAML::LoadFile(tags_config_file);
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(
            node_->get_logger(), "Failed to load tags config file '%s': %s", tags_config_file.c_str(), e.what());

        return;
    }

    if (!tags_config["standalone_tags"])
    {
        RCLCPP_WARN(node_->get_logger(), "No standalone tags specified in '%s'.", tags_config_file.c_str());
    }
    else
    {
        try
        {
            standalone_tag_descriptions_ = parseStandaloneTags(tags_config["standalone_tags"]);
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(node_->get_logger(), "Error loading standalone tags: %s", e.what());
        }
    }

    if (!tags_config["tag_bundles"])
    {
        RCLCPP_WARN(node_->get_logger(), "No tag bundles specified in '%s'.", tags_config_file.c_str());
    }
    else
    {
        try
        {
            tag_bundle_descriptions_ = parseTagBundles(tags_config["tag_bundles"]);
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(node_->get_logger(), "Error loading tag bundles: %s", e.what());
        }
    }
}

void TagDetector::processBundles(std::vector<AprilTags::TagDetection>& tag_detections,
                                 image_geometry::PinholeCameraModel& camera_model,
                                 std_msgs::msg::Header header)
{
    std::map<std::string, std::vector<cv::Point3d>> bundle_object_points;
    std::map<std::string, std::vector<cv::Point2d>> bundle_image_points;

    for (AprilTags::TagDetection& detection : tag_detections)
    {
        const int tag_id = detection.id;
        for (TagBundleDescription& bundle : tag_bundle_descriptions_)
        {
            if (bundle.id2idx_.find(tag_id) != bundle.id2idx_.end())
            {
                const std::string bundle_name = bundle.name();

                const double s = bundle.memberSize(tag_id) / 2;
                addObjectPoints(s, bundle.memberT_oi(tag_id), bundle_object_points[bundle_name]);

                addImagePoints(detection, bundle_image_points[bundle_name]);
            }
        }
    }

    for (TagBundleDescription& bundle : tag_bundle_descriptions_)
    {
        const std::string bundle_name = bundle.name();

        std::map<std::string, std::vector<cv::Point3d>>::iterator it = bundle_object_points.find(bundle_name);
        if (it != bundle_object_points.end())
        {
            Eigen::Matrix4d transform = getRelativeTransform(bundle_object_points[bundle_name],
                                                             bundle_image_points[bundle_name],
                                                             camera_model.fx(),
                                                             camera_model.fy(),
                                                             camera_model.cx(),
                                                             camera_model.cy());
            Eigen::Matrix3d rot = transform.block(0, 0, 3, 3);
            Eigen::Quaternion<double> rot_quaternion(rot);

            geometry_msgs::msg::PoseStamped bundle_pose = makeTagPose(transform, rot_quaternion, header);

            geometry_msgs::msg::TransformStamped transform_stamped;
            transform_stamped.header = header;
            transform_stamped.child_frame_id = bundle_name;
            transform_stamped.transform.translation.x = bundle_pose.pose.position.x;
            transform_stamped.transform.translation.y = bundle_pose.pose.position.y;
            transform_stamped.transform.translation.z = bundle_pose.pose.position.z;
            transform_stamped.transform.rotation = bundle_pose.pose.orientation;

            tf_pub_.sendTransform(transform_stamped);
        }
    }

    RCLCPP_INFO(node_->get_logger(), "Found %zu bundles.", bundle_object_points.size());
}

void TagDetector::addObjectPoints(double s, cv::Matx44d T_oi, std::vector<cv::Point3d>& objectPoints) const
{
    // Add to object point vector the tag corner coordinates in the bundle frame
    // Going counterclockwise starting from the bottom left corner
    objectPoints.push_back(T_oi.get_minor<3, 4>(0, 0) * cv::Vec4d(-s, -s, 0, 1));
    objectPoints.push_back(T_oi.get_minor<3, 4>(0, 0) * cv::Vec4d(s, -s, 0, 1));
    objectPoints.push_back(T_oi.get_minor<3, 4>(0, 0) * cv::Vec4d(s, s, 0, 1));
    objectPoints.push_back(T_oi.get_minor<3, 4>(0, 0) * cv::Vec4d(-s, s, 0, 1));
}

void TagDetector::addImagePoints(AprilTags::TagDetection& detection, std::vector<cv::Point2d>& imagePoints) const
{
    // Add to image point vector the tag corners in the image frame
    // Going counterclockwise starting from the bottom left corner
    for (int i = 0; i < 4; i++)
    {
        // Homography projection taking tag local frame coordinates to image pixels
        imagePoints.push_back(cv::Point2d(detection.p[i].first, detection.p[i].second));
    }
}

Eigen::Matrix4d TagDetector::getRelativeTransform(std::vector<cv::Point3d> objectPoints,
                                                  std::vector<cv::Point2d> imagePoints,
                                                  double fx,
                                                  double fy,
                                                  double cx,
                                                  double cy) const
{
    // perform Perspective-n-Point camera pose estimation using the
    // above 3D-2D point correspondences
    cv::Mat rvec, tvec;
    cv::Matx33d cameraMatrix(fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Vec4f distCoeffs(0, 0, 0, 0); // distortion coefficients
    // TODO Perhaps something like SOLVEPNP_EPNP would be faster? Would
    // need to first check WHAT is a bottleneck in this code, and only
    // do this if PnP solution is the bottleneck.
    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    cv::Matx33d R;
    cv::Rodrigues(rvec, R);
    Eigen::Matrix3d wRo;
    wRo << R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1), R(1, 2), R(2, 0), R(2, 1), R(2, 2);

    Eigen::Matrix4d T; // homogeneous transformation matrix
    T.topLeftCorner(3, 3) = wRo;
    T.col(3).head(3) << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);
    T.row(3) << 0, 0, 0, 1;
    return T;
}

geometry_msgs::msg::PoseStamped TagDetector::makeTagPose(const Eigen::Matrix4d& transform,
                                                         const Eigen::Quaternion<double> rot_quaternion,
                                                         const std_msgs::msg::Header& header)
{
    geometry_msgs::msg::PoseStamped pose;
    pose.header = header;
    //===== Position and orientation
    pose.pose.position.x = transform(0, 3);
    pose.pose.position.y = transform(1, 3);
    pose.pose.position.z = transform(2, 3);
    pose.pose.orientation.x = rot_quaternion.x();
    pose.pose.orientation.y = rot_quaternion.y();
    pose.pose.orientation.z = rot_quaternion.z();
    pose.pose.orientation.w = rot_quaternion.w();
    return pose;
}

// Parse standalone tag descriptions
std::map<int, TagDescription> TagDetector::parseStandaloneTags(const YAML::Node& standalone_tags)
{
    std::map<int, TagDescription> descriptions;
    if (!standalone_tags.IsSequence())
    {
        throw std::runtime_error("'standalone_tags' must be a YAML sequence.");
    }

    for (std::size_t i = 0; i < standalone_tags.size(); i++)
    {
        const YAML::Node tag_description = standalone_tags[i];
        if (!tag_description.IsMap())
        {
            throw std::runtime_error("Each standalone tag entry must be a YAML map.");
        }

        const int id = tag_description["id"].as<int>();
        const double size = tag_description["size"].as<double>();

        std::string frame_name;
        if (tag_description["name"])
        {
            frame_name = tag_description["name"].as<std::string>();
        }
        else
        {
            frame_name = "tag_" + std::to_string(id);
        }

        TagDescription description(id, size, frame_name);
        RCLCPP_INFO(
            node_->get_logger(), "Loaded tag config: %d, size: %f, frame_name: %s", id, size, frame_name.c_str());

        descriptions.insert(std::make_pair(id, description));
    }

    return descriptions;
}

std::vector<TagBundleDescription> TagDetector::parseTagBundles(const YAML::Node& tag_bundles)
{
    std::vector<TagBundleDescription> descriptions;
    if (!tag_bundles.IsSequence())
    {
        throw std::runtime_error("'tag_bundles' must be a YAML sequence.");
    }

    for (std::size_t i = 0; i < tag_bundles.size(); i++)
    {
        const YAML::Node bundle_description = tag_bundles[i];
        if (!bundle_description.IsMap())
        {
            throw std::runtime_error("Each tag bundle entry must be a YAML map.");
        }

        std::string bundle_name;
        if (bundle_description["name"])
        {
            bundle_name = bundle_description["name"].as<std::string>();
        }
        else
        {
            bundle_name = "bundle_" + std::to_string(i);
        }

        TagBundleDescription bundle_i(bundle_name);
        RCLCPP_INFO(node_->get_logger(), "Loading tag bundle '%s'", bundle_i.name().c_str());

        const YAML::Node member_tags = bundle_description["layout"];
        if (!member_tags || !member_tags.IsSequence())
        {
            throw std::runtime_error("Each tag bundle must contain a 'layout' YAML sequence.");
        }

        for (std::size_t j = 0; j < member_tags.size(); j++)
        {
            const YAML::Node tag = member_tags[j];
            if (!tag.IsMap())
            {
                throw std::runtime_error("Each tag bundle layout entry must be a YAML map.");
            }

            const int id = tag["id"].as<int>();
            const double size = tag["size"].as<double>();

            TagDescription* standalone_description;
            if (findTagDescription(id, standalone_description, false) && size != standalone_description->size)
            {
                throw std::runtime_error("Standalone tag size and bundle tag size do not match.");
            }

            const double x = yamlGetDoubleWithDefault(tag, "x", 0.0);
            const double y = yamlGetDoubleWithDefault(tag, "y", 0.0);
            const double z = yamlGetDoubleWithDefault(tag, "z", 0.0);
            const double qw = yamlGetDoubleWithDefault(tag, "qw", 1.0);
            const double qx = yamlGetDoubleWithDefault(tag, "qx", 0.0);
            const double qy = yamlGetDoubleWithDefault(tag, "qy", 0.0);
            const double qz = yamlGetDoubleWithDefault(tag, "qz", 0.0);

            Eigen::Quaterniond q_tag(qw, qx, qy, qz);
            q_tag.normalize();
            Eigen::Matrix3d R_oi = q_tag.toRotationMatrix();

            cv::Matx44d T_mj(R_oi(0, 0),
                             R_oi(0, 1),
                             R_oi(0, 2),
                             x,
                             R_oi(1, 0),
                             R_oi(1, 1),
                             R_oi(1, 2),
                             y,
                             R_oi(2, 0),
                             R_oi(2, 1),
                             R_oi(2, 2),
                             z,
                             0,
                             0,
                             0,
                             1);

            bundle_i.addMemberTag(id, size, T_mj);
            RCLCPP_INFO(node_->get_logger(),
                        " %zu) id: %d, size: %f, p = [%f, %f, %f], q = [%f, %f, %f, %f]",
                        j,
                        id,
                        size,
                        x,
                        y,
                        z,
                        qw,
                        qx,
                        qy,
                        qz);
        }

        descriptions.push_back(bundle_i);
    }

    return descriptions;
}

double
TagDetector::yamlGetDoubleWithDefault(const YAML::Node& yaml_value, const std::string& field, double defaultValue) const
{
    if (!yaml_value[field])
    {
        return defaultValue;
    }

    return yaml_value[field].as<double>();
}

bool TagDetector::findTagDescription(int id, TagDescription*& descriptionContainer, bool printWarning)
{
    std::map<int, TagDescription>::iterator description_itr = standalone_tag_descriptions_.find(id);
    if (description_itr != standalone_tag_descriptions_.end())
    {
        descriptionContainer = &(description_itr->second);
        return true;
    }

    for (TagBundleDescription& tag_bundle : tag_bundle_descriptions_)
    {
        std::vector<int> bundle_ids = tag_bundle.bundleIds();
        std::vector<int>::iterator it = std::find(bundle_ids.begin(), bundle_ids.end(), id);
        if (it != bundle_ids.end())
        {
            descriptionContainer = tag_bundle.member(id);
            return true;
        }
    }

    if (printWarning)
    {
        RCLCPP_WARN_THROTTLE(node_->get_logger(),
                             *node_->get_clock(),
                             10000,
                             "Requested description of standalone tag ID [%d], but no description was found...",
                             id);
    }

    return false;
}

} // namespace apriltag_ros
