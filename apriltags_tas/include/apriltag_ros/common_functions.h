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
 *
 ** common_functions.h *********************************************************
 *
 * Wrapper classes for AprilTag standalone and bundle configs, modified for
 * the usage with apriltags_tas.
 *
 * $Revision: 1.0 $
 * $Date: 2017/12/17 13:23:14 $
 * $Author: dmalyuta $
 *
 * Originator:        Danylo Malyuta, JPL
 ******************************************************************************/

#ifndef APRILTAG_ROS_COMMON_FUNCTIONS_H
#define APRILTAG_ROS_COMMON_FUNCTIONS_H

#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <XmlRpcException.h>
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <image_geometry/pinhole_camera_model.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <ros/console.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/transform_broadcaster.h>

#include <apriltags/TagDetector.h>

namespace apriltag_ros
{

class TagDescription
{
  public:
    TagDescription(){};
    TagDescription(int id, double size, std::string& frame_name) : id(id), size(size), frame_name(frame_name)
    {
    }

    // Tag description
    int id;
    double size;
    std::string frame_name;
    cv::Matx44d T_oi; // If member of a bundle: Rigid transform from tag i frame to bundle origin frame
};

class TagBundleDescription
{
  public:
    std::map<int, int> id2idx_; // (id2idx_[<tag ID>]=<index in tags_>) mapping

    TagBundleDescription(std::string name) : name_(name)
    {
    }

    void addMemberTag(int id, double size, cv::Matx44d T_oi)
    {
        TagDescription member;
        member.id = id;
        member.size = size;
        member.T_oi = T_oi;
        tags_.push_back(member);
        id2idx_[id] = tags_.size() - 1;
    }

    std::string name() const
    {
        return name_;
    }
    // Get IDs of bundle member tags
    std::vector<int> bundleIds()
    {
        std::vector<int> ids;
        for (unsigned int i = 0; i < tags_.size(); i++)
        {
            ids.push_back(tags_[i].id);
        }
        return ids;
    }
    // Get sizes of bundle member tags
    std::vector<double> bundleSizes()
    {
        std::vector<double> sizes;
        for (unsigned int i = 0; i < tags_.size(); i++)
        {
            sizes.push_back(tags_[i].size);
        }
        return sizes;
    }
    int memberID(int tagID)
    {
        return tags_[id2idx_[tagID]].id;
    }
    double memberSize(int tagID)
    {
        return tags_[id2idx_[tagID]].size;
    }
    cv::Matx44d memberT_oi(int tagID)
    {
        return tags_[id2idx_[tagID]].T_oi;
    }
    TagDescription* member(int tagID)
    {
        return &tags_[id2idx_[tagID]];
    }

  private:
    // Bundle description
    std::string name_;
    std::vector<TagDescription> tags_;
};

class TagDetector
{
  private:
    // Other members
    std::map<int, TagDescription> standalone_tag_descriptions_;
    std::vector<TagBundleDescription> tag_bundle_descriptions_;
    tf::TransformBroadcaster tf_pub_;

  public:
    TagDetector(ros::NodeHandle pnh);
    ~TagDetector() = default;

    // Store standalone and bundle tag descriptions
    std::map<int, TagDescription> parseStandaloneTags(XmlRpc::XmlRpcValue& standalone_tag_descriptions);
    std::vector<TagBundleDescription> parseTagBundles(XmlRpc::XmlRpcValue& tag_bundles);
    double xmlRpcGetDoubleWithDefault(XmlRpc::XmlRpcValue& xmlValue, std::string field, double defaultValue) const;

    bool findTagDescription(int id, TagDescription*& descriptionContainer, bool printWarning = true);

    geometry_msgs::PoseStamped makeTagPose(const Eigen::Matrix4d& transform,
                                           const Eigen::Quaternion<double> rot_quaternion,
                                           const std_msgs::Header& header);

    void processBundles(std::vector<AprilTags::TagDetection>& tag_detections,
                        image_geometry::PinholeCameraModel& camera_model,
                        std_msgs::Header header);

    // Get the pose of the tag in the camera frame
    // Returns homogeneous transformation matrix [R,t;[0 0 0 1]] which
    // takes a point expressed in the tag frame to the same point
    // expressed in the camera frame. As usual, R is the (passive)
    // rotation from the tag frame to the camera frame and t is the
    // vector from the camera frame origin to the tag frame origin,
    // expressed in the camera frame.
    Eigen::Matrix4d getRelativeTransform(std::vector<cv::Point3d> objectPoints,
                                         std::vector<cv::Point2d> imagePoints,
                                         double fx,
                                         double fy,
                                         double cx,
                                         double cy) const;

    void addImagePoints(AprilTags::TagDetection& detection, std::vector<cv::Point2d>& imagePoints) const;
    void addObjectPoints(double s, cv::Matx44d T_oi, std::vector<cv::Point3d>& objectPoints) const;
};

} // namespace apriltag_ros

#endif // APRILTAG_ROS_COMMON_FUNCTIONS_H
