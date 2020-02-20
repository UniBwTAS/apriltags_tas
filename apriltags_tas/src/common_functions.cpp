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

#include <apriltag_ros/common_functions.h>

namespace apriltag_ros
{

TagDetector::TagDetector(ros::NodeHandle pnh)
{
    // Parse standalone tag descriptions specified by user (stored on ROS
    // parameter server)
    XmlRpc::XmlRpcValue standalone_tag_descriptions;
    if (!pnh.getParam("standalone_tags", standalone_tag_descriptions))
    {
        ROS_WARN("No april tags specified");
    }
    else
    {
        try
        {
            standalone_tag_descriptions_ = parseStandaloneTags(standalone_tag_descriptions);
        }
        catch (XmlRpc::XmlRpcException e)
        {
            // in case any of the asserts in parseStandaloneTags() fail
            ROS_ERROR_STREAM("Error loading standalone tag descriptions: " << e.getMessage().c_str());
        }
    }

    // parse tag bundle descriptions specified by user (stored on ROS parameter
    // server)
    XmlRpc::XmlRpcValue tag_bundle_descriptions;
    if (!pnh.getParam("tag_bundles", tag_bundle_descriptions))
    {
        ROS_WARN("No tag bundles specified");
    }
    else
    {
        try
        {
            tag_bundle_descriptions_ = parseTagBundles(tag_bundle_descriptions);
        }
        catch (XmlRpc::XmlRpcException e)
        {
            // In case any of the asserts in parseStandaloneTags() fail
            ROS_ERROR_STREAM("Error loading tag bundle descriptions: " << e.getMessage().c_str());
        }
    }
}

void TagDetector::processBundles(std::vector<AprilTags::TagDetection>& tag_detections,
                                 image_geometry::PinholeCameraModel& camera_model,
                                 std_msgs::Header header)
{
    std::map<std::string, std::vector<cv::Point3d>> bundleObjectPoints;
    std::map<std::string, std::vector<cv::Point2d>> bundleImagePoints;

    for (AprilTags::TagDetection& detection : tag_detections)
    {
        int tagID = detection.id;
        for (unsigned int j = 0; j < tag_bundle_descriptions_.size(); j++)
        {
            // Iterate over the registered bundles
            TagBundleDescription bundle = tag_bundle_descriptions_[j];

            if (bundle.id2idx_.find(tagID) != bundle.id2idx_.end())
            {
                // This detected tag belongs to the j-th tag bundle (its ID was found in
                // the bundle description)
                std::string bundleName = bundle.name();

                //===== Corner points in the world frame coordinates
                double s = bundle.memberSize(tagID) / 2;
                addObjectPoints(s, bundle.memberT_oi(tagID), bundleObjectPoints[bundleName]);

                //===== Corner points in the image frame coordinates
                addImagePoints(detection, bundleImagePoints[bundleName]);
            }
        }
    }

    //=================================================================
    // Estimate bundle origin pose for each bundle in which at least one
    // member tag was detected

    for (unsigned int j = 0; j < tag_bundle_descriptions_.size(); j++)
    {
        // Get bundle name
        std::string bundleName = tag_bundle_descriptions_[j].name();

        std::map<std::string, std::vector<cv::Point3d>>::iterator it = bundleObjectPoints.find(bundleName);
        if (it != bundleObjectPoints.end())
        {
            // Some member tags of this bundle were detected, get the bundle's
            // position!
            TagBundleDescription& bundle = tag_bundle_descriptions_[j];

            Eigen::Matrix4d transform = getRelativeTransform(bundleObjectPoints[bundleName],
                                                             bundleImagePoints[bundleName],
                                                             camera_model.fx(),
                                                             camera_model.fy(),
                                                             camera_model.cx(),
                                                             camera_model.cy());
            Eigen::Matrix3d rot = transform.block(0, 0, 3, 3);
            Eigen::Quaternion<double> rot_quaternion(rot);

            geometry_msgs::PoseStamped bundle_pose = makeTagPose(transform, rot_quaternion, header);

            geometry_msgs::PoseStamped pose;
            pose.pose = bundle_pose.pose;
            pose.header = header;
            tf::Stamped<tf::Transform> tag_transform;
            tf::poseStampedMsgToTF(pose, tag_transform);
            tf_pub_.sendTransform(
                tf::StampedTransform(tag_transform, tag_transform.stamp_, header.frame_id, bundle.name()));
        }
    }
    ROS_INFO_STREAM("Found " << bundleObjectPoints.size() << " bundles.");
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

geometry_msgs::PoseStamped TagDetector::makeTagPose(const Eigen::Matrix4d& transform,
                                                    const Eigen::Quaternion<double> rot_quaternion,
                                                    const std_msgs::Header& header)
{
    geometry_msgs::PoseStamped pose;
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
std::map<int, TagDescription> TagDetector::parseStandaloneTags(XmlRpc::XmlRpcValue& standalone_tags)
{
    // Create map that will be filled by the function and returned in the end
    std::map<int, TagDescription> descriptions;
    // Ensure the type is correct
    ROS_ASSERT(standalone_tags.getType() == XmlRpc::XmlRpcValue::TypeArray);
    // Loop through all tag descriptions
    for (int32_t i = 0; i < standalone_tags.size(); i++)
    {

        // i-th tag description
        XmlRpc::XmlRpcValue& tag_description = standalone_tags[i];

        // Assert the tag description is a struct
        ROS_ASSERT(tag_description.getType() == XmlRpc::XmlRpcValue::TypeStruct);
        // Assert type of field "id" is an int
        ROS_ASSERT(tag_description["id"].getType() == XmlRpc::XmlRpcValue::TypeInt);
        // Assert type of field "size" is a double
        ROS_ASSERT(tag_description["size"].getType() == XmlRpc::XmlRpcValue::TypeDouble);

        int id = (int)tag_description["id"]; // tag id
        // Tag size (square, side length in meters)
        double size = (double)tag_description["size"];

        // Custom frame name, if such a field exists for this tag
        std::string frame_name;
        if (tag_description.hasMember("name"))
        {
            // Assert type of field "name" is a string
            ROS_ASSERT(tag_description["name"].getType() == XmlRpc::XmlRpcValue::TypeString);
            frame_name = (std::string)tag_description["name"];
        }
        else
        {
            std::stringstream frame_name_stream;
            frame_name_stream << "tag_" << id;
            frame_name = frame_name_stream.str();
        }

        TagDescription description(id, size, frame_name);
        ROS_INFO_STREAM("Loaded tag config: " << id << ", size: " << size << ", frame_name: " << frame_name.c_str());
        // Add this tag's description to map of descriptions
        descriptions.insert(std::make_pair(id, description));
    }

    return descriptions;
}

// parse tag bundle descriptions
std::vector<TagBundleDescription> TagDetector::parseTagBundles(XmlRpc::XmlRpcValue& tag_bundles)
{
    std::vector<TagBundleDescription> descriptions;
    ROS_ASSERT(tag_bundles.getType() == XmlRpc::XmlRpcValue::TypeArray);

    // Loop through all tag bundle descritions
    for (int32_t i = 0; i < tag_bundles.size(); i++)
    {
        ROS_ASSERT(tag_bundles[i].getType() == XmlRpc::XmlRpcValue::TypeStruct);
        // i-th tag bundle description
        XmlRpc::XmlRpcValue& bundle_description = tag_bundles[i];

        std::string bundleName;
        if (bundle_description.hasMember("name"))
        {
            ROS_ASSERT(bundle_description["name"].getType() == XmlRpc::XmlRpcValue::TypeString);
            bundleName = (std::string)bundle_description["name"];
        }
        else
        {
            std::stringstream bundle_name_stream;
            bundle_name_stream << "bundle_" << i;
            bundleName = bundle_name_stream.str();
        }
        TagBundleDescription bundle_i(bundleName);
        ROS_INFO("Loading tag bundle '%s'", bundle_i.name().c_str());

        ROS_ASSERT(bundle_description["layout"].getType() == XmlRpc::XmlRpcValue::TypeArray);
        XmlRpc::XmlRpcValue& member_tags = bundle_description["layout"];

        // Loop through each member tag of the bundle
        for (int32_t j = 0; j < member_tags.size(); j++)
        {
            ROS_ASSERT(member_tags[j].getType() == XmlRpc::XmlRpcValue::TypeStruct);
            XmlRpc::XmlRpcValue& tag = member_tags[j];

            ROS_ASSERT(tag["id"].getType() == XmlRpc::XmlRpcValue::TypeInt);
            int id = tag["id"];

            ROS_ASSERT(tag["size"].getType() == XmlRpc::XmlRpcValue::TypeDouble);
            double size = tag["size"];

            // Make sure that if this tag was specified also as standalone,
            // then the sizes match
            TagDescription* standaloneDescription;
            if (findTagDescription(id, standaloneDescription, false))
            {
                ROS_ASSERT(size == standaloneDescription->size);
            }

            // Get this tag's pose with respect to the bundle origin
            double x = xmlRpcGetDoubleWithDefault(tag, "x", 0.);
            double y = xmlRpcGetDoubleWithDefault(tag, "y", 0.);
            double z = xmlRpcGetDoubleWithDefault(tag, "z", 0.);
            double qw = xmlRpcGetDoubleWithDefault(tag, "qw", 1.);
            double qx = xmlRpcGetDoubleWithDefault(tag, "qx", 0.);
            double qy = xmlRpcGetDoubleWithDefault(tag, "qy", 0.);
            double qz = xmlRpcGetDoubleWithDefault(tag, "qz", 0.);
            Eigen::Quaterniond q_tag(qw, qx, qy, qz);
            q_tag.normalize();
            Eigen::Matrix3d R_oi = q_tag.toRotationMatrix();

            // Build the rigid transform from tag_j to the bundle origin
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

            // Register the tag member
            bundle_i.addMemberTag(id, size, T_mj);
            ROS_INFO_STREAM(" " << j << ") id: " << id << ", size: " << size << ", "
                                << "p = [" << x << "," << y << "," << z << "], "
                                << "q = [" << qw << "," << qx << "," << qy << "," << qz << "]");
        }
        descriptions.push_back(bundle_i);
    }
    return descriptions;
}

double
TagDetector::xmlRpcGetDoubleWithDefault(XmlRpc::XmlRpcValue& xmlValue, std::string field, double defaultValue) const
{
    if (xmlValue.hasMember(field))
    {
        ROS_ASSERT((xmlValue[field].getType() == XmlRpc::XmlRpcValue::TypeDouble) ||
                   (xmlValue[field].getType() == XmlRpc::XmlRpcValue::TypeInt));
        if (xmlValue[field].getType() == XmlRpc::XmlRpcValue::TypeInt)
        {
            int tmp = xmlValue[field];
            return (double)tmp;
        }
        else
        {
            return xmlValue[field];
        }
    }
    else
    {
        return defaultValue;
    }
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
        ROS_WARN_THROTTLE(10.0,
                          "Requested description of standalone tag ID [%d],"
                          " but no description was found...",
                          id);
    }
    return false;
}

} // namespace apriltag_ros
