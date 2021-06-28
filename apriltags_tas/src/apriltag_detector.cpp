#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_broadcaster.h>

#include <apriltags_msgs/AprilTag.h>
#include <apriltags_msgs/AprilTagDetections.h>

#include <apriltags_tas/apriltag_detector.h>
#include <apriltags_tas/edge_cost_functor.h>
#include <opencv2/viz/types.hpp>

#include <chrono>

AprilTagDetector::AprilTagDetector(const sensor_msgs::CameraInfo::ConstPtr& camera_info,
                                   apriltag_ros::TagDetector& tag_config)
    : tag_config_(tag_config)
{
    if (camera_info)
    {
        camera_model_.fromCameraInfo(camera_info);
    }
}

void AprilTagDetector::reconfigure(apriltags_tas::AprilTagDetectorConfig& config, uint32_t level)
{
    if (config.publish_tf && !camera_model_.initialized())
    {
        config.publish_tf = false;
        ROS_WARN("No camera info available. Publishing tf is disabled.");
    }

    config_ = config;

    if (level & 1)
    {
        AprilTags::TagCodes tag_codes{AprilTags::tagCodes36h11};

        if (config.tag_family == 0)
        {
            tag_codes = AprilTags::TagCodes(AprilTags::tagCodes16h5);
        }
        else if (config.tag_family == 1)
        {
            tag_codes = AprilTags::TagCodes(AprilTags::tagCodes25h7);
        }
        else if (config.tag_family == 2)
        {
            tag_codes = AprilTags::TagCodes(AprilTags::tagCodes25h9);
        }
        else if (config.tag_family == 3)
        {
            tag_codes = AprilTags::TagCodes(AprilTags::tagCodes36h9);
        }
        else if (config.tag_family == 4)
        {
            tag_codes = AprilTags::TagCodes(AprilTags::tagCodes36h11);
        }

        apriltag_cpp_detector_ = std::make_shared<AprilTags::TagDetector>(tag_codes);
    }

    if (!image_.empty())
    {
        process(image_);
    }
}

void AprilTagDetector::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    image_ = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
    img_header_ = msg->header;

    process(image_);
}

void AprilTagDetector::process(const cv::Mat& image)
{
    if (detections_pub_.getNumSubscribers() == 0 && image_pub_.getNumSubscribers() == 0)
    {
        ROS_WARN_STREAM("No subscribers => Do not detect tags!");

        return;
    }

    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    std::vector<AprilTags::TagDetection> tag_detections = detectAprilTags(gray_image);

    if (config_.only_known_tags)
    {
        filterUnknownTags(tag_detections);
    }

    if (config_.refine_corners)
    {
        refineCornerPointsByDirectEdgeOptimization(gray_image, tag_detections);
    }

    if (config_.filter_cross_corners)
    {
        filterCrossCorners(gray_image, tag_detections);
    }

    publishTagDetections(tag_detections, img_header_);

    if (config_.publish_tf)
    {
        publishTfTransform(tag_detections, img_header_);
        tag_config_.processBundles(tag_detections, camera_model_, img_header_);
    }

    if (config_.draw_image)
    {
        drawTagDetections(image.clone(), tag_detections);
    }
}

std::vector<AprilTags::TagDetection> AprilTagDetector::detectAprilTags(cv::Mat& img) noexcept
{
    auto t_last = std::chrono::high_resolution_clock::now();

    std::vector<AprilTags::TagDetection> tag_detections = apriltag_cpp_detector_->extractTags(img);

    const int t_total =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t_last)
            .count();
    ROS_INFO_STREAM("Detected " << tag_detections.size() << " tags in " << t_total << " ms.");
    return tag_detections;
}

void AprilTagDetector::refineCornerPointsByDirectEdgeOptimization(
    cv::Mat& img, std::vector<AprilTags::TagDetection>& tag_detections) noexcept
{
    for (AprilTags::TagDetection& tag : tag_detections)
    {
        std::pair<float, float> x_range(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest());
        std::pair<float, float> y_range(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest());
        for (std::pair<float, float>& corner_point : tag.p)
        {
            x_range.first = std::min(x_range.first, corner_point.first);
            x_range.second = std::max(x_range.second, corner_point.first);
            y_range.first = std::min(y_range.first, corner_point.second);
            y_range.second = std::max(y_range.second, corner_point.second);
        }

        x_range.first = std::max(x_range.first - 10, static_cast<float>(0));
        x_range.second = std::min(x_range.second + 10, static_cast<float>(img.cols - 1));
        y_range.first = std::max(y_range.first - 10, static_cast<float>(0));
        y_range.second = std::min(y_range.second + 10, static_cast<float>(img.rows - 1));

        try
        {
            cv::Rect roi(cv::Point(x_range.first, y_range.first), cv::Point(x_range.second, y_range.second));

            cv::Mat cropped_img = img(roi);

            std::array<Eigen::Vector2d, 4> estimated_edge_normals;
            std::array<double, 4> estimated_edge_offsets;
            std::array<cv::Mat, 4> mask_images;

            int line_thickness = 5;

            auto nextCornerIndex = [](const int i) {
                if (i <= 2)
                {
                    return i + 1;
                }
                else
                {
                    return 0;
                }
            };

            for (int i = 0; i < 4; i++)
            {
                const int next_corner_index = nextCornerIndex(i);

                const Eigen::Vector2d roi_offset_vector(roi.x, roi.y);

                const Eigen::Vector2d corner(tag.p[i].first, tag.p[i].second);
                const Eigen::Vector2d next_corner(tag.p[next_corner_index].first, tag.p[next_corner_index].second);

                const Eigen::Hyperplane<double, 2> edge_line =
                    Eigen::Hyperplane<double, 2>::Through(corner - roi_offset_vector, next_corner - roi_offset_vector);

                estimated_edge_normals[i] = edge_line.normal();
                estimated_edge_offsets[i] = edge_line.offset();

                mask_images[i].create(cropped_img.rows, cropped_img.cols, CV_8U);
                mask_images[i].setTo(0);

                const cv::Point2i current_corner_point(std::round(tag.p[i].first - roi.x),
                                                       std::round(tag.p[i].second - roi.y));

                const cv::Point2i next_corner_point(std::round(tag.p[next_corner_index].first - roi.x),
                                                    std::round(tag.p[next_corner_index].second - roi.y));

                cv::line(mask_images[i], current_corner_point, next_corner_point, cv::Scalar(255), line_thickness);

                cv::rectangle(mask_images[i], current_corner_point, current_corner_point, cv::Scalar(0), 10);
                cv::rectangle(mask_images[i], next_corner_point, next_corner_point, cv::Scalar(0), 10);
            }

            ceres::Problem optimization_problem;

            // Set up the only cost function (also known as residual). This uses
            // auto-differentiation to obtain the derivative (jacobian).

            ceres::NumericDiffOptions numeric_diff_options;

            auto addEdgeResidualBlocks = [&optimization_problem,
                                          &mask_images,
                                          &cropped_img,
                                          &estimated_edge_normals,
                                          &estimated_edge_offsets,
                                          &numeric_diff_options,
                                          &nextCornerIndex](const int i) {
                const int pixel_count = cv::countNonZero(mask_images[i]);

                ceres::CostFunction* cost_function =
                    new ceres::NumericDiffCostFunction<EdgeCostFunctor, ceres::CENTRAL, ceres::DYNAMIC, 2, 1>(
                        new EdgeCostFunctor(cropped_img, mask_images[i]),
                        ceres::TAKE_OWNERSHIP,
                        pixel_count,
                        numeric_diff_options);
                optimization_problem.AddResidualBlock(
                    cost_function, nullptr, estimated_edge_normals[i].data(), &estimated_edge_offsets[i]);

                optimization_problem.SetParameterization(estimated_edge_normals[i].data(),
                                                         new ceres::HomogeneousVectorParameterization(2));
            };

            addEdgeResidualBlocks(0);
            addEdgeResidualBlocks(1);
            addEdgeResidualBlocks(2);
            addEdgeResidualBlocks(3);

            ceres::Solver::Options solve_options;
            solve_options.linear_solver_type = ceres::DENSE_QR;
            solve_options.max_num_iterations = 100;

            ceres::Solver::Summary summary;
            ceres::Solve(solve_options, &optimization_problem, &summary);

            for (int edge_index = 0; edge_index < 4; edge_index++)
            {
                const int next_edge_index = nextCornerIndex(edge_index);
                const int corner_index = next_edge_index;

                const Eigen::Hyperplane<double, 2> edge_A(estimated_edge_normals[edge_index],
                                                          estimated_edge_offsets[edge_index]);
                const Eigen::Hyperplane<double, 2> edge_B(estimated_edge_normals[next_edge_index],
                                                          estimated_edge_offsets[next_edge_index]);

                const Eigen::Vector2d estimated_corner_pos_roi = edge_A.intersection(edge_B);

                tag.p[corner_index].first = estimated_corner_pos_roi.x() + roi.x;
                tag.p[corner_index].second = estimated_corner_pos_roi.y() + roi.y;
            }
        }
        catch (const std::exception& /*e*/)
        {
            tag.good = false;
        }
    }
    removeBadTags(tag_detections);

    ROS_INFO_STREAM("Refined " << tag_detections.size() << " tags.");
}

void AprilTagDetector::filterCrossCorners(cv::Mat& img, std::vector<AprilTags::TagDetection>& tag_detections) noexcept
{
    cv::Mat img_binary;
    cv::adaptiveThreshold(img, img_binary, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 21, 2);

    const cv::Mat morph_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    cv::morphologyEx(img_binary, img_binary, cv::MORPH_OPEN, morph_kernel);
    cv::morphologyEx(img_binary, img_binary, cv::MORPH_CLOSE, morph_kernel);

    int invalid_tags = 0;

    for (AprilTags::TagDetection& tag : tag_detections)
    {
        for (int i = 0; i < 4; i++)
        {
            const cv::Point2f corner(tag.p[i].first, tag.p[i].second);

            std::vector<std::pair<float, float>> sections;
            std::vector<int> section_colors;

            float r = 5;
            float last_phi = 0;
            for (float phi = 0; phi < 2 * M_PI; phi += 10.0 * M_PI / 180.0)
            {
                const float s = std::sin(phi);
                const float c = std::cos(phi);

                cv::Point2f p_rel(c * r, s * r);
                cv::Point2f p = corner + p_rel;

                cv::Mat patch;
                cv::getRectSubPix(img_binary, cv::Size(1, 1), p, patch);

                const bool binary_color = *patch.data < 128;

                // Init first section
                if (sections.empty())
                {
                    sections.emplace_back(0, phi);
                    section_colors.emplace_back(binary_color);
                }

                // Continue last section
                if (binary_color == section_colors.back())
                {
                    sections.back().second = phi;
                }
                // Start new section
                else
                {
                    sections.emplace_back(last_phi, phi);
                    section_colors.emplace_back(binary_color);
                }

                last_phi = phi;
            }

            // Merge first and last section, if they are identical
            if (section_colors.front() == section_colors.back())
            {
                sections.front().first = sections.back().first - 2 * M_PI;
                sections.pop_back();
                section_colors.pop_back();
            }

            bool corner_valid = true;
            if (sections.size() != 4)
            {
                corner_valid = false;
            }
            // Correct number of sections => Check angles for similarity
            else
            {
                const float angle_diff_1 = std::abs(std::abs(sections[0].second - sections[0].first) -
                                                    std::abs(sections[2].second - sections[2].first));
                const float angle_diff_2 = std::abs(std::abs(sections[1].second - sections[1].first) -
                                                    std::abs(sections[3].second - sections[3].first));

                if (std::max(angle_diff_1, angle_diff_2) > 30.0 * M_PI / 180.0)
                {
                    corner_valid = false;
                }
            }

            if (!corner_valid)
            {
                tag.good = false;
                tag.p[i].first = NAN;
                tag.p[i].second = NAN;
            }
        }
        if (!tag.good)
        {
            invalid_tags++;
        }
    }

    removeBadTags(tag_detections);

    ROS_INFO_STREAM("Filtered " << invalid_tags << " tags out, returning " << tag_detections.size() << " tags.");
}

void AprilTagDetector::filterUnknownTags(std::vector<AprilTags::TagDetection>& tag_detections) noexcept
{
    for (AprilTags::TagDetection& tag : tag_detections)
    {
        apriltag_ros::TagDescription* tag_description;
        if (tag.good && !tag_config_.findTagDescription(tag.id, tag_description, false))
        {
            tag.good = false;
        }
    }
    removeBadTags(tag_detections);

    ROS_INFO_STREAM("Found " << tag_detections.size() << " known tags.");
}

void AprilTagDetector::removeBadTags(std::vector<AprilTags::TagDetection>& tag_detections) noexcept
{
    tag_detections.erase(remove_if(begin(tag_detections),
                                   end(tag_detections),
                                   [](AprilTags::TagDetection const& tag) { return tag.good == false; }),
                         end(tag_detections));
}

bool AprilTagDetector::getPose(AprilTags::TagDetection& tag, geometry_msgs::Pose& pose) noexcept
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

void AprilTagDetector::publishTagDetections(std::vector<AprilTags::TagDetection>& tag_detections,
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
    detections_pub_.publish(detections_msg);
}

void AprilTagDetector::publishTfTransform(std::vector<AprilTags::TagDetection>& tag_detections,
                                          std_msgs::Header header) noexcept
{
    static tf2_ros::TransformBroadcaster br;

    for (AprilTags::TagDetection& tag : tag_detections)
    {
        apriltag_ros::TagDescription* tag_description;
        if (tag_config_.findTagDescription(tag.id, tag_description) && tag_description->frame_name != "")
        {
            geometry_msgs::Pose pose;
            bool pose_valid = getPose(tag, pose);

            if (pose_valid)
            {
                static tf2_ros::TransformBroadcaster br;
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

                br.sendTransform(transformStamped);
            }
        }
    }
}

void AprilTagDetector::drawTagDetections(cv::Mat img, std::vector<AprilTags::TagDetection>& tag_detections) noexcept
{
    int line_thickness = img.size[0] / 400;

    for (AprilTags::TagDetection& tag : tag_detections)
    {
        int tag_size_px =
            std::max(std::abs(tag.p[0].first - tag.p[1].first), std::abs(tag.p[1].first - tag.p[2].first));
        double fontscale = tag_size_px / 70.0;
        double text_thickness = fontscale * 3;

        // plot outline
        cv::line(img,
                 cv::Point2f(tag.p[0].first, tag.p[0].second),
                 cv::Point2f(tag.p[1].first, tag.p[1].second),
                 cv::viz::Color::red(),
                 line_thickness);
        cv::line(img,
                 cv::Point2f(tag.p[1].first, tag.p[1].second),
                 cv::Point2f(tag.p[2].first, tag.p[2].second),
                 cv::viz::Color::green(),
                 line_thickness);
        cv::line(img,
                 cv::Point2f(tag.p[2].first, tag.p[2].second),
                 cv::Point2f(tag.p[3].first, tag.p[3].second),
                 cv::viz::Color::blue(),
                 line_thickness);
        cv::line(img,
                 cv::Point2f(tag.p[3].first, tag.p[3].second),
                 cv::Point2f(tag.p[0].first, tag.p[0].second),
                 cv::viz::Color::magenta(),
                 line_thickness);

        // print tag id
        cv::String text = std::to_string(tag.id);
        int fontface = cv::FONT_HERSHEY_SIMPLEX;
        int baseline;
        cv::Size textsize = cv::getTextSize(text, fontface, fontscale, text_thickness, &baseline);
        cv::putText(img,
                    text,
                    cv::Point((int)(tag.cxy.first - textsize.width / 2), (int)(tag.cxy.second + textsize.height / 2)),
                    fontface,
                    fontscale,
                    cv::viz::Color::azure(),
                    text_thickness);
    }
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    image_pub_.publish(msg);
}