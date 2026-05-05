#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <apriltags/TagFamily.h>
#include <apriltags/Tag16h5.h>
#include <apriltags/Tag25h7.h>
#include <apriltags/Tag25h9.h>
#include <apriltags/Tag36h11.h>
#include <apriltags/Tag36h9.h>

#include <apriltags_tas/apriltag_detector.h>
#include <apriltags_tas/edge_cost_functor.h>

AprilTagDetector::AprilTagDetector(Logger info_logger, Logger warn_logger)
    : info_logger_(info_logger), warn_logger_(warn_logger)
{
}

void AprilTagDetector::reconfigure(const Config& config, uint32_t level)
{
    config_ = config;

    if (level & 1)
    {
        AprilTags::TagCodes tag_codes{AprilTags::tagCodes36h11};

        switch (config.tag_family)
        {
            case TagFamily::tag16h5:
                tag_codes = AprilTags::TagCodes(AprilTags::tagCodes16h5);
                break;
            case TagFamily::tag25h7:
                tag_codes = AprilTags::TagCodes(AprilTags::tagCodes25h7);
                break;
            case TagFamily::tag25h9:
                tag_codes = AprilTags::TagCodes(AprilTags::tagCodes25h9);
                break;
            case TagFamily::tag36h9:
                tag_codes = AprilTags::TagCodes(AprilTags::tagCodes36h9);
                break;
            case TagFamily::tag36h11:
                tag_codes = AprilTags::TagCodes(AprilTags::tagCodes36h11);
                break;
        }

        apriltag_cpp_detector_ = std::make_shared<AprilTags::TagDetector>(tag_codes);
    }
}

std::vector<AprilTags::TagDetection> AprilTagDetector::process(const cv::Mat& image_bgr,
                                                               const std::function<bool(int)>& is_known_tag) noexcept
{
    cv::Mat gray_image;
    cv::cvtColor(image_bgr, gray_image, cv::COLOR_BGR2GRAY);

    std::vector<AprilTags::TagDetection> tag_detections = detectAprilTags(gray_image);

    if (config_.only_known_tags)
    {
        filterUnknownTags(tag_detections, is_known_tag);
    }

    if (config_.refinement_method == RefinementMethod::AdvEdgeRefinement)
    {
        refineCornerPointsByDirectEdgeOptimization(gray_image, tag_detections);
    }
    else if (config_.refinement_method == RefinementMethod::CornerRefinement)
    {
        refineCornerPointsByOpenCVCornerRefinement(gray_image, tag_detections);
    }

    if (config_.filter_cross_corners)
    {
        filterCrossCorners(gray_image, tag_detections);
    }

    return tag_detections;
}

std::vector<AprilTags::TagDetection> AprilTagDetector::detectAprilTags(cv::Mat& img) noexcept
{
    if (!apriltag_cpp_detector_)
    {
        return {};
    }

    const auto t_last = std::chrono::high_resolution_clock::now();

    std::vector<AprilTags::TagDetection> tag_detections = apriltag_cpp_detector_->extractTags(img);

    const int t_total =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t_last)
            .count();

    if (info_logger_)
    {
        info_logger_("Detected " + std::to_string(tag_detections.size()) + " tags in " + std::to_string(t_total) +
                     " ms.");
    }
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

            const int line_thickness = 5;

            auto nextCornerIndex = [](const int i) { return (i + 1) % 4; };

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
            ceres::NumericDiffOptions numeric_diff_options;

            auto addEdgeResidualBlocks = [&optimization_problem,
                                          &mask_images,
                                          &cropped_img,
                                          &estimated_edge_normals,
                                          &estimated_edge_offsets,
                                          &numeric_diff_options](const int i)
            {
                const int pixel_count = cv::countNonZero(mask_images[i]);

                ceres::CostFunction* cost_function =
                    new ceres::NumericDiffCostFunction<EdgeCostFunctor, ceres::CENTRAL, ceres::DYNAMIC, 2, 1>(
                        new EdgeCostFunctor(cropped_img, mask_images[i]),
                        ceres::TAKE_OWNERSHIP,
                        pixel_count,
                        numeric_diff_options);
                optimization_problem.AddResidualBlock(
                    cost_function, nullptr, estimated_edge_normals[i].data(), &estimated_edge_offsets[i]);

#if CERES_VERSION_MAJOR >= 2
                optimization_problem.SetManifold(estimated_edge_normals[i].data(), new ceres::SphereManifold<2>());
#else
                optimization_problem.SetParameterization(estimated_edge_normals[i].data(),
                                                         new ceres::HomogeneousVectorParameterization(2));
#endif
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
        catch (const std::exception&)
        {
            tag.good = false;
        }
    }

    removeBadTags(tag_detections);

    if (info_logger_)
    {
        info_logger_("Refined " + std::to_string(tag_detections.size()) + " tags.");
    }
}

void AprilTagDetector::refineCornerPointsByOpenCVCornerRefinement(
    cv::Mat& img, std::vector<AprilTags::TagDetection>& tag_detections) noexcept
{
    for (AprilTags::TagDetection& tag : tag_detections)
    {
        std::vector<cv::Point2f> corners;

        for (int i = 0; i < 4; i++)
        {
            corners.emplace_back(tag.p[i].first, tag.p[i].second);
        }

        const cv::Size win_size(10, 10);
        const cv::Size zero_zone(-1, -1);
        const cv::TermCriteria term_criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.0001);

        cv::cornerSubPix(img, corners, win_size, zero_zone, term_criteria);

        for (int i = 0; i < 4; i++)
        {
            tag.p[i].first = corners[i].x;
            tag.p[i].second = corners[i].y;
        }
    }
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
        auto cornerPos = [&tag](const int i) { return cv::Point2f(tag.p[i].first, tag.p[i].second); };

        const float l1 = cv::norm(cornerPos(0) - cornerPos(1));
        const float l2 = cv::norm(cornerPos(1) - cornerPos(2));
        const float l3 = cv::norm(cornerPos(2) - cornerPos(3));
        const float l4 = cv::norm(cornerPos(3) - cornerPos(0));

        const float mean_tag_size = (l1 + l2 + l3 + l4) / 4.0f;

        for (int i = 0; i < 4; i++)
        {
            const cv::Point2f corner = cornerPos(i);

            std::vector<std::pair<float, float>> sections;
            std::vector<int> section_colors;

            const float r = mean_tag_size * (config_.filter_cross_corners_radius_percent / 100.0f);
            float last_phi = 0;

            for (float phi = 0; phi < 2 * M_PI; phi += 10.0f * M_PI / 180.0f)
            {
                const float s = std::sin(phi);
                const float c = std::cos(phi);

                const cv::Point2f p = corner + cv::Point2f(c * r, s * r);

                cv::Mat patch;
                cv::getRectSubPix(img_binary, cv::Size(1, 1), p, patch);

                const bool binary_color = *patch.data < 128;

                if (sections.empty())
                {
                    sections.emplace_back(0, phi);
                    section_colors.emplace_back(binary_color);
                }

                if (binary_color == section_colors.back())
                {
                    sections.back().second = phi;
                }
                else
                {
                    sections.emplace_back(last_phi, phi);
                    section_colors.emplace_back(binary_color);
                }

                last_phi = phi;
            }

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
            else
            {
                const float angle_diff_1 = std::abs(std::abs(sections[0].second - sections[0].first) -
                                                    std::abs(sections[2].second - sections[2].first));
                const float angle_diff_2 = std::abs(std::abs(sections[1].second - sections[1].first) -
                                                    std::abs(sections[3].second - sections[3].first));

                if (std::max(angle_diff_1, angle_diff_2) > 30.0f * M_PI / 180.0f)
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

    if (info_logger_)
    {
        info_logger_("Filtered " + std::to_string(invalid_tags) + " tags out, returning " +
                     std::to_string(tag_detections.size()) + " tags.");
    }
}

void AprilTagDetector::filterUnknownTags(std::vector<AprilTags::TagDetection>& tag_detections,
                                         const std::function<bool(int)>& is_known_tag) noexcept
{
    for (AprilTags::TagDetection& tag : tag_detections)
    {
        if (tag.good && !is_known_tag(tag.id))
        {
            tag.good = false;
        }
    }

    removeBadTags(tag_detections);

    if (info_logger_)
    {
        info_logger_("Found " + std::to_string(tag_detections.size()) + " known tags.");
    }
}

void AprilTagDetector::removeBadTags(std::vector<AprilTags::TagDetection>& tag_detections) noexcept
{
    tag_detections.erase(std::remove_if(begin(tag_detections),
                                        end(tag_detections),
                                        [](const AprilTags::TagDetection& tag) { return !tag.good; }),
                         end(tag_detections));
}

void AprilTagDetector::drawTagDetections(cv::Mat& img,
                                         const std::vector<AprilTags::TagDetection>& tag_detections) const noexcept
{
    const int line_thickness = img.size[0] / 400;

    for (const AprilTags::TagDetection& tag : tag_detections)
    {
        const int tag_size_px =
            std::max(std::abs(tag.p[0].first - tag.p[1].first), std::abs(tag.p[1].first - tag.p[2].first));
        const double fontscale = tag_size_px / 70.0;
        const double text_thickness = fontscale * 3;

        cv::line(img,
                 cv::Point2f(tag.p[0].first, tag.p[0].second),
                 cv::Point2f(tag.p[1].first, tag.p[1].second),
                 cv::Scalar(0, 0, 255),
                 line_thickness);
        cv::line(img,
                 cv::Point2f(tag.p[1].first, tag.p[1].second),
                 cv::Point2f(tag.p[2].first, tag.p[2].second),
                 cv::Scalar(0, 255, 0),
                 line_thickness);
        cv::line(img,
                 cv::Point2f(tag.p[2].first, tag.p[2].second),
                 cv::Point2f(tag.p[3].first, tag.p[3].second),
                 cv::Scalar(255, 0, 0),
                 line_thickness);
        cv::line(img,
                 cv::Point2f(tag.p[3].first, tag.p[3].second),
                 cv::Point2f(tag.p[0].first, tag.p[0].second),
                 cv::Scalar(255, 0, 255),
                 line_thickness);

        const cv::String text = std::to_string(tag.id);
        const int fontface = cv::FONT_HERSHEY_SIMPLEX;
        int baseline;
        const cv::Size textsize = cv::getTextSize(text, fontface, fontscale, text_thickness, &baseline);
        cv::putText(img,
                    text,
                    cv::Point(static_cast<int>(tag.cxy.first - textsize.width / 2),
                              static_cast<int>(tag.cxy.second + textsize.height / 2)),
                    fontface,
                    fontscale,
                    cv::Scalar(255, 255, 0),
                    text_thickness);
    }
}
