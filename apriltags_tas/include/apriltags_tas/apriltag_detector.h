#pragma once

#include <apriltags/TagDetector.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

using Logger = std::function<void(const std::string&)>;
enum class TagFamily
{
    tag16h5 = 0,
    tag25h7 = 1,
    tag25h9 = 2,
    tag36h9 = 3,
    tag36h11 = 4
};

enum class RefinementMethod
{
    NoRefinement = 0,
    CornerRefinement = 1,
    AdvEdgeRefinement = 2
};

struct Config
{
    TagFamily tag_family{TagFamily::tag36h11};
    RefinementMethod refinement_method{RefinementMethod::NoRefinement};
    bool only_known_tags{false};
    bool filter_cross_corners{true};
    double filter_cross_corners_radius_percent{5.0};
    bool publish_tf{true};
    bool draw_image{true};
};

class AprilTagDetector
{
  public:
    AprilTagDetector(Logger info_logger, Logger warn_logger);

    void reconfigure(const Config& config, uint32_t level);

    std::vector<AprilTags::TagDetection> process(const cv::Mat& image_bgr,
                                                 const std::function<bool(int)>& is_known_tag) noexcept;

    void drawTagDetections(cv::Mat& img, const std::vector<AprilTags::TagDetection>& tag_detections) const noexcept;

  private:
    std::vector<AprilTags::TagDetection> detectAprilTags(cv::Mat& img) noexcept;
    void refineCornerPointsByDirectEdgeOptimization(cv::Mat& img,
                                                    std::vector<AprilTags::TagDetection>& tag_detections) noexcept;
    void refineCornerPointsByOpenCVCornerRefinement(cv::Mat& img,
                                                    std::vector<AprilTags::TagDetection>& tag_detections) noexcept;
    void filterCrossCorners(cv::Mat& img, std::vector<AprilTags::TagDetection>& tag_detections) noexcept;
    void filterUnknownTags(std::vector<AprilTags::TagDetection>& tag_detections,
                           const std::function<bool(int)>& is_known_tag) noexcept;

    static void removeBadTags(std::vector<AprilTags::TagDetection>& tag_detections) noexcept;

    Logger info_logger_;
    Logger warn_logger_;

    Config config_;
    std::shared_ptr<AprilTags::TagDetector> apriltag_cpp_detector_;
};
