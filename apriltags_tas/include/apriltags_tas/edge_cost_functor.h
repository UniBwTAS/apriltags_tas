#pragma once

#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

class EdgeCostFunctor
{
  public:
    EdgeCostFunctor(const cv::Mat& rawImg, const cv::Mat& maskImg)
        : raw_img(rawImg), mask_img(maskImg)
    {
        template_img.create(1, 2, CV_8U);
        template_img.data[0] = 0;
        template_img.data[1] = 255;
    }

    template<typename T>
    T getSubPixel(const cv::Mat& img, cv::Point_<T> pt) const
    {
        const T x = floor(pt.x);
        const T y = floor(pt.y);

        const int x0 = cv::borderInterpolate(x, img.cols, cv::BORDER_REPLICATE);
        const int x1 = cv::borderInterpolate(x + 1, img.cols, cv::BORDER_REPLICATE);
        const int y0 = cv::borderInterpolate(y, img.rows, cv::BORDER_REPLICATE);
        const int y1 = cv::borderInterpolate(y + 1, img.rows, cv::BORDER_REPLICATE);

        const T a = pt.x - T(x);
        const T c = pt.y - T(y);

        return (T(img.at<uint8_t>(y0, x0)) * (T(1.0) - a) + T(img.at<uint8_t>(y0, x1)) * a) * (T(1.0) - c) +
               (T(img.at<uint8_t>(y1, x0)) * (T(1.0) - a) + T(img.at<uint8_t>(y1, x1)) * a) * c;
    }

    template<typename T>
    bool operator()(const T* const edge_normal_ptr, const T* const edge_offset_ptr, T* residual) const
    {
        const Eigen::Map<const Eigen::Matrix<T, 2, 1>> edge_normal(edge_normal_ptr);
        const double& edge_offset = *edge_offset_ptr;

        const Eigen::Hyperplane<T, 2> line(edge_normal, edge_offset);

        int residual_index = 0;
        for (int row = 0; row < raw_img.rows; row++)
        {
            const auto raw_img_row_ptr = raw_img.ptr<uint8_t>(row);
            const auto mask_row_ptr = mask_img.ptr<uint8_t>(row);

            for (int col = 0; col < raw_img.cols; col++)
            {
                const auto eval = mask_row_ptr[col];

                if (eval)
                {
                    const Eigen::Vector2d raw_img_pos(col, row);
                    const double dist_to_line = line.signedDistance(raw_img_pos);

                    const ::cv::Point2d template_pt(dist_to_line + 0.5, 0);

                    const double pred_pixel_value = getSubPixel<double>(template_img, template_pt);
                    const uint8_t current_pixel_values = raw_img_row_ptr[col];
                    residual[residual_index] = pred_pixel_value - current_pixel_values;
                    residual_index++;
                }
            }
        }

        return true;
    }

  private:
    cv::Mat raw_img;
    cv::Mat mask_img;
    cv::Mat template_img;
};