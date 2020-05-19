#ifndef TAGDETECTOR_H
#define TAGDETECTOR_H

#include <memory>
#include <vector>

#include "opencv2/opencv.hpp"

#include "apriltags/TagDetection.h"
#include "apriltags/TagFamily.h"

namespace AprilTags
{

class TagDetector
{
  public:
    typedef std::shared_ptr<TagDetector> Ptr;

    const TagFamily thisTagFamily;

    //! Constructor
    // note: TagFamily is instantiated here from TagCodes
    TagDetector(const TagCodes& tagCodes) : thisTagFamily(tagCodes)
    {
    }

    std::vector<TagDetection> extractTags(const cv::Mat& image) const;
};

// 	/*! \brief Class that efficiently performs multi-family tag detections. */
// 	class MultiTagDetector {
// 	public:
//
// 		typedef std::shared_ptr<MultiTagDetector> Ptr;
//
// 		const std::vector<TagFamily> tagFamilies;
//
// 		MultiTagDetector( const std::vector<TagFamily>& families ) : tagFamilies( families ) {}
//
// 		std::vector<TagDetection> extractTags( const cv::Mat& image ) const;
//
// 	};

} // namespace AprilTags

#endif
