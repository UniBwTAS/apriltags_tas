#ifndef TAG_DETECTION_UTILS_H

#include "apriltags/TagDetection.h"
#include "apriltags/TagFamily.h"
#include "apriltags/Quad.h"
#include "apriltags/UnionFindSimple.h"
#include "apriltags/Segment.h"

namespace AprilTags
{

// 	template<class D>
// 	void convert_mat( const cv::Mat& src, FloatImage& dst, float scale=1.0 )
// 	{
// 		int width = src.cols;
// 		int height = src.rows;
// 		dst = FloatImage( width, height );
// 		int i = 0;
// 		for( int y = 0; y < height; y++ )
// 		{
// 			for( int x = 0; x < width; x++ )
// 			{
// 				dst.set( x, y, src.at<D>(y,x)*scale );
// 				i++;
// 			}
// 		}
// 	}

	/*! \brief Steps 1-2. Calculate image gradients */
	void preprocess_image( const cv::Mat& orig, cv::Mat& fim,
						   cv::Mat& fimSeg, cv::Mat& fimTheta, cv::Mat& fimMag,
						   float sigma = 0.0, float segSigma = 0.8 );

	/*! \brief Step 3. Find edges */
	void extract_edges( const cv::Mat& fimSeg, const cv::Mat& fimMag,
						   const cv::Mat& fimTheta, UnionFindSimple& uf );

	/*! \brief Step 4-5. Fit line segments */
	void fit_segments( const cv::Mat& fimSeg, const cv::Mat& fimMag,
					   const cv::Mat& fimTheta, UnionFindSimple& uf,
					   std::vector<Segment>& segments );

	/*! \brief Step 6-7. Find quads */
	void find_quads( std::vector<Segment>& segments,
					 const cv::Size imSize,
					 const std::pair<int,int>& opticalCenter,
					 std::vector<Quad>& quads );

	/*! \brief Step 8-9. Decode quads from a float image for a given family. */
	std::vector<TagDetection> decode_quads( const cv::Mat& fim,
											const std::vector<Quad>& quads,
											const TagFamily& family );

}

#endif
