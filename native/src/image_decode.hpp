#pragma once
#include <cstddef>
#include <opencv2/core.hpp>
namespace faceid {
cv::Mat decode_image(const unsigned char *bytes, std::size_t size);
int exif_orientation(const unsigned char *bytes, std::size_t size);
}
