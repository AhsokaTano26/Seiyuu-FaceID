#include "image_decode.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace faceid {
namespace {
std::uint32_t be32(const unsigned char *p) {
    return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16) |
           (std::uint32_t(p[2]) << 8) | p[3];
}
std::uint32_t le32(const unsigned char *p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
           (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}
int tiff_orientation(const unsigned char *p, std::size_t n) {
    if (n >= 6 && std::memcmp(p, "Exif\0\0", 6) == 0) { p += 6; n -= 6; }
    if (n < 8) return 1;
    const bool little = p[0] == 'I' && p[1] == 'I';
    if (!little && !(p[0] == 'M' && p[1] == 'M')) return 1;
    auto u16 = [little](const unsigned char *v) -> unsigned {
        return little ? unsigned(v[0]) | (unsigned(v[1]) << 8) : (unsigned(v[0]) << 8) | v[1];
    };
    auto u32 = [little](const unsigned char *v) { return little ? le32(v) : be32(v); };
    if (u16(p + 2) != 42) return 1;
    const std::size_t offset = u32(p + 4);
    if (offset > n - 2) return 1;
    const std::size_t count = u16(p + offset);
    if (count > (n - offset - 2) / 12) return 1;
    for (std::size_t i = 0; i < count; ++i) {
        const auto *entry = p + offset + 2 + i * 12;
        if (u16(entry) == 0x112 && u16(entry + 2) == 3 && u32(entry + 4) == 1) {
            const int value = static_cast<int>(u16(entry + 8));
            return value >= 1 && value <= 8 ? value : 1;
        }
    }
    return 1;
}
}

int exif_orientation(const unsigned char *bytes, std::size_t size) {
    if (!bytes) return 1;
    if (size >= 2 && bytes[0] == 0xff && bytes[1] == 0xd8) {
        std::size_t pos = 2;
        while (pos + 4 <= size) {
            if (bytes[pos] != 0xff) break;
            while (pos < size && bytes[pos] == 0xff) ++pos;
            if (pos >= size) break;
            const unsigned marker = bytes[pos++];
            if (marker == 0xda || marker == 0xd9) break;
            if (marker == 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
            if (pos + 2 > size) break;
            const std::size_t length = (std::size_t(bytes[pos]) << 8) | bytes[pos + 1];
            if (length < 2 || length > size - pos) break;
            if (marker == 0xe1 && length >= 8 && std::memcmp(bytes + pos + 2, "Exif\0\0", 6) == 0)
                return tiff_orientation(bytes + pos + 2, length - 2);
            pos += length;
        }
    } else if (size >= 8 && std::memcmp(bytes, "\x89PNG\r\n\x1a\n", 8) == 0) {
        std::size_t pos = 8;
        while (pos + 12 <= size) {
            const std::size_t length = be32(bytes + pos);
            if (length > size - pos - 12) break;
            if (std::memcmp(bytes + pos + 4, "eXIf", 4) == 0)
                return tiff_orientation(bytes + pos + 8, length);
            pos += length + 12;
        }
    } else if (size >= 12 && std::memcmp(bytes, "RIFF", 4) == 0 && std::memcmp(bytes + 8, "WEBP", 4) == 0) {
        std::size_t pos = 12;
        while (pos + 8 <= size) {
            const std::size_t length = le32(bytes + pos + 4);
            if (length > size - pos - 8) break;
            if (std::memcmp(bytes + pos, "EXIF", 4) == 0)
                return tiff_orientation(bytes + pos + 8, length);
            pos += length + 8 + (length & 1);
        }
    }
    return 1;
}

cv::Mat decode_image(const unsigned char *bytes, std::size_t size) {
    cv::Mat encoded(1, static_cast<int>(size), CV_8UC1, const_cast<unsigned char *>(bytes));
    cv::Mat image;
    const bool gray16_png = size >= 26 && std::memcmp(bytes, "\x89PNG\r\n\x1a\n", 8) == 0 && bytes[24] == 16 && bytes[25] == 0;
    if (gray16_png) {
        // Pillow opens this as I;16 and saturates when converting to RGB.
        const auto gray16 = cv::imdecode(encoded, cv::IMREAD_UNCHANGED);
        if (gray16.empty() || gray16.type() != CV_16UC1) throw std::invalid_argument("invalid_image");
        cv::Mat gray8;
        gray16.convertTo(gray8, CV_8U);
        cv::cvtColor(gray8, image, cv::COLOR_GRAY2BGR);
    } else image = cv::imdecode(encoded, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (image.empty()) throw std::invalid_argument("invalid_image");
    if (image.total() > 20000000) throw std::invalid_argument("image_too_large");
    cv::Mat transformed;
    switch (exif_orientation(bytes, size)) {
        case 2: cv::flip(image, transformed, 1); break;
        case 3: cv::rotate(image, transformed, cv::ROTATE_180); break;
        case 4: cv::flip(image, transformed, 0); break;
        case 5: cv::transpose(image, transformed); break;
        case 6: cv::rotate(image, transformed, cv::ROTATE_90_CLOCKWISE); break;
        case 7: cv::transpose(image, transformed); cv::flip(transformed, transformed, -1); break;
        case 8: cv::rotate(image, transformed, cv::ROTATE_90_COUNTERCLOCKWISE); break;
        default: return image;
    }
    return transformed;
}
}
