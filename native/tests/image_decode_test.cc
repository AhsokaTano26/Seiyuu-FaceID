#include "image_decode.hpp"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include <opencv2/imgcodecs.hpp>

using Bytes = std::vector<unsigned char>;

static Bytes tiff(int orientation) {
    return {'I','I',42,0, 8,0,0,0, 1,0, 0x12,1,3,0,1,0,0,0,
            static_cast<unsigned char>(orientation),0,0,0, 0,0,0,0};
}
static void put32(Bytes &bytes, std::uint32_t value, bool big) {
    for (int i = 0; i < 4; ++i) bytes.push_back(static_cast<unsigned char>(value >> ((big ? 3-i : i) * 8)));
}
static Bytes jpeg(int orientation, const Bytes &encoded = Bytes{0xff,0xd8,0xff,0xd9}) {
    Bytes payload{'E','x','i','f',0,0};
    auto tags = tiff(orientation);
    payload.insert(payload.end(), tags.begin(), tags.end());
    Bytes bytes{0xff,0xd8,0xff,0xe1, 0,static_cast<unsigned char>(payload.size()+2)};
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    bytes.insert(bytes.end(), encoded.begin()+2, encoded.end());
    return bytes;
}
static std::uint32_t crc32(const unsigned char *bytes, std::size_t length) {
    std::uint32_t crc = 0xffffffff;
    for (std::size_t i = 0; i < length; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
    }
    return ~crc;
}
static Bytes png(int orientation, const Bytes &encoded = {}) {
    Bytes bytes = encoded.empty() ? Bytes{0x89,'P','N','G',13,10,26,10} : Bytes(encoded.begin(), encoded.begin()+33);
    auto tags = tiff(orientation);
    put32(bytes, static_cast<std::uint32_t>(tags.size()), true);
    const auto crc_start = bytes.size();
    bytes.insert(bytes.end(), {'e','X','I','f'});
    bytes.insert(bytes.end(), tags.begin(), tags.end());
    put32(bytes, crc32(bytes.data()+crc_start, bytes.size()-crc_start), true);
    if (!encoded.empty()) bytes.insert(bytes.end(), encoded.begin()+33, encoded.end());
    return bytes;
}
static Bytes webp(int orientation, const Bytes &encoded = {}) {
    auto tags = tiff(orientation);
    Bytes bytes{'R','I','F','F',0,0,0,0,'W','E','B','P'};
    if (!encoded.empty()) {
        assert(encoded.size() > 16 && std::memcmp(encoded.data()+12, "VP8X", 4) != 0);
        bytes.insert(bytes.end(), {'V','P','8','X'});
        put32(bytes, 10, false);
        // EXIF flag, reserved bytes, 40x24 canvas (minus one), all little-endian.
        bytes.insert(bytes.end(), {8,0,0,0,39,0,0,23,0,0});
        bytes.insert(bytes.end(), encoded.begin()+12, encoded.end());
    }
    bytes.insert(bytes.end(), {'E','X','I','F'});
    put32(bytes, static_cast<std::uint32_t>(tags.size()), false);
    bytes.insert(bytes.end(), tags.begin(), tags.end());
    if (tags.size() & 1) bytes.push_back(0);
    const auto length = bytes.size()-8;
    for (int i = 0; i < 4; ++i) bytes[4+i] = static_cast<unsigned char>(length >> (8*i));
    return bytes;
}

int main() {
    assert(faceid::exif_orientation(nullptr, 100) == 1);
    for (int orientation = 1; orientation <= 8; ++orientation) {
        for (auto fixture : {jpeg(orientation), png(orientation), webp(orientation)}) {
            assert(faceid::exif_orientation(fixture.data(), fixture.size()) == orientation);
            for (std::size_t length = 0; length < fixture.size(); ++length) {
                const int partial = faceid::exif_orientation(fixture.data(), length);
                assert(partial >= 1 && partial <= 8);
            }
        }
    }
    auto bad = jpeg(9);
    assert(faceid::exif_orientation(bad.data(), bad.size()) == 1);

    // Pillow converts 16-bit grayscale PNG to RGB by saturation, not division by 256.
    const unsigned short gray_values[] = {0,1,128,255,256,1024,50000,65535};
    cv::Mat gray16(1, 8, CV_16UC1, const_cast<unsigned short *>(gray_values));
    Bytes gray_encoded;
    assert(cv::imencode(".png", gray16, gray_encoded));
    const auto gray_decoded = faceid::decode_image(gray_encoded.data(), gray_encoded.size());
    for (int x = 0; x < 8; ++x) {
        const auto expected = static_cast<unsigned char>(std::min<unsigned short>(gray_values[x], 255));
        assert(gray_decoded.at<cv::Vec3b>(0,x) == cv::Vec3b(expected,expected,expected));
    }

    cv::Mat pixels(24, 40, CV_8UC3);
    for (int y = 0; y < pixels.rows; ++y)
        for (int x = 0; x < pixels.cols; ++x) pixels.at<cv::Vec3b>(y,x) = {static_cast<unsigned char>(x*5),static_cast<unsigned char>(y*9),static_cast<unsigned char>((x+y)*3)};
    for (const char *format : {".jpg", ".png", ".webp"}) {
      Bytes encoded;
      assert(cv::imencode(format, pixels, encoded));
      const auto baseline = cv::imdecode(encoded, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
      for (int orientation = 1; orientation <= 8; ++orientation) {
        auto fixture = std::strcmp(format, ".jpg") == 0 ? jpeg(orientation, encoded) :
                       std::strcmp(format, ".png") == 0 ? png(orientation, encoded) : webp(orientation, encoded);
        const auto decoded = faceid::decode_image(fixture.data(), fixture.size());
        assert(decoded.rows == (orientation >= 5 ? baseline.cols : baseline.rows));
        assert(decoded.cols == (orientation >= 5 ? baseline.rows : baseline.cols));
        for (int y = 0; y < baseline.rows; ++y) {
            for (int x = 0; x < baseline.cols; ++x) {
                int destination_y = y, destination_x = x;
                switch (orientation) {
                    case 2: destination_x = baseline.cols-1-x; break;
                    case 3: destination_y = baseline.rows-1-y; destination_x = baseline.cols-1-x; break;
                    case 4: destination_y = baseline.rows-1-y; break;
                    case 5: destination_y = x; destination_x = y; break;
                    case 6: destination_y = x; destination_x = baseline.rows-1-y; break;
                    case 7: destination_y = baseline.cols-1-x; destination_x = baseline.rows-1-y; break;
                    case 8: destination_y = baseline.cols-1-x; destination_x = y; break;
                }
                assert(decoded.at<cv::Vec3b>(destination_y,destination_x) == baseline.at<cv::Vec3b>(y,x));
            }
        }
      }
    }
}
