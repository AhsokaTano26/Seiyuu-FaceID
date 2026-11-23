#include "faceid.h"
#include "image_decode.hpp"
#include "denormal_guard.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/geometry.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace faceid {
using json = nlohmann::json;
constexpr std::size_t kEmbedding = 512;
constexpr std::size_t kFeatures = 1024;
constexpr std::size_t kMaxPixels = 20000000;

class Error : public std::runtime_error {
public:
    Error(std::string code, std::string message) : std::runtime_error(std::move(message)), code(std::move(code)) {}
    std::string code;
};

void require(bool condition, const char *message) {
    if (!condition) throw Error("model_error", message);
}
bool finite(double value) { return std::isfinite(value); }
int positive_int(const json &object, const char *name, int limit = 256) {
    const auto &value = object.at(name);
    require(value.is_number_integer(), "Runtime dimensions and threads must be integers.");
    const double parsed = value.get<double>();
    require(parsed > 0 && parsed <= limit, "Runtime dimensions or threads are outside their supported range.");
    return static_cast<int>(parsed);
}
double bounded(const json &object, const char *name, double minimum, double maximum) {
    const auto &value = object.at(name);
    require(value.is_number(), "Configuration value must be numeric.");
    const double parsed = value.get<double>();
    require(finite(parsed) && parsed >= minimum && parsed <= maximum, "Configuration value is outside its supported range.");
    return parsed;
}
bool boolean(const json &object, const char *name) {
    require(object.at(name).is_boolean(), "Runtime flag must be a boolean.");
    return object.at(name).get<bool>();
}
json load_config(const char *path) {
    std::ifstream stream(path);
    require(stream.good(), "Could not open native model configuration.");
    json value;
    stream >> value;
    require(value.is_object() && value.at("schema_version") == 1, "Unsupported native model schema.");
    return value;
}
double round_two(double value) { return std::nearbyint(value * 100.0) / 100.0; }
int round_size(double value) { return static_cast<int>(std::nearbyint(value)); }

struct Runtime {
    int threads, detector_threads, inter_threads;
    bool parallel, spinning, arena, flush_denormals;
    explicit Runtime(const json &j)
        : threads(positive_int(j, "intra_op_threads")), detector_threads(positive_int(j, "detector_intra_op_threads")),
          inter_threads(positive_int(j, "inter_op_threads")), parallel(boolean(j, "parallel_backbones")),
          spinning(boolean(j, "allow_spinning")), arena(boolean(j, "enable_cpu_mem_arena")),
          flush_denormals(j.contains("flush_denormals") ? boolean(j, "flush_denormals") : false) {
        require(!flush_denormals || experimental::ScopedDenormalMode::supported(),
                "flush_denormals requires Linux x86_64 with MXCSR FTZ and DAZ support.");
    }
};

class Model {
public:
    Model(Ort::Env &env, const std::filesystem::path &path, int threads, const Runtime &runtime)
        : session_(nullptr), memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          flush_denormals_(runtime.flush_denormals) {
        require(std::filesystem::is_regular_file(path), "An ONNX model file is missing.");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads);
        options.SetInterOpNumThreads(runtime.inter_threads);
        options.AddConfigEntry("session.intra_op.allow_spinning", runtime.spinning ? "1" : "0");
        options.AddConfigEntry("session.inter_op.allow_spinning", runtime.spinning ? "1" : "0");
        options.AddConfigEntry("session.set_denormal_as_zero", runtime.flush_denormals ? "1" : "0");
        if (!runtime.arena) options.DisableCpuMemArena();
        {
            // ORT 1.29 uses call_once for the constructing caller's FP state.
            // Restore that caller even when session construction throws.
            const experimental::ScopedDenormalMode mode(flush_denormals_);
            session_ = Ort::Session(env, path.c_str(), options);
        }
        require(session_.GetInputCount() == 1, "Expected exactly one model input.");
        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = session_.GetInputNameAllocated(0, allocator).get();
        const auto input_type = session_.GetInputTypeInfo(0);
        const auto input = input_type.GetTensorTypeAndShapeInfo();
        require(input.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "Expected float32 model input.");
        const auto shape = input.GetShape();
        require(shape.size() == 4 && (shape[0] == 1 || shape[0] < 0) && shape[1] == 3, "Expected 1x3xHxW input.");
        height_ = shape[2]; width_ = shape[3];
        for (std::size_t index = 0; index < session_.GetOutputCount(); ++index)
            output_names_.emplace_back(session_.GetOutputNameAllocated(index, allocator).get());
        for (const auto &name : output_names_) output_ptrs_.push_back(name.c_str());
    }

    std::vector<Ort::Value> run(std::vector<float> &input, int width, int height) {
        require((width_ < 0 || width_ == width) && (height_ < 0 || height_ == height), "Model spatial dimensions do not match configuration.");
        const std::array<std::int64_t, 4> shape{1, 3, height, width};
        auto tensor = Ort::Value::CreateTensor<float>(memory_, input.data(), input.size(), shape.data(), shape.size());
        const char *name = input_name_.c_str();
        std::vector<Ort::Value> outputs;
        {
            // Each cgo caller and the persistent R50 worker sets its own mode;
            // no floating-point state is kept across calls or leaked to Go.
            const experimental::ScopedDenormalMode mode(flush_denormals_);
            outputs = session_.Run(Ort::RunOptions{nullptr}, &name, &tensor, 1, output_ptrs_.data(), output_ptrs_.size());
        }
        for (const auto &value : outputs)
            require(value.IsTensor() && value.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                    "Expected float32 model output.");
        return outputs;
    }
    std::size_t output_count() const { return output_names_.size(); }
private:
    Ort::Session session_;
    Ort::MemoryInfo memory_;
    std::string input_name_;
    std::vector<std::string> output_names_;
    std::vector<const char *> output_ptrs_;
    std::int64_t height_, width_;
    bool flush_denormals_;
};

// One reusable worker. It never keeps a Go/C caller's pixels after a call returns.
class Worker {
public:
    Worker() : thread_([this] { loop(); }) {}
    ~Worker() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        ready_.notify_one();
        thread_.join();
    }
    std::future<std::vector<float>> submit(std::function<std::vector<float>()> fn) {
        auto task = std::make_shared<std::packaged_task<std::vector<float>()>>(std::move(fn));
        auto future = task->get_future();
        { std::lock_guard<std::mutex> lock(mutex_); task_ = [task] { (*task)(); }; }
        ready_.notify_one();
        return future;
    }
private:
    void loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return stop_ || bool(task_); });
                if (stop_ && !task_) return;
                task.swap(task_);
            }
            task();
        }
    }
    std::mutex mutex_;
    std::condition_variable ready_;
    std::function<void()> task_;
    bool stop_ = false;
    std::thread thread_;
};

struct Detection {
    std::array<float, 4> box;
    float score;
    std::array<cv::Point2f, 5> points;
};

class Engine {
public:
    explicit Engine(const char *path)
        : config_(load_config(path)), runtime_(config_.at("runtime")), env_(ORT_LOGGING_LEVEL_WARNING, "seiyuu-faceid-native") {
        env_.DisableTelemetryEvents();
        const auto base = std::filesystem::absolute(path).parent_path();
        const auto &models = config_.at("models");
        const auto &detector = config_.at("detector");
        const auto &input_size = detector.at("input_size");
        require(input_size.is_array() && input_size.size() == 2, "Detector input size must contain width and height.");
        require(input_size[0].is_number_integer() && input_size[1].is_number_integer(), "Detector input size must be integers.");
        require(input_size[0].get<double>() > 0 && input_size[0].get<double>() <= 4096 &&
                input_size[1].get<double>() > 0 && input_size[1].get<double>() <= 4096, "Detector input size is outside its supported range.");
        det_width_ = input_size[0].get<int>(); det_height_ = input_size[1].get<int>();
        require(det_width_ > 0 && det_height_ > 0 && det_width_ <= 4096 && det_height_ <= 4096 &&
                det_width_ % 32 == 0 && det_height_ % 32 == 0, "Detector input size must be positive multiples of 32, at most 4096.");
        max_long_edge_ = positive_int(detector, "max_long_edge", 20000);
        score_threshold_ = bounded(detector, "score_threshold", 0.000001, 0.999999);
        nms_threshold_ = bounded(detector, "nms_threshold", 0, 1);
        min_face_size_ = positive_int(config_.at("quality"), "min_face_size", 20000);
        min_sharpness_ = bounded(config_.at("quality"), "min_sharpness", 0, 1e12);
        threshold_ = bounded(config_.at("recognition"), "probability_threshold", 0, 1);
        names_ = config_.at("class_names").get<std::vector<std::string>>();
        require(names_.size() >= 2 && names_.size() <= 10000, "Invalid number of identity classes.");
        require(std::set<std::string>(names_.begin(), names_.end()).size() == names_.size(), "Duplicate identity name.");
        for (const auto &name : names_) require(!name.empty(), "Empty identity name.");
        top_k_ = positive_int(config_.at("recognition"), "top_k", static_cast<int>(names_.size()));
        model_id_ = config_.at("model_id").get<std::string>();
        require(model_id_.size() == 64 && model_id_.find_first_not_of("0123456789abcdef") == std::string::npos, "Invalid model identifier.");
        const auto &head = config_.at("head");
        mean_ = head.at("mean").get<std::vector<double>>();
        scale_ = head.at("scale").get<std::vector<double>>();
        intercept_ = head.at("intercept").get<std::vector<float>>();
        const auto coefficients = head.at("coefficients").get<std::vector<std::vector<float>>>();
        require(mean_.size() == kFeatures && scale_.size() == kFeatures && intercept_.size() == names_.size() && coefficients.size() == names_.size(), "Classifier parameter dimensions do not match.");
        for (std::size_t i = 0; i < kFeatures; ++i)
            require(finite(mean_[i]) && finite(scale_[i]) && scale_[i] > 0, "Invalid scaler parameter.");
        for (std::size_t i = 0; i < names_.size(); ++i) {
            require(coefficients[i].size() == kFeatures && finite(intercept_[i]), "Invalid classifier row.");
            for (const float value : coefficients[i]) require(finite(value), "Nonfinite classifier coefficient.");
            coefficients_.insert(coefficients_.end(), coefficients[i].begin(), coefficients[i].end());
        }
        detector_ = std::make_unique<Model>(env_, base / models.at("detector").get<std::string>(), runtime_.detector_threads, runtime_);
        adaface_ = std::make_unique<Model>(env_, base / models.at("adaface").get<std::string>(), runtime_.threads, runtime_);
        r50_ = std::make_unique<Model>(env_, base / models.at("r50").get<std::string>(), runtime_.threads, runtime_);
        require(detector_->output_count() == 9 && adaface_->output_count() == 1 && r50_->output_count() == 1, "Unexpected number of model outputs.");
        if (runtime_.parallel) worker_ = std::make_unique<Worker>();
        // Parameters now have compact typed storage; do not retain a second JSON copy.
        config_ = json();
    }

    json metadata() const {
        return {{"model_id", model_id_}, {"recognition_id", "native-v1:" + model_id_},
                {"class_count", names_.size()}, {"class_names", names_},
                {"defaults", {{"top_k", top_k_}, {"probability_threshold", threshold_}}}};
    }
    void validate_request(int mode, int top_k, double threshold) const {
        if (mode < 0 || mode > 3 || top_k < 1 || std::size_t(top_k) > names_.size() || !finite(threshold) || threshold < 0 || threshold > 1)
            throw Error("invalid_argument", "Invalid recognition parameters.");
    }

    json process(const cv::Mat &image, int mode, int top_k, double threshold) {
        std::lock_guard<std::mutex> lock(inference_mutex_);
        validate_request(mode, top_k, threshold);
        if (mode == 2) {
            if (image.cols != 112 || image.rows != 112) throw Error("invalid_argument", "Aligned crops must be 112 by 112 pixels.");
            const auto features = embed(image);
            const auto probabilities = score(features);
            return {{"features", features}, {"probabilities", probabilities}, {"candidates", candidates(probabilities, top_k)}};
        }
        json faces = json::array();
        const auto min_score = mode == 3 && threshold > 0 ? threshold : float(score_threshold_);
        for (const auto &detection : detect(image, min_score, mode != 3)) {
            json box = json::array();
            for (float value : detection.box) box.push_back(round_two(value));
            json item{{"bbox", box}, {"detector_score", detection.score}};
            if (mode == 1) {
                json points = json::array();
                for (auto point : detection.points) points.push_back({round_two(point.x), round_two(point.y)});
                item["landmarks"] = points;
            } else if (mode == 0) {
                const auto quality = assess_quality(image, detection);
                item["quality_score"] = quality.at("score");
                item["quality_reasons"] = quality.at("reasons");
                item["probability_threshold"] = threshold;
                item["score_type"] = "class_probability";
                item["identity"] = nullptr;
                item["status"] = "review";
                item["candidates"] = json::array();
                if (!quality.at("reasons").empty()) {
                    item["decision_reason"] = "quality_rejected";
                } else {
                    const auto ranked = candidates(score(embed(align(image, detection))), top_k);
                    item["candidates"] = ranked;
                    if (ranked[0].at("score").get<double>() >= threshold) {
                        item["identity"] = ranked[0].at("identity");
                        item["status"] = "recognized";
                        item["decision_reason"] = nullptr;
                    } else item["decision_reason"] = "low_probability";
                }
            }
            faces.push_back(std::move(item));
        }
        return {{"width", image.cols}, {"height", image.rows}, {"faces", faces}};
    }

private:
    std::vector<float> embed_one(Model &model, const cv::Mat &image, bool bgr) {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(112, 112), 0, 0, cv::INTER_LINEAR);
        std::vector<float> input(3 * 112 * 112);
        for (int y = 0; y < 112; ++y) {
            const auto *row = resized.ptr<cv::Vec3b>(y);
            for (int x = 0; x < 112; ++x) {
                for (int c = 0; c < 3; ++c) {
                    float value = row[x][bgr ? c : 2 - c];
                    if (bgr) { value /= 255.0f; value -= 0.5f; value /= 0.5f; }
                    else { value /= 127.5f; value -= 1.0f; }
                    input[(c * 112 + y) * 112 + x] = value;
                }
            }
        }
        auto output = model.run(input, 112, 112);
        require(output[0].GetTensorTypeAndShapeInfo().GetElementCount() == kEmbedding, "Recognizer must produce 512 dimensions.");
        const float *values = output[0].GetTensorData<float>();
        double squares = 0;
        for (std::size_t i = 0; i < kEmbedding; ++i) {
            require(finite(values[i]), "Recognizer returned nonfinite values.");
            squares += double(values[i]) * values[i];
        }
        // NumPy's norm returns a float32 dot product and float32 square root.
        const float norm = std::sqrt(static_cast<float>(squares));
        require(finite(norm) && norm > 1e-12f, "Recognizer returned a zero norm.");
        std::vector<float> result(kEmbedding);
        for (std::size_t i = 0; i < kEmbedding; ++i) result[i] = values[i] / norm;
        return result;
    }
    std::vector<float> embed(const cv::Mat &image) {
        std::vector<float> first, second;
        if (worker_) {
            auto future = worker_->submit([this, image] { return embed_one(*r50_, image, false); });
            try { first = embed_one(*adaface_, image, true); }
            catch (...) { future.wait(); throw; }
            second = future.get();
        } else {
            first = embed_one(*adaface_, image, true);
            second = embed_one(*r50_, image, false);
        }
        first.insert(first.end(), second.begin(), second.end());
        const float factor = static_cast<float>(std::sqrt(2.0));
        for (float &value : first) value /= factor;
        return first;
    }
    std::vector<float> score(const std::vector<float> &features) const {
        std::array<float, kFeatures> scaled;
        for (std::size_t i = 0; i < kFeatures; ++i) {
            scaled[i] = static_cast<float>(double(features[i]) - mean_[i]);
            scaled[i] = static_cast<float>(double(scaled[i]) / scale_[i]);
        }
        std::vector<float> logits(names_.size());
        for (std::size_t row = 0; row < names_.size(); ++row) {
            double sum = 0;
            for (std::size_t i = 0; i < kFeatures; ++i) sum += double(scaled[i]) * coefficients_[row * kFeatures + i];
            logits[row] = static_cast<float>(sum) + intercept_[row];
        }
        const float maximum = *std::max_element(logits.begin(), logits.end());
        double total = 0;
        for (float &value : logits) { value = std::exp(value - maximum); total += value; }
        const float sum = static_cast<float>(total);
        require(finite(sum) && sum > 0, "Classifier returned invalid probabilities.");
        for (float &value : logits) value /= sum;
        return logits;
    }
    json candidates(const std::vector<float> &probabilities, int top_k) const {
        std::vector<std::size_t> order(names_.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return probabilities[a] > probabilities[b]; });
        json result = json::array();
        for (int index = 0; index < top_k; ++index)
            result.push_back({{"identity", names_[order[index]]}, {"score", probabilities[order[index]]}});
        return result;
    }

    std::vector<Detection> detect(const cv::Mat &image, double min_score, bool need_landmarks) {
        cv::Mat working = image;
        double resize_scale = 1;
        const int long_edge = std::max(image.cols, image.rows);
        if (long_edge > max_long_edge_) {
            resize_scale = double(max_long_edge_) / long_edge;
            const int width = round_size(image.cols * resize_scale), height = round_size(image.rows * resize_scale);
            if (width < 1 || height < 1) throw Error("invalid_image", "Image aspect ratio is unsupported.");
            cv::resize(image, working, cv::Size(width, height));
        }
        const double scale = std::min(double(det_width_) / working.cols, double(det_height_) / working.rows);
        const int width = round_size(working.cols * scale), height = round_size(working.rows * scale);
        if (width < 1 || height < 1) throw Error("invalid_image", "Image aspect ratio is unsupported.");
        cv::Mat resized;
        cv::resize(working, resized, cv::Size(width, height));
        const int pad_x = (det_width_ - width) / 2, pad_y = (det_height_ - height) / 2;
        cv::Mat canvas = cv::Mat::zeros(det_height_, det_width_, CV_8UC3);
        resized.copyTo(canvas(cv::Rect(pad_x, pad_y, width, height)));
        std::vector<float> input(std::size_t(3) * det_width_ * det_height_);
        for (int y = 0; y < det_height_; ++y) {
            const auto *row = canvas.ptr<cv::Vec3b>(y);
            for (int x = 0; x < det_width_; ++x)
                for (int c = 0; c < 3; ++c)
                    input[(c * det_height_ + y) * det_width_ + x] = (float(row[x][2 - c]) - 127.5f) * (1.0f / 128.0f);
        }
        auto output = detector_->run(input, det_width_, det_height_);
        std::vector<Detection> detections;
        for (int level = 0; level < 3; ++level) {
            const int stride = 8 << level, feature_width = det_width_ / stride, feature_height = det_height_ / stride;
            const auto count = output[level].GetTensorTypeAndShapeInfo().GetElementCount();
            const auto locations = std::size_t(feature_width) * feature_height;
            require(count > 0 && count % locations == 0 && count / locations <= 16, "Detector output shape does not match input.");
            require(output[level + 3].GetTensorTypeAndShapeInfo().GetElementCount() == count * 4 && output[level + 6].GetTensorTypeAndShapeInfo().GetElementCount() == count * 10, "Detector output tensors have incompatible shapes.");
            const auto anchors = count / locations;
            const auto *scores = output[level].GetTensorData<float>();
            const auto *distances = output[level + 3].GetTensorData<float>();
            const auto *points = output[level + 6].GetTensorData<float>();
            for (std::size_t index = 0; index < count; ++index) {
                require(finite(scores[index]), "Detector returned a nonfinite score.");
                if (scores[index] < min_score) continue;
                const float center_x = float((index / anchors) % feature_width) * stride;
                const float center_y = float((index / anchors) / feature_width) * stride;
                Detection detection{{center_x - distances[4 * index] * stride, center_y - distances[4 * index + 1] * stride,
                                     center_x + distances[4 * index + 2] * stride, center_y + distances[4 * index + 3] * stride}, scores[index], {}};
                for (std::size_t i = 0; need_landmarks && i < 5; ++i) {
                    detection.points[i] = {center_x + points[index * 10 + i * 2] * stride, center_y + points[index * 10 + i * 2 + 1] * stride};
                    require(finite(detection.points[i].x) && finite(detection.points[i].y), "Detector returned invalid landmarks.");
                }
                for (float value : detection.box) require(finite(value), "Detector returned an invalid box.");
                detections.push_back(detection);
            }
        }
        std::vector<std::size_t> order(detections.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
            return detections[a].score == detections[b].score ? a > b : detections[a].score > detections[b].score;
        });
        std::vector<Detection> kept;
        auto area = [](const auto &box) { return std::max(0.0f, box[2] - box[0]) * std::max(0.0f, box[3] - box[1]); };
        while (!order.empty()) {
            const auto current = order.front();
            Detection selected = detections[current];
            std::vector<std::size_t> remaining;
            for (std::size_t p = 1; p < order.size(); ++p) {
                const auto &a = selected.box;
                const auto &b = detections[order[p]].box;
                const float intersection = std::max(0.0f, std::min(a[2], b[2]) - std::max(a[0], b[0])) *
                                           std::max(0.0f, std::min(a[3], b[3]) - std::max(a[1], b[1]));
                const float combined = area(a) + area(b) - intersection;
                const float iou = combined > 0 ? intersection / combined : 0;
                if (iou <= float(nms_threshold_)) remaining.push_back(order[p]);
            }
            order.swap(remaining);
            for (std::size_t i = 0; i < 4; ++i) {
                const bool x = i % 2 == 0;
                float value = selected.box[i] - float(x ? pad_x : pad_y);
                value /= float(scale); value /= float(resize_scale);
                selected.box[i] = std::clamp(value, 0.0f, float((x ? image.cols : image.rows) - 1));
            }
            if (need_landmarks) {
                for (auto &point : selected.points) {
                    point.x -= float(pad_x); point.x /= float(scale); point.x /= float(resize_scale);
                    point.y -= float(pad_y); point.y /= float(scale); point.y /= float(resize_scale);
                }
            }
            kept.push_back(selected);
        }
        return kept;
    }
    json assess_quality(const cv::Mat &image, const Detection &detection) const {
        const auto &box = detection.box;
        const double size = std::min(box[2] - box[0], box[3] - box[1]);
        const int x1 = std::clamp(int(std::floor(box[0])), 0, image.cols), y1 = std::clamp(int(std::floor(box[1])), 0, image.rows);
        const int x2 = std::clamp(int(std::ceil(box[2])), 0, image.cols), y2 = std::clamp(int(std::ceil(box[3])), 0, image.rows);
        double sharpness = 0;
        if (x2 > x1 && y2 > y1) {
            cv::Mat gray, laplacian;
            cv::cvtColor(image(cv::Rect(x1, y1, x2 - x1, y2 - y1)), gray, cv::COLOR_BGR2GRAY);
            cv::Laplacian(gray, laplacian, CV_64F);
            cv::Scalar mean, deviation;
            cv::meanStdDev(laplacian, mean, deviation);
            sharpness = deviation[0] * deviation[0];
        }
        json reasons = json::array();
        if (size < min_face_size_) reasons.push_back("face_too_small");
        if (sharpness < min_sharpness_) reasons.push_back("too_blurry");
        const double size_score = std::min(1.0, size / std::max(min_face_size_ * 2, 1));
        const double sharpness_score = std::min(1.0, sharpness / std::max(min_sharpness_ * 2, 1.0));
        return {{"score", (size_score + sharpness_score) / 2}, {"reasons", reasons}};
    }
    cv::Mat align(const cv::Mat &image, const Detection &detection) const {
        static const std::vector<cv::Point2f> target{{38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f}, {41.5493f, 92.3655f}, {70.7299f, 92.2041f}};
        const std::vector<cv::Point2f> source(detection.points.begin(), detection.points.end());
        const auto transform = cv::estimateAffinePartial2D(source, target, cv::noArray(), cv::LMEDS);
        require(!transform.empty(), "Could not estimate face alignment.");
        cv::Mat result;
        cv::warpAffine(image, result, transform, cv::Size(112, 112), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
        return result;
    }

    json config_;
    Runtime runtime_;
    Ort::Env env_;
    int det_width_, det_height_, max_long_edge_, min_face_size_, top_k_;
    double score_threshold_, nms_threshold_, min_sharpness_, threshold_;
    std::string model_id_;
    std::vector<std::string> names_;
    std::vector<double> mean_, scale_;
    std::vector<float> coefficients_, intercept_;
    std::unique_ptr<Model> detector_, adaface_, r50_;
    std::unique_ptr<Worker> worker_;
    std::mutex inference_mutex_;
};

char *copy_json(const json &value) {
    const std::string text = value.dump();
    auto *buffer = static_cast<char *>(std::malloc(text.size() + 1));
    if (!buffer) throw std::bad_alloc();
    std::memcpy(buffer, text.c_str(), text.size() + 1);
    return buffer;
}
template<class Function> int boundary(char **result, Function function) noexcept {
    if (!result) return 1;
    *result = nullptr;
    try { *result = copy_json(function()); return 0; }
    catch (...) {
        try {
            try { throw; }
            catch (const Error &error) { *result = copy_json({{"error", {{"code", error.code}, {"message", error.what()}}}}); }
            catch (const json::exception &) { *result = copy_json({{"error", {{"code", "model_error"}, {"message", "Invalid native model configuration."}}}}); }
            catch (const std::bad_alloc &) { *result = copy_json({{"error", {{"code", "internal_error"}, {"message", "Insufficient memory for inference."}}}}); }
            catch (...) { *result = copy_json({{"error", {{"code", "internal_error"}, {"message", "Native inference failed."}}}}); }
        } catch (...) { *result = nullptr; }
        return 1;
    }
}
}

struct faceid_engine { explicit faceid_engine(const char *path) : impl(path) {} faceid::Engine impl; };

extern "C" int faceid_create(const char *path, faceid_engine **engine, char **error) {
    if (engine) *engine = nullptr;
    if (!error) return 1;
    std::unique_ptr<faceid_engine> created;
    const int status = faceid::boundary(error, [&]() -> faceid::json {
        if (!path || !engine) throw faceid::Error("invalid_argument", "Model path and output pointer are required.");
        setenv("ORT_DISABLE_TELEMETRY", "1", 1);
        created = std::make_unique<faceid_engine>(path);
        return nullptr;
    });
    if (!status) { *engine = created.release(); faceid_free(*error); *error = nullptr; }
    return status;
}
extern "C" void faceid_destroy(faceid_engine *engine) { delete engine; }
extern "C" void faceid_free(char *text) { std::free(text); }
extern "C" int faceid_model(faceid_engine *engine, char **result) {
    return faceid::boundary(result, [&]() -> faceid::json {
        if (!engine) throw faceid::Error("invalid_argument", "An initialized engine is required.");
        return engine->impl.metadata();
    });
}
extern "C" int faceid_process_bgr(faceid_engine *engine, const unsigned char *pixels, std::size_t length,
                                   int width, int height, int mode, int top_k, double threshold, char **result) {
    return faceid::boundary(result, [&]() -> faceid::json {
        if (!engine || !pixels || width <= 0 || height <= 0 || std::size_t(width) > faceid::kMaxPixels / std::size_t(height))
            throw faceid::Error("invalid_argument", "Invalid BGR image dimensions or engine.");
        if (length != std::size_t(width) * height * 3) throw faceid::Error("invalid_argument", "BGR image buffer length does not match its dimensions.");
        cv::Mat image(height, width, CV_8UC3, const_cast<unsigned char *>(pixels));
        return engine->impl.process(image, mode, top_k, threshold);
    });
}
extern "C" int faceid_process_encoded(faceid_engine *engine, const unsigned char *bytes, std::size_t length,
                                       int mode, int top_k, double threshold, char **result) {
    return faceid::boundary(result, [&]() -> faceid::json {
        if (!engine || !bytes) throw faceid::Error("invalid_argument", "An initialized engine and image bytes are required.");
        if (!length) throw faceid::Error("empty_file", "The uploaded file is empty.");
        if (length > 10 * 1024 * 1024) throw faceid::Error("file_too_large", "The uploaded file exceeds the byte limit.");
        engine->impl.validate_request(mode, top_k, threshold);
        cv::Mat image;
        try { image = faceid::decode_image(bytes, length); }
        catch (const std::invalid_argument &error) {
            if (std::string(error.what()) == "image_too_large") throw faceid::Error("image_too_large", "The image exceeds the pixel limit.");
            throw faceid::Error("invalid_image", "Could not decode image.");
        } catch (const cv::Exception &) { throw faceid::Error("invalid_image", "Could not decode image."); }
        return engine->impl.process(image, mode, top_k, threshold);
    });
}
