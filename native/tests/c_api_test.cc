#include "faceid.h"
#include <cassert>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <limits>

static void failure(int status, char *text, const char *code) {
    assert(status != 0);
    assert(text != nullptr);
    assert(std::strstr(text, code) != nullptr);
    faceid_free(text);
}

int main() {
    faceid_engine *engine = reinterpret_cast<faceid_engine *>(1);
    char *error = nullptr;
    int status = faceid_create(nullptr, &engine, &error);
    failure(status, error, "invalid_argument");
    assert(engine == nullptr);
    error = nullptr;
    status = faceid_model(nullptr, &error);
    failure(status, error, "invalid_argument");
    error = nullptr;
    status = faceid_process_bgr(nullptr, nullptr, 0, 112, 112, 2, 5, .95, &error);
    failure(status, error, "invalid_argument");
    error = nullptr;
    status = faceid_process_encoded(nullptr, nullptr, 0, 0, 5, .95, &error);
    failure(status, error, "invalid_argument");
    assert(faceid_model(nullptr, nullptr) != 0);
    assert(faceid_create("missing.json", nullptr, nullptr) != 0);
    faceid_destroy(nullptr);
    faceid_free(nullptr);

    const auto fixture = std::filesystem::temp_directory_path() / "faceid-invalid-model-fixture.json";
    { std::ofstream output(fixture); output << "{\"schema_version\":1}"; }
    error = nullptr;
    status = faceid_create(fixture.c_str(), &engine, &error);
    failure(status, error, "model_error");
    assert(engine == nullptr);
    // A 64-bit JSON integer must not wrap to an accepted two-thread value.
    { std::ofstream output(fixture); output << R"({"schema_version":1,"runtime":{"intra_op_threads":4294967298,"detector_intra_op_threads":2,"inter_op_threads":1,"parallel_backbones":true,"allow_spinning":false,"enable_cpu_mem_arena":true}})"; }
    error = nullptr;
    status = faceid_create(fixture.c_str(), &engine, &error);
    failure(status, error, "outside their supported range");
    assert(engine == nullptr);
    std::filesystem::remove(fixture);
    return 0;
}
