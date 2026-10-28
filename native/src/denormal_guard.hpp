#pragma once
#include <stdexcept>

#if defined(__linux__) && defined(__x86_64__)
#include <array>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#endif

// Experimental, per-call floating-point mode. No process-global mutable state.
// Use around Session construction and Run, never across separate cgo calls.
namespace faceid::experimental {
class ScopedDenormalMode {
public:
    static bool supported() noexcept {
#if defined(__linux__) && defined(__x86_64__)
        // DAZ is not guaranteed merely by SSE support. The FXSAVE image exposes
        // supported MXCSR bits; zero means the conservative architectural mask.
        static const std::uint32_t supported_bits = []() noexcept {
            alignas(16) std::array<unsigned char, 512> state{};
            _fxsave(state.data());
            std::uint32_t mask = 0;
            std::memcpy(&mask, state.data() + 28, sizeof(mask));
            return mask == 0 ? std::uint32_t{0x0000ffbfU} : mask;
        }();
        return (supported_bits & kModeMask) == kModeMask;
#else
        return false;
#endif
    }

    explicit ScopedDenormalMode(bool enabled) {
        if (enabled && !supported()) {
            throw std::invalid_argument(
                "flush_denormals requires Linux x86_64 with MXCSR FTZ and DAZ support");
        }
#if defined(__linux__) && defined(__x86_64__)
        saved_ = _mm_getcsr();
        const unsigned target = (saved_ & ~kModeMask) | (enabled ? kModeMask : 0U);
        _mm_setcsr(target);
#endif
    }

    ~ScopedDenormalMode() noexcept {
#if defined(__linux__) && defined(__x86_64__)
        _mm_setcsr(saved_);
#endif
    }
    ScopedDenormalMode(const ScopedDenormalMode &) = delete;
    ScopedDenormalMode &operator=(const ScopedDenormalMode &) = delete;
    ScopedDenormalMode(ScopedDenormalMode &&) = delete;
    ScopedDenormalMode &operator=(ScopedDenormalMode &&) = delete;
private:
#if defined(__linux__) && defined(__x86_64__)
    static constexpr unsigned kModeMask = 0x8040U;  // FTZ bit 15, DAZ bit 6
    unsigned saved_ = 0;
#endif
};
}
