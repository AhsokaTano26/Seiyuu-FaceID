#if defined(FACEID_TEST_BROKEN_GUARD)
#include "denormal_guard.broken.hpp"
#else
#include "denormal_guard.hpp"
#endif

#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(__linux__) && defined(__x86_64__)
#include <xmmintrin.h>
#endif

using Guard = faceid::experimental::ScopedDenormalMode;
static void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

#if defined(__linux__) && defined(__x86_64__)
constexpr unsigned kModeMask = 0x8040U;
class RestoreCaller {
public:
    RestoreCaller() noexcept : saved_(_mm_getcsr()) {}
    ~RestoreCaller() noexcept { _mm_setcsr(saved_); }
private:
    unsigned saved_;
};

static void exercise_scope(unsigned seed) {
    _mm_setcsr(seed);
    {
        Guard on(true);
        check(_mm_getcsr() == (seed | kModeMask), "true must set only DAZ/FTZ");
        {
            Guard off(false);
            check(_mm_getcsr() == (seed & ~kModeMask), "false must explicitly clear DAZ/FTZ");
        }
        check(_mm_getcsr() == (seed | kModeMask), "nested false did not restore outer true");
    }
    check(_mm_getcsr() == seed, "normal return did not restore the complete caller MXCSR");
    try {
        Guard off(false);
        check(_mm_getcsr() == (seed & ~kModeMask), "exception scope mode is wrong");
        throw 42;
    } catch (int value) {
        check(value == 42, "unexpected exception marker");
    }
    check(_mm_getcsr() == seed, "exception did not restore complete caller MXCSR");
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static float runtime_multiply(float left, float right) { return left * right; }
static std::uint32_t bits(float value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
static void arithmetic() {
    RestoreCaller restore;
    _mm_setcsr(0x1f80U);  // nearest rounding, masked exceptions, gradual underflow
    volatile float subnormal = std::numeric_limits<float>::denorm_min();
    volatile float smallest_normal = std::numeric_limits<float>::min();
    volatile float one = 1.0f;
    volatile float half = 0.5f;
    {
        Guard off(false);
        check(bits(runtime_multiply(subnormal, one)) == 1U, "off lost a subnormal input");
        check(bits(runtime_multiply(smallest_normal, half)) == 0x00400000U, "off lost a subnormal result");
    }
    check(_mm_getcsr() == 0x1f80U, "off leaked arithmetic status flags");
    {
        Guard on(true);
        check(bits(runtime_multiply(subnormal, one)) == 0U, "DAZ did not zero a subnormal input");
        check(bits(runtime_multiply(smallest_normal, half)) == 0U, "FTZ did not zero a subnormal result");
    }
    check(_mm_getcsr() == 0x1f80U, "on leaked arithmetic status flags");
}
static void thread_scopes() {
    RestoreCaller restore;
    const auto original = _mm_getcsr();
    const std::array<unsigned, 4> modes{0U, 0x40U, 0x8000U, 0x8040U};
    for (unsigned rounding : {0U, 0x2000U, 0x4000U, 0x6000U}) {
        for (unsigned mode : modes) {
            // Keep exception masks safe, seed nonzero status flags, exercise every rounding mode.
            const unsigned seed = ((original | 0x1f80U) & ~0xe07fU) | rounding | mode | 0x21U;
            exercise_scope(seed);
        }
    }
    arithmetic();
}
#endif

int main() {
    try {
        if (!Guard::supported()) {
            Guard off(false);
            bool rejected = false;
            try { Guard on(true); }
            catch (const std::invalid_argument &) { rejected = true; }
            check(rejected, "unsupported true must throw std::invalid_argument");
            std::cout << "PASS: false is accepted; true is rejected on this unsupported platform.\n";
            std::cout << "SKIP: MXCSR, FTZ/DAZ arithmetic and x86 thread-state tests require supported Linux x86_64.\n";
            return 0;
        }
#if defined(__linux__) && defined(__x86_64__)
        thread_scopes();
        // Each thread persists across repeated calls, like the application's worker.
        std::array<std::exception_ptr, 8> failures{};
        std::array<std::thread, 8> threads;
        for (std::size_t index = 0; index < threads.size(); ++index) {
            threads[index] = std::thread([&, index] {
                try { for (int call = 0; call < 16; ++call) thread_scopes(); }
                catch (...) { failures[index] = std::current_exception(); }
            });
        }
        for (auto &thread : threads) thread.join();
        for (auto failure : failures) if (failure) std::rethrow_exception(failure);
        std::cout << "PASS: true/false, nesting, complete state restoration, exceptions, FTZ/DAZ arithmetic, 8 persistent threads.\n";
        return 0;
#else
        check(false, "non-Linux-x86_64 platform incorrectly reports support");
#endif
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
