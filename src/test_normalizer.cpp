// Tests for ObsNormalizer. Pure C++, no torch, no GL.

#include "obs_normalizer.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace
{
int g_failures = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond))                                                           \
        {                                                                      \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__,       \
                         #cond);                                               \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

bool
approx(double a, double b, double eps = 1e-3)
{
    return std::fabs(a - b) <= eps;
}
} // namespace

int
main()
{
    // 1) Before any observations: normalize() is a no-op.
    {
        ObsNormalizer n(3);
        float x[3] = {1.0f, -2.0f, 5.0f};
        n.normalize(x);
        CHECK(x[0] == 1.0f && x[1] == -2.0f && x[2] == 5.0f);
        std::printf("test 1 OK  no-op before observe\n");
    }

    // 2) Recover a known mean and std to reasonable precision.
    {
        ObsNormalizer n(2);
        std::mt19937 rng(123);
        std::normal_distribution<float> d0(3.0f, 2.0f); // mean 3, sd 2
        std::normal_distribution<float> d1(-1.0f, 0.5f);
        for (int i = 0; i < 20000; ++i)
        {
            float o[2] = {d0(rng), d1(rng)};
            n.observe(o);
        }
        CHECK(approx(n.mean()[0], 3.0, 0.05));
        CHECK(approx(n.mean()[1], -1.0, 0.02));
        auto sd = n.stddev();
        CHECK(approx(sd[0], 2.0, 0.05));
        CHECK(approx(sd[1], 0.5, 0.02));
        std::printf("test 2 OK  mean=[%.3f %.3f] sd=[%.3f %.3f]\n",
                    n.mean()[0], n.mean()[1], sd[0], sd[1]);
    }

    // 3) After training, normalize() produces ~zero-mean, ~unit-variance.
    {
        ObsNormalizer n(1);
        std::mt19937 rng(7);
        std::normal_distribution<float> d(10.0f, 4.0f);
        std::vector<float> samples;
        samples.reserve(10000);
        for (int i = 0; i < 10000; ++i)
        {
            float v = d(rng);
            samples.push_back(v);
            n.observe(&v);
        }
        double sum = 0, sumsq = 0;
        for (float v : samples)
        {
            n.normalize(&v);
            sum += v;
            sumsq += double(v) * v;
        }
        double mean = sum / samples.size();
        double var  = sumsq / samples.size() - mean * mean;
        CHECK(approx(mean, 0.0, 0.02));
        CHECK(approx(std::sqrt(var), 1.0, 0.02));
        std::printf("test 3 OK  normalized mean=%.4f sd=%.4f\n", mean,
                    std::sqrt(var));
    }

    if (g_failures)
    {
        std::fprintf(stderr, "\n%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("\nall tests passed\n");
    return 0;
}
