// Unit tests for BallPlateEnv. No GL, no torch.
//
// Covers:
//   1) reset() is deterministic given a seed.
//   2) Different seeds produce different initial positions.
//   3) Zero action, ball starts at center → episode terminates by maxSteps.
//   4) Full tilt held → ball falls off → episode terminates with big
//      negative terminal reward.
//   5) reset() after termination clears done + step count.

#include "env.hpp"

#include <cmath>
#include <cstdio>

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
approxEq(float a, float b, float eps = 1e-6f)
{
    return std::fabs(a - b) <= eps;
}
} // namespace

int
main()
{
    // 1) Reset determinism.
    {
        BallPlateEnv a, b;
        Observation oa = a.reset(1234);
        Observation ob = b.reset(1234);
        CHECK(approxEq(oa.px, ob.px));
        CHECK(approxEq(oa.pz, ob.pz));
        CHECK(oa.vx == 0.0f && oa.vz == 0.0f);
        CHECK(oa.tiltX == 0.0f && oa.tiltZ == 0.0f);
        std::printf("test 1 OK  reset determinism\n");
    }

    // 2) Different seeds → (very likely) different starts.
    {
        BallPlateEnv e;
        Observation o1 = e.reset(1);
        Observation o2 = e.reset(2);
        CHECK(!(approxEq(o1.px, o2.px) && approxEq(o1.pz, o2.pz)));
        std::printf("test 2 OK  seed sensitivity\n");
    }

    // 3) No action from center → episode reaches maxSteps, never falls.
    {
        BallPlateEnv::Config cfg;
        cfg.maxSteps    = 100;
        cfg.initHalfRange = 0.0f; // spawn exactly at center
        BallPlateEnv e(cfg);
        e.reset(7);
        int steps    = 0;
        bool fellOff = false;
        StepResult r{};
        while (!e.done())
        {
            r = e.step(0.0f, 0.0f);
            ++steps;
            if (std::fabs(r.obs.px) > cfg.plateWidth * 0.5f
                || std::fabs(r.obs.pz) > cfg.plateDepth * 0.5f)
                fellOff = true;
        }
        CHECK(steps == cfg.maxSteps);
        CHECK(!fellOff);
        CHECK(r.reward > -1.0f); // small penalty, no terminal
        std::printf("test 3 OK  timeout termination, steps=%d\n", steps);
    }

    // 4) Full tilt from center → ball must fall off within maxSteps.
    {
        BallPlateEnv::Config cfg;
        cfg.maxSteps    = 600;
        cfg.initHalfRange = 0.0f;
        BallPlateEnv e(cfg);
        e.reset(9);
        StepResult r{};
        int steps = 0;
        while (!e.done())
        {
            r = e.step(1.0f, 1.0f);
            ++steps;
        }
        bool fell = std::fabs(r.obs.px) > cfg.plateWidth * 0.5f
                    || std::fabs(r.obs.pz) > cfg.plateDepth * 0.5f;
        CHECK(fell);
        CHECK(steps < cfg.maxSteps); // fell before timeout
        CHECK(r.reward < -50.0f);    // terminal penalty applied
        std::printf("test 4 OK  fall-off termination at step %d, r=%.2f\n",
                    steps, r.reward);
    }

    // 5) reset() clears done and step count.
    {
        BallPlateEnv::Config cfg;
        cfg.maxSteps = 5;
        BallPlateEnv e(cfg);
        e.reset(3);
        while (!e.done())
            e.step(0.0f, 0.0f);
        CHECK(e.done());
        CHECK(e.stepCount() == cfg.maxSteps);
        e.reset(3);
        CHECK(!e.done());
        CHECK(e.stepCount() == 0);
        std::printf("test 5 OK  reset clears state\n");
    }

    if (g_failures)
    {
        std::fprintf(stderr, "\n%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("\nall tests passed\n");
    return 0;
}
