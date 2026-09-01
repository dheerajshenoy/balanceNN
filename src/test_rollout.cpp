// Tests for RolloutBuffer. Pure C++, no torch, no GL.

#include "rollout_buffer.hpp"

#include <cmath>
#include <cstdio>
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
approx(float a, float b, float eps = 1e-5f)
{
    return std::fabs(a - b) <= eps;
}
} // namespace

int
main()
{
    // 1) Storage: size grows, obs / action round-trip through raw storage.
    {
        RolloutBuffer buf(4, /*obs*/ 2, /*act*/ 1);
        CHECK(buf.size() == 0);
        CHECK(!buf.full());
        float obs[2]    = {1.0f, 2.0f};
        float action[1] = {0.5f};
        buf.add(obs, action, /*logp*/ -0.7f, /*rew*/ 1.0f, /*done*/ false,
                /*value*/ 0.3f);
        CHECK(buf.size() == 1);
        CHECK(buf.obs()[0] == 1.0f && buf.obs()[1] == 2.0f);
        CHECK(buf.actions()[0] == 0.5f);
        CHECK(buf.logProbs()[0] == -0.7f);
        CHECK(buf.rewards()[0] == 1.0f);
        CHECK(buf.dones()[0] == 0);
        CHECK(buf.values()[0] == 0.3f);
        std::printf("test 1 OK  storage round-trip\n");
    }

    // 2) GAE with lambda=1, gamma=1, all-zero values, no dones, last_value=0.
    //    In this degenerate case, advantage_t == sum of future rewards
    //    (Monte-Carlo return with no baseline), so it's an easy hand-check.
    {
        RolloutBuffer buf(4, 1, 1);
        float o = 0, a = 0;
        for (int i = 0; i < 4; ++i)
            buf.add(&o, &a, 0.0f, /*rew*/ 1.0f, /*done*/ false, /*value*/ 0.0f);
        buf.computeGAE(/*last_value*/ 0.0f, /*gamma*/ 1.0f, /*lambda*/ 1.0f);
        CHECK(approx(buf.advantages()[0], 4.0f));
        CHECK(approx(buf.advantages()[1], 3.0f));
        CHECK(approx(buf.advantages()[2], 2.0f));
        CHECK(approx(buf.advantages()[3], 1.0f));
        // returns = advantages + values (values are 0 here).
        CHECK(approx(buf.returns()[0], 4.0f));
        CHECK(approx(buf.returns()[3], 1.0f));
        std::printf("test 2 OK  MC returns  adv=[%.1f %.1f %.1f %.1f]\n",
                    buf.advantages()[0], buf.advantages()[1],
                    buf.advantages()[2], buf.advantages()[3]);
    }

    // 3) done=true resets bootstrapping. Two mini-episodes of length 2, each
    //    with r=1 each step. With gamma=1, lambda=1, v=0, last_value=0:
    //    step 0 (done=false): adv = r0 + adv1 = 1 + 1 = 2
    //    step 1 (done=true) : adv = r1 + 0     = 1
    //    step 2 (done=false): adv = r2 + adv3 = 1 + 1 = 2
    //    step 3 (done=true) : adv = r3 + 0     = 1
    {
        RolloutBuffer buf(4, 1, 1);
        float o = 0, a = 0;
        buf.add(&o, &a, 0, 1.0f, false, 0.0f);
        buf.add(&o, &a, 0, 1.0f, true, 0.0f);
        buf.add(&o, &a, 0, 1.0f, false, 0.0f);
        buf.add(&o, &a, 0, 1.0f, true, 0.0f);
        buf.computeGAE(0.0f, 1.0f, 1.0f);
        CHECK(approx(buf.advantages()[0], 2.0f));
        CHECK(approx(buf.advantages()[1], 1.0f));
        CHECK(approx(buf.advantages()[2], 2.0f));
        CHECK(approx(buf.advantages()[3], 1.0f));
        std::printf("test 3 OK  done resets bootstrapping\n");
    }

    // 4) One-step TD (lambda=0) with gamma=0.9. Non-trivial hand check:
    //    values = [1, 2, 3], rewards = [0, 0, 0], no dones, last_value = 4.
    //    delta_t = r_t + 0.9 * V(s_{t+1}) - V(s_t)
    //    delta_0 = 0 + 0.9*2 - 1 = 0.8
    //    delta_1 = 0 + 0.9*3 - 2 = 0.7
    //    delta_2 = 0 + 0.9*4 - 3 = 0.6
    //    With lambda=0, adv_t == delta_t.
    {
        RolloutBuffer buf(3, 1, 1);
        float o = 0, a = 0;
        buf.add(&o, &a, 0, 0.0f, false, 1.0f);
        buf.add(&o, &a, 0, 0.0f, false, 2.0f);
        buf.add(&o, &a, 0, 0.0f, false, 3.0f);
        buf.computeGAE(/*last_value*/ 4.0f, /*gamma*/ 0.9f, /*lambda*/ 0.0f);
        CHECK(approx(buf.advantages()[0], 0.8f, 1e-4f));
        CHECK(approx(buf.advantages()[1], 0.7f, 1e-4f));
        CHECK(approx(buf.advantages()[2], 0.6f, 1e-4f));
        std::printf("test 4 OK  one-step TD  adv=[%.3f %.3f %.3f]\n",
                    buf.advantages()[0], buf.advantages()[1],
                    buf.advantages()[2]);
    }

    // 5) reset() drops stored data.
    {
        RolloutBuffer buf(2, 1, 1);
        float o = 0, a = 0;
        buf.add(&o, &a, 0, 1.0f, false, 0.0f);
        buf.add(&o, &a, 0, 1.0f, false, 0.0f);
        CHECK(buf.full());
        buf.reset();
        CHECK(buf.size() == 0);
        CHECK(!buf.full());
        buf.add(&o, &a, 0, 2.0f, false, 0.0f);
        CHECK(buf.rewards()[0] == 2.0f);
        std::printf("test 5 OK  reset\n");
    }

    if (g_failures)
    {
        std::fprintf(stderr, "\n%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("\nall tests passed\n");
    return 0;
}
