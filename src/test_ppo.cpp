// PPO update sanity test. Uses libtorch. Builds a tiny synthetic rollout
// where every step: obs=[1], action=[+1], reward=1, value=0, return=1,
// advantage=+1. After many update passes we expect:
//   - the actor's mean(obs=[1]) to shift toward +1 (positive advantage
//     reinforces the taken action)
//   - the critic's value(obs=[1]) to approach the return (1.0)
//   - entropy to decrease (policy becomes more deterministic)
//   - no NaN / inf anywhere
//
// This isn't a full RL learning test — it just proves the update math and
// autograd wiring are sound.

#include "policy.hpp"
#include "ppo.hpp"
#include "rollout_buffer.hpp"
#include "training.hpp"

#include <cmath>
#include <cstdio>

int
main()
{
    torch::manual_seed(0);

    constexpr int T       = 128;
    constexpr int obs_dim = 1;
    constexpr int act_dim = 1;

    // Actor sampling under init_log_std=-0.5 gives std ~= 0.6, so actions
    // in [-1, 1] are plausible without heavy clipping.
    Actor actor(obs_dim, act_dim, /*hidden*/ 32, /*init_log_std*/ -0.5);
    Critic critic(obs_dim, /*hidden*/ 32);

    // Fill a synthetic rollout: every step has obs=1, action=+1, reward=1,
    // done=false, value=0. Log-prob under the current (initial) policy.
    RolloutBuffer buf(T, obs_dim, act_dim);
    {
        torch::NoGradGuard nograd;
        auto obs = torch::ones({T, obs_dim});
        auto act = torch::ones({T, act_dim});
        auto [logp, _ent] = actor->evaluate(obs, act);
        auto logp_a       = logp.contiguous();

        float obs_row[obs_dim] = {1.0f};
        float act_row[act_dim] = {1.0f};
        for (int i = 0; i < T; ++i)
        {
            float lp = logp_a[i].item<float>();
            buf.add(obs_row, act_row, lp, /*rew*/ 1.0f, /*done*/ false,
                    /*value*/ 0.0f);
        }
        // Bootstrap value = 0 too — makes GAE math trivial.
        buf.computeGAE(/*last_value*/ 0.0f, /*gamma*/ 0.0f, /*lambda*/ 0.0f);
        // With gamma=0: delta_t = r_t - V(s_t) = 1 - 0 = 1, so adv=1, ret=1.
    }
    auto batch = toTensorBatch(buf, /*normalize_advantages*/ false);

    // Confirm the synthetic batch is what we think it is.
    if (std::fabs(batch.advantages.mean().item<float>() - 1.0f) > 1e-4)
    {
        std::fprintf(stderr, "FAIL: synthetic advantages not ~1.0\n");
        return 1;
    }

    PPOConfig cfg;
    cfg.epochs         = 4;
    cfg.minibatch_size = 32;
    cfg.target_kl      = 1.0f; // effectively disable early stop for the test
    cfg.entropy_coef   = 0.0f;
    PPOTrainer ppo(actor, critic, cfg);

    auto obs1        = torch::ones({1, obs_dim});
    float mu_before  = actor->mean(obs1).item<float>();
    float v_before   = critic->forward(obs1).item<float>();
    float ent_before = 0;
    {
        torch::NoGradGuard nograd;
        auto [_lp, ent] = actor->evaluate(obs1, torch::ones({1, act_dim}));
        ent_before      = ent.mean().item<float>();
    }
    std::printf("before: mu=%.4f  V=%.4f  entropy=%.4f\n", mu_before,
                v_before, ent_before);

    for (int upd = 0; upd < 40; ++upd)
    {
        // Recompute log-probs under the CURRENT policy at each update so the
        // "old" data isn't stale — this is what a real PPO loop would do by
        // re-collecting a fresh rollout. Here we cheat by re-evaluating.
        {
            torch::NoGradGuard nograd;
            auto lp
                = std::get<0>(actor->evaluate(batch.obs, batch.actions))
                      .contiguous();
            batch.log_probs = lp;
            batch.values    = critic->forward(batch.obs).detach();
            // Advantage = return - value (single-step, gamma=0).
            batch.advantages = batch.returns - batch.values;
        }
        auto s = ppo.update(batch);
        if (upd % 10 == 0)
            std::printf("upd %2d  pl=%+.4f  vl=%.4f  ent=%.4f  kl=%.5f "
                        " clip=%.3f\n",
                        upd, s.policy_loss, s.value_loss, s.entropy,
                        s.approx_kl, s.clip_fraction);
        if (!std::isfinite(s.policy_loss) || !std::isfinite(s.value_loss))
        {
            std::fprintf(stderr, "FAIL: non-finite loss at update %d\n", upd);
            return 1;
        }
    }

    float mu_after  = actor->mean(obs1).item<float>();
    float v_after   = critic->forward(obs1).item<float>();
    float ent_after = 0;
    {
        torch::NoGradGuard nograd;
        auto [_lp, ent] = actor->evaluate(obs1, torch::ones({1, act_dim}));
        ent_after       = ent.mean().item<float>();
    }
    std::printf("after:  mu=%.4f  V=%.4f  entropy=%.4f\n", mu_after, v_after,
                ent_after);

    int failures = 0;
    auto check   = [&](bool cond, const char *msg) {
        if (!cond)
        {
            std::fprintf(stderr, "FAIL: %s\n", msg);
            ++failures;
        }
    };

    check(mu_after > mu_before + 0.1f,
          "actor mean should have shifted upward toward the reinforced "
          "action (+1)");
    check(std::fabs(v_after - 1.0f) < std::fabs(v_before - 1.0f) - 0.1f,
          "critic value should have moved substantially toward return=1");
    check(ent_after < ent_before,
          "entropy should have decreased (policy more deterministic)");

    if (failures)
    {
        std::fprintf(stderr, "\n%d failure(s)\n", failures);
        return 1;
    }
    std::printf("\nall tests passed\n");
    return 0;
}
