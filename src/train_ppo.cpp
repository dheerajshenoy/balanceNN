// PPO training loop for BallPlateEnv.
//
// One binary, one main(): collect rollouts, run PPO updates, log episode
// reward, and save the trained actor+critic to disk. Also computes the PD
// controller's mean reward once at the start so you can see whether the
// trained policy is catching up to (or beating) the hand-crafted baseline.
//
// Not a ctest — training takes a couple of minutes.

#include "env.hpp"
#include "obs_normalizer.hpp"
#include "pd_controller.hpp"
#include "policy.hpp"
#include "ppo.hpp"
#include "rollout_buffer.hpp"
#include "training.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace
{
constexpr int   kObsDim         = 6;
constexpr int   kActDim         = 2;
constexpr int   kRolloutLen     = 2048;
constexpr int   kNumUpdates     = 100;      // ~200k env steps total
constexpr int   kLogEvery       = 1;
constexpr int   kPDEvalEpisodes = 50;
constexpr int   kEvalEpisodes   = 20;       // final deterministic eval
constexpr float kGamma          = 0.99f;
constexpr float kLambda         = 0.95f;
constexpr uint64_t kSeedBase    = 12345;

// Pack an Observation into a plain float array in the order the network
// consumes.
void
obsToArray(const Observation &o, float *out)
{
    out[0] = o.px;
    out[1] = o.pz;
    out[2] = o.vx;
    out[3] = o.vz;
    out[4] = o.tiltX;
    out[5] = o.tiltZ;
}

// Run `episodes` full episodes under a fixed action policy `pick_action`.
// Returns mean total episode reward.
template <class F>
float
evaluate(BallPlateEnv &env, ObsNormalizer *maybe_norm, F pick_action,
         int episodes, uint64_t seed_start)
{
    double sum = 0.0;
    for (int ep = 0; ep < episodes; ++ep)
    {
        Observation obs = env.reset(seed_start + ep);
        float ep_reward = 0.0f;
        while (!env.done())
        {
            float raw[kObsDim];
            obsToArray(obs, raw);
            if (maybe_norm)
                maybe_norm->normalize(raw);
            auto [ax, az] = pick_action(raw, obs);
            auto r        = env.step(ax, az);
            ep_reward += r.reward;
            obs = r.obs;
        }
        sum += ep_reward;
    }
    return static_cast<float>(sum / episodes);
}

struct EpisodeStats
{
    std::vector<float> rewards;
    std::vector<int> lengths;
    float current_reward = 0.0f;
    int current_length   = 0;

    void step(float reward)
    {
        current_reward += reward;
        ++current_length;
    }
    void finishEpisode()
    {
        rewards.push_back(current_reward);
        lengths.push_back(current_length);
        current_reward = 0.0f;
        current_length = 0;
    }
    float meanReward() const
    {
        if (rewards.empty())
            return 0.0f;
        double s = 0;
        for (float r : rewards)
            s += r;
        return static_cast<float>(s / rewards.size());
    }
    float meanLength() const
    {
        if (lengths.empty())
            return 0.0f;
        double s = 0;
        for (int l : lengths)
            s += l;
        return static_cast<float>(s / lengths.size());
    }
    void clear()
    {
        rewards.clear();
        lengths.clear();
    }
};
} // namespace

int
main(int argc, char **argv)
{
    torch::manual_seed(0);

    std::string save_path = "trained_policy.pt";
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc)
            save_path = argv[++i];
    }

    // -------- Env, networks, trainer, buffer, normalizer.
    BallPlateEnv env; // default Config
    Actor actor(kObsDim, kActDim, /*hidden*/ 64, /*init_log_std*/ -0.5);
    Critic critic(kObsDim, /*hidden*/ 64);
    PPOConfig ppo_cfg;
    // A little entropy encouragement helps early exploration in continuous
    // control even with state-independent log_std.
    ppo_cfg.entropy_coef   = 0.001f;
    ppo_cfg.minibatch_size = 64;
    ppo_cfg.epochs         = 10;
    ppo_cfg.target_kl      = 0.02f;
    PPOTrainer trainer(actor, critic, ppo_cfg);
    RolloutBuffer buffer(kRolloutLen, kObsDim, kActDim);
    ObsNormalizer normalizer(kObsDim);

    // -------- PD baseline for reference.
    {
        BallPlateEnv pd_env;
        PDController pd;
        float pd_reward = evaluate(
            pd_env, /*normalizer*/ nullptr,
            [&](const float *, const Observation &o) {
                auto a = pd.compute(o);
                return std::pair<float, float>{a.x, a.z};
            },
            kPDEvalEpisodes, kSeedBase);
        std::printf("PD baseline: mean reward over %d episodes = %+.2f\n\n",
                    kPDEvalEpisodes, pd_reward);
    }

    // -------- Training loop.
    Observation obs = env.reset(kSeedBase);
    EpisodeStats ep;
    uint64_t reset_seed = kSeedBase + 1;
    auto t_start        = std::chrono::steady_clock::now();
    long long env_steps = 0;

    for (int upd = 0; upd < kNumUpdates; ++upd)
    {
        buffer.reset();

        for (int t = 0; t < kRolloutLen; ++t)
        {
            float raw[kObsDim];
            obsToArray(obs, raw);
            normalizer.observe(raw);
            float norm[kObsDim];
            std::memcpy(norm, raw, sizeof(norm));
            normalizer.normalize(norm);

            float action[kActDim];
            float log_prob;
            float value;
            {
                torch::NoGradGuard nograd;
                auto obs_t = torch::from_blob(norm, {1, kObsDim},
                                              torch::TensorOptions().dtype(
                                                  torch::kFloat32))
                                 .clone();
                auto [a_t, lp_t] = actor->sample(obs_t);
                action[0]        = a_t[0][0].item<float>();
                action[1]        = a_t[0][1].item<float>();
                log_prob         = lp_t[0].item<float>();
                value            = critic->forward(obs_t)[0].item<float>();
            }

            auto r = env.step(action[0], action[1]);
            ++env_steps;
            buffer.add(norm, action, log_prob, r.reward, r.done, value);

            ep.step(r.reward);
            if (r.done)
            {
                ep.finishEpisode();
                obs = env.reset(reset_seed++);
            }
            else
            {
                obs = r.obs;
            }
        }

        // Bootstrap: value of the state AFTER the last stored transition.
        // If the last transition was terminal, GAE zeroes this out anyway
        // (via (1 - done)), so any value is correct then; otherwise it must
        // be the critic's estimate at the current obs.
        float last_value = 0.0f;
        {
            torch::NoGradGuard nograd;
            float raw[kObsDim];
            obsToArray(obs, raw);
            float norm[kObsDim];
            std::memcpy(norm, raw, sizeof(norm));
            normalizer.normalize(norm);
            auto obs_t = torch::from_blob(norm, {1, kObsDim},
                                          torch::TensorOptions().dtype(
                                              torch::kFloat32))
                             .clone();
            last_value = critic->forward(obs_t)[0].item<float>();
        }
        buffer.computeGAE(last_value, kGamma, kLambda);
        auto batch = toTensorBatch(buffer, /*normalize_advantages*/ true);
        auto stats = trainer.update(batch);

        if ((upd + 1) % kLogEvery == 0)
        {
            float mean_r = ep.meanReward();
            float mean_l = ep.meanLength();
            int n_eps    = static_cast<int>(ep.rewards.size());
            auto now     = std::chrono::steady_clock::now();
            float dt_s   = std::chrono::duration<float>(now - t_start).count();
            std::printf(
                "upd %3d/%d  steps=%lld  eps=%d  meanR=%+.2f  meanLen=%.0f "
                " pl=%+.4f  vl=%.4f  ent=%.3f  kl=%.4f  cf=%.3f  "
                " eps_run=%d  time=%.1fs\n",
                upd + 1, kNumUpdates, env_steps, n_eps, mean_r, mean_l,
                stats.policy_loss, stats.value_loss, stats.entropy,
                stats.approx_kl, stats.clip_fraction, stats.epochs_run,
                dt_s);
            ep.clear();
        }
    }

    // -------- Deterministic eval of the trained policy.
    {
        BallPlateEnv eval_env;
        // Freeze normalization stats — no updates during eval.
        float det_reward = evaluate(
            eval_env, &normalizer,
            [&](const float *norm_obs, const Observation &) {
                torch::NoGradGuard nograd;
                auto obs_t
                    = torch::from_blob(const_cast<float *>(norm_obs),
                                       {1, kObsDim},
                                       torch::TensorOptions().dtype(
                                           torch::kFloat32))
                          .clone();
                auto a = actor->mean(obs_t);
                return std::pair<float, float>{a[0][0].item<float>(),
                                               a[0][1].item<float>()};
            },
            kEvalEpisodes, kSeedBase + 9999);
        std::printf(
            "\nTrained (deterministic) eval: mean reward over %d episodes "
            "= %+.2f\n",
            kEvalEpisodes, det_reward);
    }

    // -------- Save weights + normalizer stats. The runtime visualizer
    // needs the normalizer to feed the network inputs in the same units
    // the trained policy expects.
    torch::save(actor, save_path + ".actor");
    torch::save(critic, save_path + ".critic");
    normalizer.save(save_path + ".norm");
    std::printf("saved %s.{actor,critic,norm}\n", save_path.c_str());
    return 0;
}
