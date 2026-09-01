#pragma once

#include "policy.hpp"
#include "training.hpp"

#include <memory>

// Hyperparameters for one PPO — Proximal Policy Optimization — update pass.
// Defaults are the widely-used values from the original PPO paper /
// Spinning Up implementation; they work well as a starting point for
// continuous-control tasks like ball-on-plate.
struct PPOConfig
{
    float clip_epsilon = 0.2f;   // policy-ratio clip range
    float value_coef   = 0.5f;   // weight of value-function loss in total
    float entropy_coef = 0.0f;   // weight of entropy bonus (0 = disabled).
                                 // For state-independent log_std,
                                 // adding entropy just pushes log_std up.
    float max_grad_norm = 0.5f;  // global L2 grad clip
    int epochs          = 10;    // passes over the rollout per update
    int minibatch_size  = 64;    // SGD minibatch size
    float target_kl     = 0.015f;// early-stop if approx KL exceeds 1.5x this
    bool clip_value     = true;  // enable value-function clipping
    float lr            = 3e-4f;
};

// Aggregated diagnostics from one update() call. All values averaged over
// the minibatches that actually ran (early-stopping may cut short).
struct PPOStats
{
    float policy_loss;
    float value_loss;
    float entropy;
    float approx_kl;      // rough KL divergence between old and new policy
    float clip_fraction;  // fraction of samples whose ratio hit the clip
    int epochs_run;       // may be < config.epochs due to KL early stop
};

// Owns actor + critic + optimizer, runs PPO updates on rollout batches.
class PPOTrainer
{
public:
    PPOTrainer(Actor actor, Critic critic, PPOConfig cfg = {});

    // Run one PPO update over the rollout batch. Advantages are assumed
    // already normalized (toTensorBatch does this when asked).
    PPOStats update(const RolloutBatch &batch);

    Actor actor() { return m_actor; }
    Critic critic() { return m_critic; }
    const PPOConfig &config() const { return m_cfg; }

private:
    Actor m_actor;
    Critic m_critic;
    PPOConfig m_cfg;
    std::unique_ptr<torch::optim::Adam> m_optimizer;
    std::vector<torch::Tensor> m_allParams;
};
