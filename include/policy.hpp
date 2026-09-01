#pragma once

#include <torch/torch.h>

// Actor network: maps an observation to a diagonal-Gaussian probability
// distribution over continuous actions.
//
// - Two hidden layers of 64 units with tanh activation (standard PPO shape).
// - Output head produces the per-dim MEAN of the action distribution.
// - A separate learnable parameter holds the LOG standard deviation of the
//   distribution — one value per action dim, independent of the observation.
//   This "state-independent log-std" is standard for PPO on continuous
//   control; a state-dependent head sometimes helps but usually isn't
//   worth the extra complexity here.
//
// Sampling: action = mean + std * eps, where eps ~ Normal(0, 1). The final
// action is clipped to [-1, 1] by the environment, so the raw sample can
// legally exceed that.
struct ActorImpl : torch::nn::Module
{
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear mu{nullptr};
    torch::Tensor log_std;

    int64_t act_dim = 0;

    ActorImpl(int64_t obs_dim, int64_t act_dim, int64_t hidden = 64,
              double init_log_std = -0.5);

    // Returns the deterministic action mean (no sampling). Use for eval.
    torch::Tensor mean(const torch::Tensor &obs);

    // Sample an action from the current policy. Returns
    // (action, log_prob) where log_prob is summed across action dims.
    // Both tensors have batch dim matching `obs`.
    std::pair<torch::Tensor, torch::Tensor> sample(const torch::Tensor &obs);

    // Given a previously-sampled action, compute (log_prob, entropy) under
    // the CURRENT policy. Needed for the PPO update step.
    std::pair<torch::Tensor, torch::Tensor> evaluate(const torch::Tensor &obs,
                                                     const torch::Tensor &act);

private:
    // Shared hidden trunk.
    torch::Tensor trunk(const torch::Tensor &obs);
};
TORCH_MODULE(Actor);

// Critic network: maps an observation to a scalar value estimate
// (expected discounted return from that state under the current policy).
// Same trunk shape as the actor but with a scalar output.
struct CriticImpl : torch::nn::Module
{
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear v{nullptr};

    CriticImpl(int64_t obs_dim, int64_t hidden = 64);

    // Returns shape [batch] (value squeezed from the trailing dim of 1).
    torch::Tensor forward(const torch::Tensor &obs);
};
TORCH_MODULE(Critic);
