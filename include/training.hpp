#pragma once

#include "rollout_buffer.hpp"

#include <torch/torch.h>

// Batched tensors for a PPO update pass. All tensors are on CPU, float32.
// Shapes:
//   obs        [T, obs_dim]
//   actions    [T, act_dim]
//   log_probs  [T]
//   advantages [T]
//   returns    [T]
//   values     [T]  (old value estimates — kept for optional value clipping)
struct RolloutBatch
{
    torch::Tensor obs;
    torch::Tensor actions;
    torch::Tensor log_probs;
    torch::Tensor advantages;
    torch::Tensor returns;
    torch::Tensor values;
};

// Copy the buffer's populated rows into tensors. If normalize_advantages is
// true, applies the standard PPO trick: adv = (adv - mean) / (std + 1e-8),
// which typically improves gradient signal quality.
RolloutBatch toTensorBatch(const RolloutBuffer &buf,
                           bool normalize_advantages = true);
