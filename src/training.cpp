#include "training.hpp"

RolloutBatch
toTensorBatch(const RolloutBuffer &buf, bool normalize_advantages)
{
    int T = buf.size();
    int O = buf.obsDim();
    int A = buf.actDim();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);

    // torch::from_blob does NOT copy — we .clone() so the tensors own their
    // memory and outlive the caller's buffer.
    RolloutBatch b;
    b.obs = torch::from_blob(const_cast<float *>(buf.obs().data()), {T, O},
                             opts).clone();
    b.actions
        = torch::from_blob(const_cast<float *>(buf.actions().data()), {T, A},
                           opts).clone();
    b.log_probs
        = torch::from_blob(const_cast<float *>(buf.logProbs().data()), {T},
                           opts).clone();
    b.advantages
        = torch::from_blob(const_cast<float *>(buf.advantages().data()), {T},
                           opts).clone();
    b.returns
        = torch::from_blob(const_cast<float *>(buf.returns().data()), {T},
                           opts).clone();
    b.values
        = torch::from_blob(const_cast<float *>(buf.values().data()), {T}, opts)
              .clone();

    if (normalize_advantages && T > 1)
    {
        auto mean = b.advantages.mean();
        auto std  = b.advantages.std(/*unbiased=*/true);
        b.advantages = (b.advantages - mean) / (std + 1e-8);
    }
    return b;
}
