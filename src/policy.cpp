#include "policy.hpp"

namespace
{
constexpr double kLog2Pi = 1.8378770664093454835606594728112352797228; // ln(2π)
}

ActorImpl::ActorImpl(int64_t obs_dim, int64_t act_dim_, int64_t hidden,
                     double init_log_std)
    : act_dim(act_dim_)
{
    fc1 = register_module("fc1", torch::nn::Linear(obs_dim, hidden));
    fc2 = register_module("fc2", torch::nn::Linear(hidden, hidden));
    mu  = register_module("mu", torch::nn::Linear(hidden, act_dim));

    // Register log_std as a learnable parameter so it shows up in
    // parameters() and gets gradient updates.
    log_std = register_parameter(
        "log_std",
        torch::full({act_dim}, init_log_std, torch::TensorOptions().dtype(
                                                 torch::kFloat32)));
}

torch::Tensor
ActorImpl::trunk(const torch::Tensor &obs)
{
    auto h = torch::tanh(fc1->forward(obs));
    h      = torch::tanh(fc2->forward(h));
    return h;
}

torch::Tensor
ActorImpl::mean(const torch::Tensor &obs)
{
    return mu->forward(trunk(obs));
}

std::pair<torch::Tensor, torch::Tensor>
ActorImpl::sample(const torch::Tensor &obs)
{
    auto m   = mu->forward(trunk(obs));
    auto std = log_std.exp();
    auto eps = torch::randn_like(m);
    auto act = m + std * eps;

    // Diagonal Gaussian log-prob, summed across action dims:
    //   log N(x | m, s) = -0.5 * ((x-m)/s)^2 - log(s) - 0.5 * log(2π)
    auto diff  = (act - m) / std;
    auto logp  = -0.5 * diff.pow(2) - log_std - 0.5 * kLog2Pi;
    auto sum_logp = logp.sum(-1);
    return {act, sum_logp};
}

std::pair<torch::Tensor, torch::Tensor>
ActorImpl::evaluate(const torch::Tensor &obs, const torch::Tensor &act)
{
    auto m        = mu->forward(trunk(obs));
    auto std      = log_std.exp();
    auto diff     = (act - m) / std;
    auto logp     = -0.5 * diff.pow(2) - log_std - 0.5 * kLog2Pi;
    auto sum_logp = logp.sum(-1);
    // Entropy of a diagonal Gaussian: 0.5 * (1 + log(2π)) + log(std), summed.
    auto ent      = (0.5 * (1.0 + kLog2Pi) + log_std).sum();
    // Broadcast to batch so caller can .mean() cleanly.
    ent = ent.expand({obs.size(0)});
    return {sum_logp, ent};
}

CriticImpl::CriticImpl(int64_t obs_dim, int64_t hidden)
{
    fc1 = register_module("fc1", torch::nn::Linear(obs_dim, hidden));
    fc2 = register_module("fc2", torch::nn::Linear(hidden, hidden));
    v   = register_module("v", torch::nn::Linear(hidden, 1));
}

torch::Tensor
CriticImpl::forward(const torch::Tensor &obs)
{
    auto h = torch::tanh(fc1->forward(obs));
    h      = torch::tanh(fc2->forward(h));
    return v->forward(h).squeeze(-1);
}
