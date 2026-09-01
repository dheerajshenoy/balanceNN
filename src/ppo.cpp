#include "ppo.hpp"

#include <algorithm>

PPOTrainer::PPOTrainer(Actor actor, Critic critic, PPOConfig cfg)
    : m_actor(std::move(actor)), m_critic(std::move(critic)), m_cfg(cfg)
{
    // Single optimizer over ALL learnable params (actor trunk + mu + log_std,
    // critic trunk + value head). Keeping them in one Adam instance is the
    // standard PPO setup and lets us grad-clip everything at once.
    for (auto &p : m_actor->parameters())
        m_allParams.push_back(p);
    for (auto &p : m_critic->parameters())
        m_allParams.push_back(p);
    m_optimizer = std::make_unique<torch::optim::Adam>(
        m_allParams, torch::optim::AdamOptions(m_cfg.lr));
}

PPOStats
PPOTrainer::update(const RolloutBatch &batch)
{
    const int T = static_cast<int>(batch.obs.size(0));
    const int mb = std::min(m_cfg.minibatch_size, T);
    const float eps = m_cfg.clip_epsilon;

    double sum_pl = 0, sum_vl = 0, sum_ent = 0, sum_kl = 0, sum_cf = 0;
    int mb_count = 0;
    int epochs_run = 0;

    for (int epoch = 0; epoch < m_cfg.epochs; ++epoch)
    {
        ++epochs_run;
        auto perm = torch::randperm(T, torch::TensorOptions().dtype(torch::kLong));

        double epoch_kl = 0;
        int epoch_mbs   = 0;

        for (int start = 0; start < T; start += mb)
        {
            int end     = std::min(start + mb, T);
            auto idx    = perm.slice(0, start, end);

            auto obs      = batch.obs.index_select(0, idx);
            auto act      = batch.actions.index_select(0, idx);
            auto old_logp = batch.log_probs.index_select(0, idx);
            auto adv      = batch.advantages.index_select(0, idx);
            auto ret      = batch.returns.index_select(0, idx);
            auto old_val  = batch.values.index_select(0, idx);

            // --- Policy loss (clipped surrogate).
            auto [new_logp, entropy_t] = m_actor->evaluate(obs, act);
            auto ratio = (new_logp - old_logp).exp();
            auto surr1 = ratio * adv;
            auto surr2 = torch::clamp(ratio, 1.0f - eps, 1.0f + eps) * adv;
            auto policy_loss = -torch::min(surr1, surr2).mean();

            // --- Value loss (optionally clipped so critic doesn't move too
            // far per update either — same idea as the policy clip).
            auto new_val = m_critic->forward(obs);
            torch::Tensor value_loss;
            if (m_cfg.clip_value)
            {
                auto v_clipped
                    = old_val + (new_val - old_val).clamp(-eps, eps);
                auto vl1   = (new_val - ret).pow(2);
                auto vl2   = (v_clipped - ret).pow(2);
                value_loss = 0.5f * torch::max(vl1, vl2).mean();
            }
            else
            {
                value_loss = 0.5f * (new_val - ret).pow(2).mean();
            }

            auto entropy    = entropy_t.mean();
            auto total_loss = policy_loss + m_cfg.value_coef * value_loss
                              - m_cfg.entropy_coef * entropy;

            m_optimizer->zero_grad();
            total_loss.backward();
            torch::nn::utils::clip_grad_norm_(m_allParams,
                                              m_cfg.max_grad_norm);
            m_optimizer->step();

            // --- Diagnostics.
            // Approximate KL, form from Schulman: mean((ratio - 1) - log(ratio))
            // — always non-negative, more stable than mean(old - new).
            auto approx_kl = ((ratio - 1.0f) - (new_logp - old_logp)).mean();
            auto clip_frac
                = ((ratio - 1.0f).abs() > eps).to(torch::kFloat32).mean();

            sum_pl += policy_loss.item<float>();
            sum_vl += value_loss.item<float>();
            sum_ent += entropy.item<float>();
            sum_kl += approx_kl.item<float>();
            sum_cf += clip_frac.item<float>();
            ++mb_count;
            epoch_kl += approx_kl.item<float>();
            ++epoch_mbs;
        }

        // KL early stop: if this epoch's average KL blew past target, we've
        // moved too far from the old policy; skip remaining epochs.
        if (m_cfg.target_kl > 0.0f && epoch_mbs > 0
            && (epoch_kl / epoch_mbs) > 1.5f * m_cfg.target_kl)
            break;
    }

    PPOStats stats;
    if (mb_count == 0)
    {
        stats = {0, 0, 0, 0, 0, 0};
    }
    else
    {
        stats.policy_loss   = static_cast<float>(sum_pl / mb_count);
        stats.value_loss    = static_cast<float>(sum_vl / mb_count);
        stats.entropy       = static_cast<float>(sum_ent / mb_count);
        stats.approx_kl     = static_cast<float>(sum_kl / mb_count);
        stats.clip_fraction = static_cast<float>(sum_cf / mb_count);
        stats.epochs_run    = epochs_run;
    }
    return stats;
}
