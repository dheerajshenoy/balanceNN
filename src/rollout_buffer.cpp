#include "rollout_buffer.hpp"

#include <cassert>
#include <cstring>

RolloutBuffer::RolloutBuffer(int capacity, int obs_dim, int act_dim)
    : m_capacity(capacity), m_obsDim(obs_dim), m_actDim(act_dim),
      m_obs(capacity * obs_dim), m_actions(capacity * act_dim),
      m_logProbs(capacity), m_rewards(capacity), m_dones(capacity),
      m_values(capacity), m_advantages(capacity), m_returns(capacity)
{
}

void
RolloutBuffer::add(const float *obs, const float *action, float log_prob,
                   float reward, bool done, float value)
{
    assert(m_size < m_capacity);
    std::memcpy(&m_obs[m_size * m_obsDim], obs,
                sizeof(float) * m_obsDim);
    std::memcpy(&m_actions[m_size * m_actDim], action,
                sizeof(float) * m_actDim);
    m_logProbs[m_size] = log_prob;
    m_rewards[m_size]  = reward;
    m_dones[m_size]    = done ? 1 : 0;
    m_values[m_size]   = value;
    ++m_size;
}

void
RolloutBuffer::computeGAE(float last_value, float gamma, float lambda)
{
    // Iterate backward. For step t:
    //   next_v   = V(s_{t+1})   (last_value if t is the final stored step)
    //   nonterm  = (1 - done_t) (0 if the episode ended here)
    //   delta    = r_t + gamma * next_v * nonterm - V(s_t)
    //   adv_t    = delta + gamma * lambda * nonterm * adv_{t+1}
    //   return_t = adv_t + V(s_t)
    float running_adv = 0.0f;
    for (int t = m_size - 1; t >= 0; --t)
    {
        float next_v  = (t == m_size - 1) ? last_value : m_values[t + 1];
        float nonterm = m_dones[t] ? 0.0f : 1.0f;
        float delta   = m_rewards[t] + gamma * next_v * nonterm - m_values[t];
        running_adv   = delta + gamma * lambda * nonterm * running_adv;
        m_advantages[t] = running_adv;
        m_returns[t]    = running_adv + m_values[t];
    }
}
