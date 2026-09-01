#pragma once

#include <cstdint>
#include <vector>

// PPO rollout buffer. Stores one trajectory (or several concatenated) of
// (observation, action, log-probability, reward, done, value) tuples, then
// computes GAE (Generalized Advantage Estimation) advantages and returns
// once the rollout is complete.
//
// Pure C++ — no torch, no GL. Torch tensor conversion is a separate free
// function in training.hpp (kept out of this header so unit tests can link
// without libtorch).
//
// Terminology:
//   done   = the environment ended the episode at this step (fell off /
//            timed out). The next value bootstrap is zeroed for terminal
//            steps.
//   value  = the CRITIC's estimate at the time the observation was seen,
//            before the reward. Used both as V(s_t) in the delta term and
//            for value-function loss + optional value clipping in PPO.
//   last_value = the critic's estimate of the state AFTER the last stored
//                transition. If that transition was terminal, pass 0.
//                Otherwise, evaluate the critic on the next observation
//                and pass its scalar value.
class RolloutBuffer
{
public:
    RolloutBuffer(int capacity, int obs_dim, int act_dim);

    int capacity() const { return m_capacity; }
    int size() const { return m_size; }
    int obsDim() const { return m_obsDim; }
    int actDim() const { return m_actDim; }
    bool full() const { return m_size == m_capacity; }

    void reset() { m_size = 0; }

    // Append one step. Pointers must point to at least obsDim / actDim
    // floats respectively. Must not be called when full().
    void add(const float *obs, const float *action, float log_prob,
             float reward, bool done, float value);

    // Compute GAE-λ advantages and n-step returns for the stored data.
    // gamma  : reward discount factor (typical 0.99)
    // lambda : GAE bias-variance tradeoff (0 = one-step TD, 1 = Monte Carlo;
    //          typical 0.95)
    // Must be called after data is populated and before advantages()/returns()
    // are read.
    void computeGAE(float last_value, float gamma = 0.99f,
                    float lambda = 0.95f);

    // Raw storage (row-major, size() rows). Read-only.
    const std::vector<float> &obs() const { return m_obs; }
    const std::vector<float> &actions() const { return m_actions; }
    const std::vector<float> &logProbs() const { return m_logProbs; }
    const std::vector<float> &rewards() const { return m_rewards; }
    const std::vector<uint8_t> &dones() const { return m_dones; }
    const std::vector<float> &values() const { return m_values; }
    const std::vector<float> &advantages() const { return m_advantages; }
    const std::vector<float> &returns() const { return m_returns; }

private:
    int m_capacity;
    int m_obsDim;
    int m_actDim;
    int m_size = 0;

    std::vector<float> m_obs;        // capacity * obs_dim
    std::vector<float> m_actions;    // capacity * act_dim
    std::vector<float> m_logProbs;   // capacity
    std::vector<float> m_rewards;    // capacity
    std::vector<uint8_t> m_dones;    // capacity
    std::vector<float> m_values;     // capacity
    std::vector<float> m_advantages; // capacity (filled by computeGAE)
    std::vector<float> m_returns;    // capacity (filled by computeGAE)
};
