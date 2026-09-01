#include "env.hpp"

#include <cmath>

namespace
{
float
clamp01(float v)
{
    if (v < -1.0f)
        return -1.0f;
    if (v > 1.0f)
        return 1.0f;
    return v;
}
} // namespace

BallPlateEnv::BallPlateEnv(Config cfg) : m_cfg(cfg)
{
}

bool
BallPlateEnv::ballFellOff(float px, float pz, float halfW, float halfD)
{
    return std::fabs(px) > halfW || std::fabs(pz) > halfD;
}

Observation
BallPlateEnv::makeObs() const
{
    return Observation{m_ball.pos.x, m_ball.pos.y, m_ball.vel.x, m_ball.vel.y,
                       m_tiltX,      m_tiltZ};
}

Observation
BallPlateEnv::reset(uint64_t seed)
{
    m_rng.seed(seed);
    std::uniform_real_distribution<float> u(-m_cfg.initHalfRange,
                                            m_cfg.initHalfRange);
    m_ball.pos.x = u(m_rng);
    m_ball.pos.y = u(m_rng);
    m_ball.vel   = {0.0f, 0.0f};
    m_tiltX      = 0.0f;
    m_tiltZ      = 0.0f;
    m_step       = 0;
    m_done       = false;
    m_obs        = makeObs();
    return m_obs;
}

StepResult
BallPlateEnv::step(float actionX, float actionZ)
{
    float ax = clamp01(actionX);
    float az = clamp01(actionZ);
    m_tiltX  = ax * m_cfg.maxTilt;
    m_tiltZ  = az * m_cfg.maxTilt;

    m_ball = stepBallPhysics(m_ball, m_tiltX, m_tiltZ, m_cfg.dt, m_cfg.muK,
                             /*damping*/ 0.0f);
    ++m_step;

    float halfW = m_cfg.plateWidth * 0.5f;
    float halfD = m_cfg.plateDepth * 0.5f;
    bool fell   = ballFellOff(m_ball.pos.x, m_ball.pos.y, halfW, halfD);

    // Reward: position penalty dominates; small vel/action penalties
    // discourage thrashing; heavy terminal penalty for falling.
    float posSq = m_ball.pos.x * m_ball.pos.x + m_ball.pos.y * m_ball.pos.y;
    float velSq = m_ball.vel.x * m_ball.vel.x + m_ball.vel.y * m_ball.vel.y;
    float actSq = ax * ax + az * az;
    float reward = -posSq - 0.01f * velSq - 0.001f * actSq;
    if (fell)
        reward -= 100.0f;

    m_done = fell || m_step >= m_cfg.maxSteps;
    m_obs  = makeObs();
    return StepResult{m_obs, reward, m_done};
}
