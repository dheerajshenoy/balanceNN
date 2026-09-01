#pragma once

#include "ball_physics.hpp"

#include <cstdint>
#include <random>

// Six-dimensional environment observation. Kept as a plain struct so the env
// stays testable without torch; convert to a tensor at the training boundary.
struct Observation
{
    float px, pz;       // ball position in plate-local XZ
    float vx, vz;       // ball velocity in plate-local XZ
    float tiltX, tiltZ; // current plate tilt about world X / Z (radians)
};

struct StepResult
{
    Observation obs;
    float reward;
    bool done;
};

struct BallPlateEnvConfig
{
    float plateWidth    = 4.0f;
    float plateDepth    = 4.0f;
    float maxTilt       = 0.6f; // action=1.0 → tilt=maxTilt (rad)
    float dt            = 1.0f / 60.0f;
    int maxSteps        = 600;  // 10 s at 60 Hz
    float muK           = 0.05f;
    float initHalfRange = 0.5f; // reset spawns ball in ±this box
};

// Ball-on-plate RL environment. Actions are 2D in [-1, 1], mapped to target
// tilts by multiplying by maxTilt. Physics uses the same stepBallPhysics
// integrator as the visualizer, so the trained policy transfers directly.
class BallPlateEnv
{
public:
    using Config = BallPlateEnvConfig;

    BallPlateEnv() = default;
    explicit BallPlateEnv(Config cfg);

    // Reset to a random initial position (deterministic given seed) with zero
    // velocity, zero tilt. Returns the initial observation.
    Observation reset(uint64_t seed);

    // Advance one dt. actionX/actionZ are clamped to [-1, 1] and become the
    // target tilt (no rate limit — tilt snaps). Returns next obs + reward +
    // done. Calling step() after done is a programming error; call reset().
    StepResult step(float actionX, float actionZ);

    const Observation &observe() const { return m_obs; }
    const Config &config() const { return m_cfg; }
    int stepCount() const { return m_step; }
    bool done() const { return m_done; }

    // Shared with the visualizer so the "fell off" definition stays in one
    // place.
    static bool ballFellOff(float px, float pz, float halfW, float halfD);

private:
    Observation makeObs() const;

    Config m_cfg;
    BallState m_ball{};
    float m_tiltX = 0.0f;
    float m_tiltZ = 0.0f;
    int m_step    = 0;
    bool m_done   = true; // must reset() before first step
    Observation m_obs{};
    std::mt19937 m_rng;
};
