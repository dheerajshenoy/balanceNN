#pragma once

// Ball-on-plate physics core. Pure math — no GL, no SDL, no torch.
// Shared by the visualizer, the RL environment, and unit tests.

struct Vec2
{
    float x;
    float y;
};

struct BallState
{
    Vec2 pos;
    Vec2 vel;
};

// Standalone ball-on-plate integrator (semi-implicit Euler).
// Applies gravity projected onto the tilted surface, then kinetic friction
// (Coulomb: constant decel of muK*gravity opposite to velocity, clamped so
// the ball actually STOPS instead of asymptotically decaying), then
// optional linear viscous damping. tiltX / tiltZ are plate rotations about
// world X / Z (radians). Returns `s` advanced by one dt. No boundary
// handling — callers decide what "fell off" means.
BallState stepBallPhysics(BallState s, float tiltX, float tiltZ, float dt,
                          float muK = 0.3f, float damping = 0.0f,
                          float gravity = 9.81f);
