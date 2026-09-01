#pragma once

#include "plate.hpp"
#include "render_common.hpp"

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
// world X / Z (radians). The returned state is `s` advanced by one dt.
// Does no boundary handling.
BallState stepBallPhysics(BallState s, float tiltX, float tiltZ, float dt,
                          float muK = 0.3f, float damping = 0.0f,
                          float gravity = 9.81f);

// A sphere that rolls on a tilted plate. State (position, velocity,
// acceleration) is in the plate's local XZ (Vec2::x → plate X,
// Vec2::y → plate Z).
class Sphere
{
public:
    Sphere(float radius = 0.2f, int stacks = 32, int slices = 48);
    ~Sphere();

    Sphere(const Sphere &)            = delete;
    Sphere &operator=(const Sphere &) = delete;

    float radius() const { return m_radius; }

    Vec2 position() const { return m_position; }
    Vec2 velocity() const { return m_velocity; }
    Vec2 acceleration() const { return m_acceleration; }

    void setPosition(float x, float z) { m_position = {x, z}; }
    void setVelocity(float vx, float vz) { m_velocity = {vx, vz}; }

    bool hasFallen() const { return m_fallen; }
    void reset(float x = 0.0f, float z = 0.0f);

    // Integrate one step (semi-implicit Euler):
    //   ax = g * sin(tiltZ), ay = g * sin(tiltX)
    //   v += a*dt;  v *= (1 - damping*dt);  p += v*dt
    // Sets hasFallen() when |p.x| > plate_half_width or |p.y| > plate_half_depth
    // (measured to sphere center — the "ball off the plate" episode-end signal).
    // A subsequent call is a no-op until reset() is invoked.
    void update(float dt, const Plate &plate);

    // Draw at (position.x, plateTopY + radius, position.y) in plate-local
    // coords, then transformed by `parent` (e.g., plate tilt + scene rot).
    void draw(const PhongProgram &prog, const Mat4 &parent,
              float plateTopY) const;

private:
    float m_radius;
    Mesh m_mesh;

    Vec2 m_position     = {0.0f, 0.0f};
    Vec2 m_velocity     = {0.0f, 0.0f};
    Vec2 m_acceleration = {0.0f, 0.0f};
    bool m_fallen       = false;
};
