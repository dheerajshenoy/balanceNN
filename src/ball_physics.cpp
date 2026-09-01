#include "ball_physics.hpp"

#include <cmath>

BallState
stepBallPhysics(BallState s, float tiltX, float tiltZ, float dt, float muK,
                float damping, float gravity)
{
    // ax comes from tilt about Z; ay from tilt about X (Vec2::y is plate-Z).
    float ax = gravity * std::sin(tiltZ);
    float ay = gravity * std::sin(tiltX);

    s.vel.x += ax * dt;
    s.vel.y += ay * dt;

    // Kinetic friction: constant decel of muK*g opposite to velocity, clamped
    // so we don't reverse the direction — the ball actually stops.
    float speed = std::sqrt(s.vel.x * s.vel.x + s.vel.y * s.vel.y);
    if (speed > 0.0f)
    {
        float dv = muK * gravity * dt;
        if (dv >= speed)
        {
            s.vel.x = 0.0f;
            s.vel.y = 0.0f;
        }
        else
        {
            float k = (speed - dv) / speed;
            s.vel.x *= k;
            s.vel.y *= k;
        }
    }

    if (damping > 0.0f)
    {
        float d = 1.0f - damping * dt;
        if (d < 0.0f)
            d = 0.0f;
        s.vel.x *= d;
        s.vel.y *= d;
    }

    s.pos.x += s.vel.x * dt;
    s.pos.y += s.vel.y * dt;
    return s;
}
