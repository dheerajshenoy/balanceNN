// Standalone smoke test for stepBallPhysics — no GL, no SDL.
// Two experiments:
//   1) Ball off-center, plate held tilted toward center. It should slide
//      toward the origin (position magnitude shrinks / crosses zero) and
//      stay bounded (no NaN, no runaway).
//   2) Same setup with damping ON vs damping OFF, so we can see that
//      oscillations decay instead of running forever.
//
// This deliberately doesn't include sphere.hpp because that pulls in GL.
// We prototype stepBallPhysics locally with the same formula; if you want
// to link against the real one instead, drop the local copy and add
// src/sphere.cpp — but sphere.cpp also drags in glad, so an isolated
// header for the physics function is cleaner. Left as a follow-up.

#include <cmath>
#include <cstdio>

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

// MUST stay in sync with stepBallPhysics in src/sphere.cpp.
BallState
stepBallPhysics(BallState s, float tiltX, float tiltZ, float dt,
                float damping = 0.5f, float gravity = 9.81f)
{
    float ax = gravity * std::sin(tiltZ);
    float ay = gravity * std::sin(tiltX);
    s.vel.x += ax * dt;
    s.vel.y += ay * dt;
    float d = 1.0f - damping * dt;
    if (d < 0.0f)
        d = 0.0f;
    s.vel.x *= d;
    s.vel.y *= d;
    s.pos.x += s.vel.x * dt;
    s.pos.y += s.vel.y * dt;
    return s;
}

int
main()
{
    constexpr float dt   = 1.0f / 120.0f; // 120 Hz sim
    constexpr int steps  = 600;           // 5 seconds
    constexpr int print_every = 30;       // ~4 samples/sec

    // P-controller: tilt the plate toward the ball's opposite side so
    // acceleration always pushes it back toward x=0. ax = g*sin(tiltZ), so
    // to push in -x when x>0 we need tiltZ < 0 when x > 0 → tiltZ = -k*x.
    constexpr float kP = 0.15f; // rad per unit position

    // ---- Experiment 1: P-controller, NO damping.
    // Expect: sustained oscillation about x=0 (energy conserved) —
    // confirms the sphere is "pulled toward center" but never settles.
    {
        std::printf("=== EXP 1: P-controller (k=%.2f), damping=0, "
                    "start x=+1.0\n",
                    kP);
        BallState s{{1.0f, 0.0f}, {0.0f, 0.0f}};
        int crossings   = 0;
        float last_sign = +1.0f;
        for (int i = 0; i <= steps; ++i)
        {
            if (i % print_every == 0)
                std::printf("t=%.2fs  x=%+.4f  vx=%+.4f\n", i * dt, s.pos.x,
                            s.vel.x);
            float tiltZ = -kP * s.pos.x;
            s = stepBallPhysics(s, /*tiltX*/ 0.0f, tiltZ, dt, /*damping*/ 0.0f);
            if (!std::isfinite(s.pos.x) || !std::isfinite(s.vel.x))
            {
                std::printf("FAIL: non-finite state at step %d\n", i);
                return 1;
            }
            float sign = s.pos.x >= 0.0f ? 1.0f : -1.0f;
            if (sign != last_sign)
            {
                ++crossings;
                last_sign = sign;
            }
        }
        std::printf("zero-crossings: %d (expect many — no damping)\n",
                    crossings);
    }

    // ---- Experiment 2: P-controller + damping. Amplitude should decay
    // and the ball should settle near x=0.
    {
        std::printf("\n=== EXP 2: P-controller (k=%.2f), damping=1.5, "
                    "start x=+1.0\n",
                    kP);
        BallState s{{1.0f, 0.0f}, {0.0f, 0.0f}};
        float max_abs_x = 0.0f;
        for (int i = 0; i <= steps; ++i)
        {
            if (i % print_every == 0)
                std::printf("t=%.2fs  x=%+.4f  vx=%+.4f\n", i * dt, s.pos.x,
                            s.vel.x);
            float tiltZ = -kP * s.pos.x;
            s = stepBallPhysics(s, 0.0f, tiltZ, dt, /*damping*/ 1.5f);
            if (i * dt > 1.0f)
            {
                float a = std::fabs(s.pos.x);
                if (a > max_abs_x)
                    max_abs_x = a;
            }
        }
        std::printf("final x=%+.4f  vx=%+.4f  peak|x| after t=1s: %.4f "
                    "(expect small — damped)\n",
                    s.pos.x, s.vel.x, max_abs_x);
    }
    return 0;
}
