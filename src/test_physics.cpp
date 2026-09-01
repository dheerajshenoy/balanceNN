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
                float muK = 0.05f, float damping = 0.0f, float gravity = 9.81f)
{
    float ax = gravity * std::sin(tiltZ);
    float ay = gravity * std::sin(tiltX);
    s.vel.x += ax * dt;
    s.vel.y += ay * dt;

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

    // ---- Experiment 1: P-controller, NO friction.
    // Expect: sustained oscillation about x=0 (energy conserved).
    {
        std::printf("=== EXP 1: P-controller (k=%.2f), muK=0, start x=+1.0\n",
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
            s = stepBallPhysics(s, /*tiltX*/ 0.0f, tiltZ, dt, /*muK*/ 0.0f);
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
        std::printf("zero-crossings: %d (expect many — no friction)\n",
                    crossings);
    }

    // ---- Experiment 2: kinetic friction only, level plate, initial push.
    // Ball should decelerate linearly and STOP exactly (v==0), not decay
    // asymptotically the way viscous damping does.
    {
        std::printf("\n=== EXP 2: level plate, muK=0.3, start vx=+2.0\n");
        BallState s{{0.0f, 0.0f}, {2.0f, 0.0f}};
        int stop_step = -1;
        for (int i = 0; i <= steps; ++i)
        {
            if (i % 10 == 0)
                std::printf("t=%.3fs  x=%+.4f  vx=%+.4f\n", i * dt, s.pos.x,
                            s.vel.x);
            s = stepBallPhysics(s, 0.0f, 0.0f, dt, /*muK*/ 0.3f);
            if (stop_step < 0 && s.vel.x == 0.0f && s.vel.y == 0.0f)
            {
                stop_step = i;
                break;
            }
        }
        if (stop_step < 0)
            std::printf("FAIL: ball never stopped\n");
        else
            std::printf("stopped at t=%.3fs, x=%+.4f (expected ~0.68s from "
                        "2.0/(0.3*9.81))\n",
                        stop_step * dt, s.pos.x);
    }

    // ---- Experiment 3: P-controller + friction. Amplitude decays and
    // the ball comes to rest near x=0 (may rest slightly off-center if the
    // required tilt to move it is below the friction threshold — for pure
    // kinetic friction with no static term, it will still creep, but only
    // while v>0; once stopped it stays stopped until tilt changes).
    {
        std::printf("\n=== EXP 3: P-controller (k=%.2f) + muK=0.05, "
                    "start x=+1.0\n",
                    kP);
        BallState s{{1.0f, 0.0f}, {0.0f, 0.0f}};
        for (int i = 0; i <= steps; ++i)
        {
            if (i % print_every == 0)
                std::printf("t=%.2fs  x=%+.4f  vx=%+.4f\n", i * dt, s.pos.x,
                            s.vel.x);
            float tiltZ = -kP * s.pos.x;
            s = stepBallPhysics(s, 0.0f, tiltZ, dt, /*muK*/ 0.05f);
        }
        std::printf("final x=%+.4f  vx=%+.4f\n", s.pos.x, s.vel.x);
    }
    return 0;
}
