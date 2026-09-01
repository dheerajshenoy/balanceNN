#pragma once

#include "env.hpp"

// Hand-tuned baseline controller. Pure math — no GL.
//
// Axis pairing (see env conventions): tiltX rotates the plate about world X,
// which tips the Z-axis, so tiltX controls Z-position. tiltZ controls
// X-position. Signs are chosen so a positive position → negative action →
// tilt that accelerates the ball back toward origin.
struct PDController
{
    float Kp = 2.0f;
    float Kd = 1.5f;

    struct Action
    {
        float x, z;
    };
    Action compute(const Observation &obs) const
    {
        return {-Kp * obs.pz - Kd * obs.vz, -Kp * obs.px - Kd * obs.vx};
    }
};
