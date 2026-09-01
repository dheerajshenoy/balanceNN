#pragma once

extern "C"
{
#include "SDL3/SDL_video.h"

#include <SDL3/SDL.h>
}

#include "env.hpp"
#include "nn.hpp"
#include "pd_controller.hpp"
#include "plate.hpp"
#include "render_common.hpp"
#include "sphere.hpp"

class BalanceNN
{

public:
    BalanceNN();
    ~BalanceNN();
    void run();

private:
    void init();
    void initGUI();
    void initNN();
    void render();
    void update();
    void handleEvents(SDL_Event &event);

    SDL_Window *m_window  = nullptr;
    SDL_GLContext m_glCtx = nullptr;

    int m_width  = 1920;
    int m_height = 1080;

    PhongProgram m_program;
    Plate *m_plate   = nullptr;
    Sphere *m_sphere = nullptr;

    // Env owns physics + tilt; renderer is a pure observer.
    BallPlateEnv m_env;

    PDController m_pd;
    bool m_autopilot = false;

    // Smoothed keyboard action (before clipping to [-1, 1]) so the tilt
    // doesn't snap on key press/release.
    float m_actionX = 0.0f;
    float m_actionZ = 0.0f;

    // Camera orbit.
    float m_yaw   = 0.0f;
    float m_pitch = 0.4f;

    // Camera distance from scene origin (+/- keys zoom).
    float m_camDist = 5.0f;

    // Accumulator to run env.step() at a fixed dt regardless of frame rate.
    float m_physicsAcc = 0.0f;

    Uint64 m_lastTicks = 0;
    bool m_running     = true;

    NN model;
};
