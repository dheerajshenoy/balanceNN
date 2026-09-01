#pragma once

extern "C"
{
#include "SDL3/SDL_video.h"

#include <SDL3/SDL.h>
}

#include "nn.hpp"
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

    // Camera orbit.
    float m_yaw   = 0.0f;
    float m_pitch = 0.4f;

    // Plate tilt (radians) driven by keys A/D (Z-tilt) and W/S (X-tilt).
    float m_tiltX = 0.0f;
    float m_tiltZ = 0.0f;

    // Camera distance from scene origin (+/- keys zoom).
    float m_camDist = 5.0f;

    Uint64 m_lastTicks = 0;
    bool m_running     = true;

    NN model;
};
