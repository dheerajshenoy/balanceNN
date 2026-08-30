#pragma once

extern "C"
{
#include "SDL3/SDL_video.h"

#include <SDL3/SDL.h>
}

#include "nn.hpp"
#include "sphere_renderer.hpp"

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

    SDL_Window *m_window   = nullptr;
    SDL_GLContext m_glCtx  = nullptr;

    int m_width  = 1280;
    int m_height = 720;

    SphereRenderer m_sphere;

    float m_yaw        = 0.0f;
    float m_pitch      = 0.4f;
    Uint64 m_lastTicks = 0;

    bool m_running = true;

    NN model;
};
