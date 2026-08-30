#pragma once

extern "C"
{
#include "SDL3/SDL_video.h"

#include <SDL3/SDL.h>
}

#include "shape3d.hpp"

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

    SDL_Window *m_window     = nullptr;
    SDL_Renderer *m_renderer = nullptr;
    SDL_Texture *m_texture   = nullptr;

    int m_width  = 1280;
    int m_height = 720;

    Mesh m_cube;
    Mesh m_sphere;

    float m_angle      = 0.0f;
    Uint64 m_lastTicks = 0;

    bool m_running = true;
};
