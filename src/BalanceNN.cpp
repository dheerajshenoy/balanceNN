#include "BalanceNN.hpp"

#include <glad/gl.h>

#include <cstdio>

BalanceNN::BalanceNN()
{
    init();
}

BalanceNN::~BalanceNN()
{
    if (m_glCtx)
        SDL_GL_DestroyContext(m_glCtx);
    SDL_DestroyWindow(m_window);
    SDL_Quit();
}

void
BalanceNN::init()
{
    initGUI();
    initNN();

    if (!m_sphere.init())
        std::fprintf(stderr, "SphereRenderer init failed\n");

    m_lastTicks = SDL_GetTicks();
}

void
BalanceNN::initGUI()
{
    SDL_Init(SDL_INIT_VIDEO);

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                        SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    m_window = SDL_CreateWindow("BalanceNN", m_width, m_height,
                                SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    m_glCtx = SDL_GL_CreateContext(m_window);
    SDL_GL_MakeCurrent(m_window, m_glCtx);
    SDL_GL_SetSwapInterval(1);

    int version = gladLoadGL(reinterpret_cast<GLADloadfunc>(
        SDL_GL_GetProcAddress));
    if (version == 0)
        std::fprintf(stderr, "Failed to load OpenGL\n");

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
}

void
BalanceNN::update()
{
    Uint64 now  = SDL_GetTicks();
    float dt    = (now - m_lastTicks) / 1000.0f;
    m_lastTicks = now;
    m_yaw += dt * 0.8f;

    SDL_GetWindowSize(m_window, &m_width, &m_height);
}

void
BalanceNN::render()
{
    glViewport(0, 0, m_width, m_height);
    glClearColor(0.06f, 0.07f, 0.09f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float aspect = static_cast<float>(m_width) /
                   static_cast<float>(m_height > 0 ? m_height : 1);
    m_sphere.draw(aspect, m_yaw, m_pitch);

    SDL_GL_SwapWindow(m_window);
}

void
BalanceNN::run()
{
    SDL_Event event;

    while (m_running)
    {
        handleEvents(event);
        update();
        render();
    }
}

void
BalanceNN::handleEvents(SDL_Event &event)
{
    while (SDL_PollEvent(&event))
    {
        if (event.type == SDL_EVENT_QUIT)
            m_running = false;
        else if (event.type == SDL_EVENT_KEY_DOWN
                 && event.key.key == SDLK_ESCAPE)
            m_running = false;
    }
}

void
BalanceNN::initNN()
{
}
