#include "BalanceNN.hpp"

BalanceNN::BalanceNN()
{
    init();
}

BalanceNN::~BalanceNN()
{
    SDL_DestroyTexture(m_texture);
    SDL_DestroyRenderer(m_renderer);
    SDL_DestroyWindow(m_window);
    SDL_Quit();
}

void
BalanceNN::init()
{
    initGUI();
    initNN();

    m_cube      = makeCube(2.0f);
    m_sphere    = makeSphere(1.2f, 14, 20);
    m_lastTicks = SDL_GetTicks();
}

void
BalanceNN::initGUI()
{
    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer("BalanceNN", m_width, m_height,
                                SDL_WINDOW_RESIZABLE, &m_window, &m_renderer);
    SDL_SetRenderVSync(m_renderer, 1);
}

void
BalanceNN::update()
{
    Uint64 now  = SDL_GetTicks();
    float dt    = (now - m_lastTicks) / 1000.0f;
    m_lastTicks = now;
    m_angle += dt * 0.8f;

    SDL_GetWindowSize(m_window, &m_width, &m_height);
}

void
BalanceNN::render()
{
    SDL_SetRenderDrawColor(m_renderer, 15, 18, 24, 255);
    SDL_RenderClear(m_renderer);

    float cy    = m_height * 0.5f;
    float focal = m_height * 0.9f; // ~60 deg vertical fov
    float camZ  = 6.0f;

    SDL_SetRenderDrawColor(m_renderer, 100, 200, 255, 255);
    drawWireframe(m_renderer, m_cube, {0, 0, 0},
                  /*rotX*/ 0.5f, /*rotY*/ m_angle, m_width * 0.30f, cy, focal,
                  camZ);

    SDL_SetRenderDrawColor(m_renderer, 255, 180, 120, 255);
    drawWireframe(m_renderer, m_sphere, {0, 0, 0},
                  /*rotX*/ 0.3f, /*rotY*/ m_angle * 1.3f, m_width * 0.70f, cy,
                  focal, camZ);

    SDL_RenderPresent(m_renderer);
}

void
BalanceNN::initNN()
{
    // Initialize neural network components here
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
