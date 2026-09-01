#include "BalanceNN.hpp"

#include <algorithm>
#include <cstdio>
#include <glad/gl.h>
#include <numbers>

namespace
{
constexpr float kActionRate   = 4.0f; // how fast keys drive action toward ±1
constexpr float kActionReturn = 6.0f; // decay toward 0 when key released
constexpr float kZoomRate     = 4.0f; // world units/s
constexpr float kMinCamDist   = 1.5f;
constexpr float kMaxCamDist   = 30.0f;
} // namespace

BalanceNN::BalanceNN()
{
    init();
}

BalanceNN::~BalanceNN()
{
    delete m_sphere;
    delete m_plate;
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

    if (!m_program.init())
        std::fprintf(stderr, "PhongProgram init failed\n");

    BallPlateEnv::Config cfg;
    cfg.plateWidth = 4.0f;
    cfg.plateDepth = 4.0f;
    m_env          = BallPlateEnv(cfg);
    m_env.reset(/*seed*/ 42);

    m_plate  = new Plate(cfg.plateWidth, 0.2f, cfg.plateDepth);
    m_sphere = new Sphere(cfg.plateWidth * 0.05f);

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
    m_glCtx  = SDL_GL_CreateContext(m_window);
    SDL_GL_MakeCurrent(m_window, m_glCtx);
    SDL_GL_SetSwapInterval(1);

    int version
        = gladLoadGL(reinterpret_cast<GLADloadfunc>(SDL_GL_GetProcAddress));
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
    if (dt > 0.05f)
        dt = 0.05f;

    SDL_GetWindowSize(m_window, &m_width, &m_height);

    // Build a target action from keys, then smooth it so tilt doesn't snap.
    const bool *keys = SDL_GetKeyboardState(nullptr);
    float dActX      = 0.0f;
    float dActZ      = 0.0f;
    if (keys[SDL_SCANCODE_W])
        dActX -= 1.0f;
    if (keys[SDL_SCANCODE_S])
        dActX += 1.0f;
    if (keys[SDL_SCANCODE_A])
        dActZ -= 1.0f;
    if (keys[SDL_SCANCODE_D])
        dActZ += 1.0f;

    auto approach = [dt](float cur, float target, float rate) {
        float step = rate * dt;
        if (cur < target - step)
            return cur + step;
        if (cur > target + step)
            return cur - step;
        return target;
    };
    m_actionX = dActX != 0.0f
                    ? approach(m_actionX, dActX, kActionRate)
                    : approach(m_actionX, 0.0f, kActionReturn);
    m_actionZ = dActZ != 0.0f
                    ? approach(m_actionZ, dActZ, kActionRate)
                    : approach(m_actionZ, 0.0f, kActionReturn);

    // Zoom.
    float dZoom = 0.0f;
    if (keys[SDL_SCANCODE_EQUALS] || keys[SDL_SCANCODE_KP_PLUS])
        dZoom -= 1.0f;
    if (keys[SDL_SCANCODE_MINUS] || keys[SDL_SCANCODE_KP_MINUS])
        dZoom += 1.0f;
    m_camDist = std::clamp(m_camDist + dZoom * kZoomRate * dt, kMinCamDist,
                           kMaxCamDist);

    // Step the env at its fixed dt, catching up whatever the render frame took.
    m_physicsAcc += dt;
    float envDt = m_env.config().dt;
    while (m_physicsAcc >= envDt)
    {
        if (m_env.done())
            m_env.reset(SDL_GetTicks());
        m_env.step(m_actionX, m_actionZ);
        m_physicsAcc -= envDt;
    }

    // Push env state into the render objects.
    const Observation &o = m_env.observe();
    m_plate->setTilt(o.tiltX, o.tiltZ);
    m_sphere->setPosition(o.px, o.pz);
    m_sphere->setVelocity(o.vx, o.vz);
}

void
BalanceNN::render()
{
    glViewport(0, 0, m_width, m_height);
    glClearColor(0.06f, 0.07f, 0.09f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float aspect = static_cast<float>(m_width)
                   / static_cast<float>(m_height > 0 ? m_height : 1);

    Mat4 sceneRot = mul(rotateY(m_yaw), rotateX(m_pitch));
    Mat4 view     = translation(0.0f, -0.4f, -m_camDist);
    Mat4 proj = perspective(60.0f * std::numbers::pi_v<float> / 180.0f, aspect,
                            0.1f, 100.0f);

    m_program.use();
    m_program.setView(view);
    m_program.setProj(proj);
    m_program.setLightDir(0.4f, 0.85f, 0.5f);
    m_program.setCameraPos(0.0f, 0.4f, m_camDist);

    m_plate->draw(m_program, sceneRot);

    Mat4 plateFrame = mul(sceneRot, m_plate->tiltMatrix());
    m_sphere->draw(m_program, plateFrame, m_plate->topY());

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
