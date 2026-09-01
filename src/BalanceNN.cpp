#include "BalanceNN.hpp"

#include <algorithm>
#include <cstdio>
#include <glad/gl.h>
#include <numbers>
#include <random>

namespace
{
constexpr float kActionRate   = 4.0f; // how fast keys drive action toward ±1
constexpr float kActionReturn = 6.0f; // decay toward 0 when key released
constexpr float kZoomRate     = 4.0f; // world units/s
constexpr float kMinCamDist   = 1.5f;
constexpr float kMaxCamDist   = 30.0f;
constexpr float kTargetInterval = 5.0f; // seconds before target re-rolls
constexpr float kTargetMargin   = 0.4f; // keep target away from plate edge

// Draw a random target position inside the plate, at least `margin` from
// each edge. Seed is state kept by the caller.
void
randomizeTarget(unsigned &seed, float halfW, float halfD, float margin,
                float &tx, float &tz)
{
    std::mt19937 rng(seed++);
    std::uniform_real_distribution<float> ux(-halfW + margin, halfW - margin);
    std::uniform_real_distribution<float> uz(-halfD + margin, halfD - margin);
    tx = ux(rng);
    tz = uz(rng);
}
} // namespace

BalanceNN::BalanceNN(std::string policy_path)
    : m_policyPath(std::move(policy_path))
{
    init();
}

BalanceNN::~BalanceNN()
{
    delete m_sphere;
    delete m_targetMarker;
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
    // Flat marker: same class, tiny thin XZ footprint.
    m_targetMarker = new Plate(0.3f, 0.02f, 0.3f);

    if (!m_policyPath.empty())
    {
        Actor a(6, 2, /*hidden*/ 64, /*init_log_std*/ -0.5);
        try
        {
            torch::load(a, m_policyPath + ".actor");
            if (!m_norm.load(m_policyPath + ".norm"))
                std::fprintf(stderr,
                             "warning: could not load %s.norm; policy will "
                             "run with un-normalized inputs\n",
                             m_policyPath.c_str());
            m_actor = a;
            m_mode  = Mode::Neural;
            std::printf("loaded policy from %s.{actor,norm}; starting in "
                        "Neural mode (press K for manual, P for PD)\n",
                        m_policyPath.c_str());
        }
        catch (const std::exception &e)
        {
            std::fprintf(stderr, "failed to load %s.actor: %s\n",
                         m_policyPath.c_str(), e.what());
        }
    }

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

    // Target-mode housekeeping: countdown timer, or immediately re-roll if
    // the ball is within the "reached" radius of the current target.
    if (m_targetMode)
    {
        m_targetTimer -= dt;
        const Observation &o = m_env.observe();
        float dx = o.px - m_targetX;
        float dz = o.pz - m_targetZ;
        bool reached = (dx * dx + dz * dz) < m_targetHitDist * m_targetHitDist;
        if (m_targetTimer <= 0.0f || reached)
        {
            randomizeTarget(m_targetSeed, m_plate->halfWidth(),
                            m_plate->halfDepth(), kTargetMargin, m_targetX,
                            m_targetZ);
            m_targetTimer = kTargetInterval;
            if (reached)
                std::printf("target reached; new target = (%+.2f, %+.2f)\n",
                            m_targetX, m_targetZ);
        }
    }

    // Step the env at its fixed dt, catching up whatever the render frame took.
    m_physicsAcc += dt;
    float envDt = m_env.config().dt;
    while (m_physicsAcc >= envDt)
    {
        if (m_env.done())
            m_env.reset(SDL_GetTicks());

        // If target mode is on, transform the observation so the target
        // looks like the origin to the controller (which was trained/tuned
        // to reach the origin). Velocity, tilt are untouched.
        Observation ctrl_obs = m_env.observe();
        if (m_targetMode)
        {
            ctrl_obs.px -= m_targetX;
            ctrl_obs.pz -= m_targetZ;
        }

        float ax = 0.0f, az = 0.0f;
        switch (m_mode)
        {
        case Mode::Keyboard:
            ax = m_actionX;
            az = m_actionZ;
            break;
        case Mode::PD:
        {
            auto a = m_pd.compute(ctrl_obs);
            ax     = a.x;
            az     = a.z;
            break;
        }
        case Mode::Neural:
            if (m_actor)
            {
                float raw[6];
                raw[0] = ctrl_obs.px;   raw[1] = ctrl_obs.pz;
                raw[2] = ctrl_obs.vx;   raw[3] = ctrl_obs.vz;
                raw[4] = ctrl_obs.tiltX; raw[5] = ctrl_obs.tiltZ;
                m_norm.normalize(raw);
                torch::NoGradGuard nograd;
                auto obs_t = torch::from_blob(raw, {1, 6},
                                              torch::TensorOptions().dtype(
                                                  torch::kFloat32))
                                 .clone();
                auto a = m_actor->mean(obs_t);
                ax = a[0][0].item<float>();
                az = a[0][1].item<float>();
            }
            break;
        }
        m_env.step(ax, az);
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

    if (m_targetMode)
    {
        // Marker sits on top of the plate. Its own tilt is 0 (it doesn't
        // need to tilt independently of the parent), and we place it in
        // plate-local space via translation, then let the plateFrame
        // parent bring it into world.
        Mat4 markerLocal = translation(m_targetX,
                                       m_plate->topY()
                                           + m_targetMarker->height() * 0.5f,
                                       m_targetZ);
        Mat4 markerFrame = mul(plateFrame, markerLocal);
        m_targetMarker->draw(m_program, markerFrame, /*r*/ 1.0f, /*g*/ 0.85f,
                             /*b*/ 0.15f);
    }

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
        else if (event.type == SDL_EVENT_KEY_DOWN)
        {
            if (event.key.key == SDLK_ESCAPE)
                m_running = false;
            // Mode selection: K = manual (keyboard), P = PD, N = neural.
            // Each key SETS the mode (not toggles) so switching is
            // unambiguous even after mode changes from elsewhere.
            else if (event.key.key == SDLK_K)
            {
                m_mode    = Mode::Keyboard;
                m_actionX = m_actionZ = 0.0f;
                std::printf("mode: keyboard\n");
            }
            else if (event.key.key == SDLK_P)
            {
                m_mode    = Mode::PD;
                m_actionX = m_actionZ = 0.0f;
                std::printf("mode: PD\n");
            }
            else if (event.key.key == SDLK_N)
            {
                if (!m_actor)
                    std::printf("no policy loaded (pass --policy <path>)\n");
                else
                {
                    m_mode    = Mode::Neural;
                    m_actionX = m_actionZ = 0.0f;
                    std::printf("mode: neural\n");
                }
            }
            else if (event.key.key == SDLK_R)
                m_env.reset(SDL_GetTicks());
            else if (event.key.key == SDLK_T)
            {
                m_targetMode = !m_targetMode;
                if (m_targetMode)
                {
                    randomizeTarget(m_targetSeed, m_plate->halfWidth(),
                                    m_plate->halfDepth(), kTargetMargin,
                                    m_targetX, m_targetZ);
                    m_targetTimer = kTargetInterval;
                    std::printf("target mode: ON  target=(%+.2f, %+.2f)\n",
                                m_targetX, m_targetZ);
                }
                else
                {
                    std::printf("target mode: OFF\n");
                }
            }
        }
    }
}

void
BalanceNN::initNN()
{
}
