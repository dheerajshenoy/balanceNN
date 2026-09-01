#pragma once

extern "C"
{
#include "SDL3/SDL_video.h"

#include <SDL3/SDL.h>
}

#include "env.hpp"
#include "nn.hpp"
#include "obs_normalizer.hpp"
#include "pd_controller.hpp"
#include "plate.hpp"
#include "policy.hpp"
#include "render_common.hpp"
#include "sphere.hpp"

#include <string>

class BalanceNN
{

public:
    // If policy_path is non-empty, we try to load <path>.actor and <path>.norm
    // at startup. Missing files just disable Neural mode without failing.
    explicit BalanceNN(std::string policy_path = "");
    ~BalanceNN();
    void run();

    enum class Mode
    {
        Keyboard,
        PD,
        Neural
    };

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
    Plate *m_plate        = nullptr;
    Plate *m_targetMarker = nullptr; // small flat plate as the goal marker
    Sphere *m_sphere      = nullptr;

    // Target mode: a random point on the plate that the controller tries
    // to reach. Toggled with T. When on, observation position is fed to
    // controllers as (obs.px - target.x, obs.pz - target.z) so the same
    // trained "go to origin" policy chases the target without retraining.
    bool m_targetMode      = false;
    float m_targetX        = 0.0f;
    float m_targetZ        = 0.0f;
    float m_targetTimer    = 0.0f;   // seconds until next randomization
    float m_targetHitDist  = 0.15f;  // ball within this = "reached"
    unsigned m_targetSeed  = 1;      // for reproducibility across runs

    // Env owns physics + tilt; renderer is a pure observer.
    BallPlateEnv m_env;

    PDController m_pd;

    Mode m_mode          = Mode::Keyboard;
    Actor m_actor        = nullptr; // null if no policy loaded
    ObsNormalizer m_norm{6}; // populated iff actor loaded (obs dim = 6)

    std::string m_policyPath;

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
