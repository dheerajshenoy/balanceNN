#pragma once

// Simple Phong-lit scene: a filled sphere sitting on a rectangular plate.
// Owns one GLSL program and two meshes. Construct after a GL context is
// current, destroy before it goes away.
class SceneRenderer
{
public:
    SceneRenderer() = default;
    ~SceneRenderer();

    SceneRenderer(const SceneRenderer &)            = delete;
    SceneRenderer &operator=(const SceneRenderer &) = delete;

    bool init(int stacks = 32, int slices = 48);

    // aspect = width / height. Rotates the whole scene by yaw (Y) and
    // pitch (X) so the sphere and plate turn together.
    void draw(float aspect, float yaw, float pitch);

public:
    struct Mesh
    {
        unsigned int vao = 0;
        unsigned int vbo = 0;
        unsigned int ebo = 0;
        int indexCount   = 0;
    };

private:
    void buildSphere(Mesh &m, int stacks, int slices);
    void buildPlate(Mesh &m, float w, float h, float d);
    void destroyMesh(Mesh &m);

    void drawMesh(const Mesh &m, const float model[16], float r, float g,
                  float b);

    Mesh m_sphere;
    Mesh m_plate;

    unsigned int m_program = 0;
    int m_uModel           = -1;
    int m_uView            = -1;
    int m_uProj            = -1;
    int m_uLight           = -1;
    int m_uColor           = -1;
    int m_uCam             = -1;
};

// Kept for source compatibility.
using SphereRenderer = SceneRenderer;
