#pragma once

// Filled UV-sphere rendered with a Phong shader. Owns a VAO/VBO/EBO and a
// GLSL program; construct after a GL context is current, destroy before it
// goes away.
class SphereRenderer
{
public:
    SphereRenderer() = default;
    ~SphereRenderer();

    SphereRenderer(const SphereRenderer &)            = delete;
    SphereRenderer &operator=(const SphereRenderer &) = delete;

    // Build the mesh and compile the shader. Returns false on shader error.
    bool init(int stacks = 32, int slices = 48);

    // aspect = width / height. Draws one sphere at the origin, rotated by
    // yaw (Y axis) and pitch (X axis) radians.
    void draw(float aspect, float yaw, float pitch);

private:
    unsigned int m_vao     = 0;
    unsigned int m_vbo     = 0;
    unsigned int m_ebo     = 0;
    unsigned int m_program = 0;
    int m_indexCount       = 0;

    // Cached uniform locations.
    int m_uModel = -1;
    int m_uView  = -1;
    int m_uProj  = -1;
    int m_uLight = -1;
    int m_uColor = -1;
    int m_uCam   = -1;
};
