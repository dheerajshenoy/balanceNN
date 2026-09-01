#include "plate.hpp"

Plate::Plate(float width, float height, float depth)
    : m_width(width), m_height(height), m_depth(depth)
{
    float x = m_width * 0.5f, y = m_height * 0.5f, z = m_depth * 0.5f;

    // Interleaved (pos, normal). Winding: CCW when viewed from outside.
    std::vector<float> verts = {
        // +Y (top)
        -x,  y,  z,   0,  1,  0,
         x,  y,  z,   0,  1,  0,
         x,  y, -z,   0,  1,  0,
        -x,  y, -z,   0,  1,  0,
        // -Y (bottom)
        -x, -y, -z,   0, -1,  0,
         x, -y, -z,   0, -1,  0,
         x, -y,  z,   0, -1,  0,
        -x, -y,  z,   0, -1,  0,
        // +X (right)
         x, -y,  z,   1,  0,  0,
         x, -y, -z,   1,  0,  0,
         x,  y, -z,   1,  0,  0,
         x,  y,  z,   1,  0,  0,
        // -X (left)
        -x, -y, -z,  -1,  0,  0,
        -x, -y,  z,  -1,  0,  0,
        -x,  y,  z,  -1,  0,  0,
        -x,  y, -z,  -1,  0,  0,
        // +Z (front)
        -x, -y,  z,   0,  0,  1,
         x, -y,  z,   0,  0,  1,
         x,  y,  z,   0,  0,  1,
        -x,  y,  z,   0,  0,  1,
        // -Z (back)
         x, -y, -z,   0,  0, -1,
        -x, -y, -z,   0,  0, -1,
        -x,  y, -z,   0,  0, -1,
         x,  y, -z,   0,  0, -1,
    };

    std::vector<unsigned int> indices;
    indices.reserve(36);
    for (unsigned int f = 0; f < 6; ++f)
    {
        unsigned int base = f * 4;
        indices.insert(indices.end(),
                       {base, base + 1, base + 2, base, base + 2, base + 3});
    }

    uploadMesh(m_mesh, verts, indices);
}

Plate::~Plate()
{
    destroyMesh(m_mesh);
}

Mat4
Plate::tiltMatrix() const
{
    // Negate rotateZ so positive tiltZ dips the +X edge DOWN (matches
    // the physics convention ax = g*sin(tiltZ): downhill is +X).
    return mul(rotateZ(-m_tiltZ), rotateX(m_tiltX));
}

void
Plate::draw(const PhongProgram &prog, const Mat4 &parent) const
{
    Mat4 model = mul(parent, tiltMatrix());
    prog.drawMesh(m_mesh, model, 0.35f, 0.4f, 0.5f);
}
