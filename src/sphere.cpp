#include "sphere.hpp"

#include <cmath>
#include <numbers>

namespace
{
constexpr float kGravity = 9.81f;
constexpr float kMuK     = 0.05f; // rolling-friction coefficient
constexpr float kDamping = 0.0f;  // extra viscous drag (usually 0)
} // namespace

Sphere::Sphere(float radius, int stacks, int slices) : m_radius(radius)
{
    constexpr float pi = std::numbers::pi_v<float>;
    std::vector<float> verts;
    verts.reserve((stacks + 1) * (slices + 1) * 6);

    for (int i = 0; i <= stacks; ++i)
    {
        float v   = static_cast<float>(i) / stacks;
        float phi = v * pi;
        float y   = std::cos(phi);
        float r   = std::sin(phi);
        for (int j = 0; j <= slices; ++j)
        {
            float u     = static_cast<float>(j) / slices;
            float theta = u * 2.0f * pi;
            float x     = r * std::cos(theta);
            float z     = r * std::sin(theta);
            verts.insert(verts.end(), {x, y, z, x, y, z}); // unit sphere
        }
    }

    std::vector<unsigned int> indices;
    indices.reserve(stacks * slices * 6);
    int ringLen = slices + 1;
    for (int i = 0; i < stacks; ++i)
        for (int j = 0; j < slices; ++j)
        {
            unsigned int a = i * ringLen + j;
            unsigned int b = a + 1;
            unsigned int c = a + ringLen;
            unsigned int d = c + 1;
            indices.insert(indices.end(), {a, c, b, b, c, d});
        }

    uploadMesh(m_mesh, verts, indices);
}

Sphere::~Sphere()
{
    destroyMesh(m_mesh);
}

void
Sphere::reset(float x, float z)
{
    m_position     = {x, z};
    m_velocity     = {0.0f, 0.0f};
    m_acceleration = {0.0f, 0.0f};
    m_fallen       = false;
}

void
Sphere::update(float dt, const Plate &plate)
{
    if (m_fallen)
        return;

    BallState next = stepBallPhysics({m_position, m_velocity}, plate.tiltX(),
                                     plate.tiltZ(), dt, kMuK, kDamping,
                                     kGravity);
    m_acceleration.x = (next.vel.x - m_velocity.x) / (dt > 0.0f ? dt : 1.0f);
    m_acceleration.y = (next.vel.y - m_velocity.y) / (dt > 0.0f ? dt : 1.0f);
    m_position       = next.pos;
    m_velocity       = next.vel;

    if (std::fabs(m_position.x) > plate.halfWidth()
        || std::fabs(m_position.y) > plate.halfDepth())
        m_fallen = true;
}

void
Sphere::draw(const PhongProgram &prog, const Mat4 &parent,
             float plateTopY) const
{
    Mat4 local = mul(translation(m_position.x, plateTopY + m_radius,
                                 m_position.y),
                     scale(m_radius));
    Mat4 model = mul(parent, local);
    prog.drawMesh(m_mesh, model, 0.85f, 0.45f, 0.25f);
}
