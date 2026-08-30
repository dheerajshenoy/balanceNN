#include "sphere_renderer.hpp"

#include <glad/gl.h>

#include <cmath>
#include <cstdio>
#include <numbers>
#include <vector>

namespace
{

constexpr const char *kVertSrc = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vWorldPos;
out vec3 vNormal;

void main()
{
    vec4 world = uModel * vec4(aPos, 1.0);
    vWorldPos = world.xyz;
    vNormal = mat3(uModel) * aNormal;
    gl_Position = uProj * uView * world;
}
)";

constexpr const char *kFragSrc = R"(
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;

uniform vec3 uLightDir;
uniform vec3 uBaseColor;
uniform vec3 uCamPos;

out vec4 FragColor;

void main()
{
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(uCamPos - vWorldPos);
    vec3 H = normalize(L + V);

    float ambient  = 0.18;
    float diffuse  = max(dot(N, L), 0.0);
    float specular = pow(max(dot(N, H), 0.0), 48.0) * 0.5;

    vec3 color = uBaseColor * (ambient + diffuse) + vec3(specular);
    color = pow(color, vec3(1.0 / 2.2));
    FragColor = vec4(color, 1.0);
}
)";

GLuint
compile(GLenum stage, const char *src)
{
    GLuint s = glCreateShader(stage);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[1024];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::fprintf(stderr, "Shader compile error: %s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

GLuint
link(GLuint vs, GLuint fs)
{
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[1024];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::fprintf(stderr, "Program link error: %s\n", log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

// Column-major 4x4. Layout: m[col*4 + row].
struct Mat4
{
    float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};

Mat4
mul(const Mat4 &a, const Mat4 &b)
{
    Mat4 r{};
    for (int c = 0; c < 4; ++c)
        for (int rr = 0; rr < 4; ++rr)
        {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k)
                s += a.m[k * 4 + rr] * b.m[c * 4 + k];
            r.m[c * 4 + rr] = s;
        }
    return r;
}

Mat4
perspective(float fovY, float aspect, float zNear, float zFar)
{
    float f = 1.0f / std::tan(fovY * 0.5f);
    Mat4 p{};
    for (int i = 0; i < 16; ++i)
        p.m[i] = 0.0f;
    p.m[0]  = f / aspect;
    p.m[5]  = f;
    p.m[10] = (zFar + zNear) / (zNear - zFar);
    p.m[11] = -1.0f;
    p.m[14] = (2.0f * zFar * zNear) / (zNear - zFar);
    return p;
}

Mat4
translation(float x, float y, float z)
{
    Mat4 m;
    m.m[12] = x;
    m.m[13] = y;
    m.m[14] = z;
    return m;
}

Mat4
rotateY(float a)
{
    float c = std::cos(a), s = std::sin(a);
    Mat4 m;
    m.m[0]  = c;
    m.m[2]  = -s;
    m.m[8]  = s;
    m.m[10] = c;
    return m;
}

Mat4
scale(float s)
{
    Mat4 m;
    m.m[0]  = s;
    m.m[5]  = s;
    m.m[10] = s;
    return m;
}

Mat4
rotateX(float a)
{
    float c = std::cos(a), s = std::sin(a);
    Mat4 m;
    m.m[5]  = c;
    m.m[6]  = s;
    m.m[9]  = -s;
    m.m[10] = c;
    return m;
}

// Upload interleaved (pos3, normal3) verts + index buffer to a fresh mesh.
void
uploadMesh(SceneRenderer::Mesh &m, const std::vector<float> &verts,
           const std::vector<unsigned int> &indices)
{
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glGenBuffers(1, &m.ebo);

    glBindVertexArray(m.vao);

    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(),
                 GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 indices.size() * sizeof(unsigned int), indices.data(),
                 GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    m.indexCount = static_cast<int>(indices.size());
}

} // namespace

// Bring SceneRenderer::Mesh into scope for uploadMesh above via friend?
// Simpler: it's a public struct, uploadMesh accesses it directly.

SceneRenderer::~SceneRenderer()
{
    destroyMesh(m_sphere);
    destroyMesh(m_plate);
    if (m_program)
        glDeleteProgram(m_program);
}

void
SceneRenderer::destroyMesh(Mesh &m)
{
    if (m.ebo)
        glDeleteBuffers(1, &m.ebo);
    if (m.vbo)
        glDeleteBuffers(1, &m.vbo);
    if (m.vao)
        glDeleteVertexArrays(1, &m.vao);
    m = {};
}

void
SceneRenderer::buildSphere(Mesh &m, int stacks, int slices)
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
            // Unit sphere → normal == position.
            verts.insert(verts.end(), {x, y, z, x, y, z});
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

    uploadMesh(m, verts, indices);
}

void
SceneRenderer::buildPlate(Mesh &m, float w, float h, float d)
{
    float x = w * 0.5f, y = h * 0.5f, z = d * 0.5f;

    // 6 faces × 4 verts each, per-face normals (flat shaded box).
    // Interleaved (pos, normal). Winding: CCW when viewed from outside.
    std::vector<float> verts = {
        // +Y (top)
        -x,  y, -z,   0,  1,  0,
         x,  y, -z,   0,  1,  0,
         x,  y,  z,   0,  1,  0,
        -x,  y,  z,   0,  1,  0,
        // -Y (bottom)
        -x, -y,  z,   0, -1,  0,
         x, -y,  z,   0, -1,  0,
         x, -y, -z,   0, -1,  0,
        -x, -y, -z,   0, -1,  0,
        // +X (right)
         x, -y, -z,   1,  0,  0,
         x, -y,  z,   1,  0,  0,
         x,  y,  z,   1,  0,  0,
         x,  y, -z,   1,  0,  0,
        // -X (left)
        -x, -y,  z,  -1,  0,  0,
        -x, -y, -z,  -1,  0,  0,
        -x,  y, -z,  -1,  0,  0,
        -x,  y,  z,  -1,  0,  0,
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

    uploadMesh(m, verts, indices);
}

bool
SceneRenderer::init(int stacks, int slices)
{
    GLuint vs = compile(GL_VERTEX_SHADER, kVertSrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, kFragSrc);
    if (!vs || !fs)
        return false;
    m_program = link(vs, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);
    if (!m_program)
        return false;

    m_uModel = glGetUniformLocation(m_program, "uModel");
    m_uView  = glGetUniformLocation(m_program, "uView");
    m_uProj  = glGetUniformLocation(m_program, "uProj");
    m_uLight = glGetUniformLocation(m_program, "uLightDir");
    m_uColor = glGetUniformLocation(m_program, "uBaseColor");
    m_uCam   = glGetUniformLocation(m_program, "uCamPos");

    buildSphere(m_sphere, stacks, slices);
    buildPlate(m_plate, /*w*/ 4.0f, /*h*/ 0.2f, /*d*/ 4.0f);
    return true;
}

void
SceneRenderer::drawMesh(const Mesh &m, const float model[16], float r, float g,
                        float b)
{
    glUniformMatrix4fv(m_uModel, 1, GL_FALSE, model);
    glUniform3f(m_uColor, r, g, b);
    glBindVertexArray(m.vao);
    glDrawElements(GL_TRIANGLES, m.indexCount, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

void
SceneRenderer::draw(float aspect, float yaw, float pitch)
{
    // Scene transform (applied to every object so they turn together).
    Mat4 sceneRot = mul(rotateY(yaw), rotateX(pitch));
    Mat4 view     = translation(0.0f, -0.4f, -5.0f);
    Mat4 proj     = perspective(60.0f * std::numbers::pi_v<float> / 180.0f,
                                aspect, 0.1f, 100.0f);

    glUseProgram(m_program);
    glUniformMatrix4fv(m_uView, 1, GL_FALSE, view.m);
    glUniformMatrix4fv(m_uProj, 1, GL_FALSE, proj.m);
    glUniform3f(m_uLight, 0.4f, 0.85f, 0.5f);
    glUniform3f(m_uCam, 0.0f, 0.4f, 5.0f);

    // Marble: 10% of the plate's 4-unit width → radius 0.2. Plate top sits
    // at y = +0.1, so the marble center is at y = 0.1 + 0.2 = 0.3.
    constexpr float plateWidth = 4.0f;
    constexpr float marbleR    = plateWidth * 0.1f * 0.5f; // 0.2
    Mat4 sphereModel =
        mul(sceneRot, mul(translation(0.0f, 0.1f + marbleR, 0.0f),
                          scale(marbleR)));
    drawMesh(m_sphere, sphereModel.m, 0.85f, 0.45f, 0.25f);

    // Plate centered at y = 0.
    drawMesh(m_plate, sceneRot.m, 0.35f, 0.4f, 0.5f);
}
