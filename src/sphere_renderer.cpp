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
    // Assuming uniform scale — otherwise pass a normal matrix.
    vNormal = mat3(uModel) * aNormal;
    gl_Position = uProj * uView * world;
}
)";

constexpr const char *kFragSrc = R"(
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;

uniform vec3 uLightDir;   // direction *toward* the light
uniform vec3 uBaseColor;
uniform vec3 uCamPos;

out vec4 FragColor;

void main()
{
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(uCamPos - vWorldPos);
    vec3 H = normalize(L + V);

    float ambient  = 0.15;
    float diffuse  = max(dot(N, L), 0.0);
    float specular = pow(max(dot(N, H), 0.0), 48.0) * 0.6;

    vec3 color = uBaseColor * (ambient + diffuse) + vec3(specular);
    // Simple gamma.
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

// Column-major 4x4 matrices (OpenGL convention). Layout: m[col*4 + row].
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
    m.m[0] = c;
    m.m[2] = -s;
    m.m[8] = s;
    m.m[10] = c;
    return m;
}

Mat4
rotateX(float a)
{
    float c = std::cos(a), s = std::sin(a);
    Mat4 m;
    m.m[5] = c;
    m.m[6] = s;
    m.m[9] = -s;
    m.m[10] = c;
    return m;
}

} // namespace

SphereRenderer::~SphereRenderer()
{
    if (m_program)
        glDeleteProgram(m_program);
    if (m_ebo)
        glDeleteBuffers(1, &m_ebo);
    if (m_vbo)
        glDeleteBuffers(1, &m_vbo);
    if (m_vao)
        glDeleteVertexArrays(1, &m_vao);
}

bool
SphereRenderer::init(int stacks, int slices)
{
    // ---- Build UV-sphere mesh: interleaved pos(3) + normal(3). ----
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
    m_indexCount = static_cast<int>(indices.size());

    // ---- GL objects. ----
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glGenBuffers(1, &m_ebo);

    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(),
                 GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
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

    // ---- Shader. ----
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
    return true;
}

void
SphereRenderer::draw(float aspect, float yaw, float pitch)
{
    Mat4 model = mul(rotateY(yaw), rotateX(pitch));
    Mat4 view  = translation(0.0f, 0.0f, -3.0f);
    Mat4 proj  = perspective(60.0f * std::numbers::pi_v<float> / 180.0f,
                             aspect, 0.1f, 100.0f);

    glUseProgram(m_program);
    glUniformMatrix4fv(m_uModel, 1, GL_FALSE, model.m);
    glUniformMatrix4fv(m_uView, 1, GL_FALSE, view.m);
    glUniformMatrix4fv(m_uProj, 1, GL_FALSE, proj.m);

    // Light direction in world space (pointing toward the light).
    glUniform3f(m_uLight, 0.4f, 0.8f, 0.5f);
    glUniform3f(m_uColor, 0.85f, 0.45f, 0.25f);
    glUniform3f(m_uCam, 0.0f, 0.0f, 3.0f);

    glBindVertexArray(m_vao);
    glDrawElements(GL_TRIANGLES, m_indexCount, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}
