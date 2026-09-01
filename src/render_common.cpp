#include "render_common.hpp"

#include <glad/gl.h>

#include <cmath>
#include <cstdio>

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

} // namespace

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
rotateZ(float a)
{
    float c = std::cos(a), s = std::sin(a);
    Mat4 m;
    m.m[0] = c;
    m.m[1] = s;
    m.m[4] = -s;
    m.m[5] = c;
    return m;
}

Mat4
scale(float s)
{
    return scale(s, s, s);
}

Mat4
scale(float sx, float sy, float sz)
{
    Mat4 m;
    m.m[0]  = sx;
    m.m[5]  = sy;
    m.m[10] = sz;
    return m;
}

void
uploadMesh(Mesh &m, const std::vector<float> &verts,
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

void
destroyMesh(Mesh &m)
{
    if (m.ebo)
        glDeleteBuffers(1, &m.ebo);
    if (m.vbo)
        glDeleteBuffers(1, &m.vbo);
    if (m.vao)
        glDeleteVertexArrays(1, &m.vao);
    m = {};
}

PhongProgram::~PhongProgram()
{
    if (m_program)
        glDeleteProgram(m_program);
}

bool
PhongProgram::init()
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
    return true;
}

void
PhongProgram::use() const
{
    glUseProgram(m_program);
}

void
PhongProgram::setView(const Mat4 &v) const
{
    glUniformMatrix4fv(m_uView, 1, GL_FALSE, v.m);
}

void
PhongProgram::setProj(const Mat4 &p) const
{
    glUniformMatrix4fv(m_uProj, 1, GL_FALSE, p.m);
}

void
PhongProgram::setLightDir(float x, float y, float z) const
{
    glUniform3f(m_uLight, x, y, z);
}

void
PhongProgram::setCameraPos(float x, float y, float z) const
{
    glUniform3f(m_uCam, x, y, z);
}

void
PhongProgram::drawMesh(const Mesh &m, const Mat4 &model, float r, float g,
                       float b) const
{
    glUniformMatrix4fv(m_uModel, 1, GL_FALSE, model.m);
    glUniform3f(m_uColor, r, g, b);
    glBindVertexArray(m.vao);
    glDrawElements(GL_TRIANGLES, m.indexCount, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}
