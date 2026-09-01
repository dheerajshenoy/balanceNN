#pragma once

#include <vector>

// Column-major 4x4. Layout: m[col*4 + row].
struct Mat4
{
    float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};

Mat4 mul(const Mat4 &a, const Mat4 &b);
Mat4 perspective(float fovY, float aspect, float zNear, float zFar);
Mat4 translation(float x, float y, float z);
Mat4 rotateX(float a);
Mat4 rotateY(float a);
Mat4 rotateZ(float a);
Mat4 scale(float s);
Mat4 scale(float sx, float sy, float sz);

struct Mesh
{
    unsigned int vao = 0;
    unsigned int vbo = 0;
    unsigned int ebo = 0;
    int indexCount   = 0;
};

// Upload interleaved (pos3, normal3) verts + indices to a fresh mesh.
void uploadMesh(Mesh &m, const std::vector<float> &verts,
                const std::vector<unsigned int> &indices);
void destroyMesh(Mesh &m);

// Shared Phong-lit program used by Plate and Sphere.
class PhongProgram
{
public:
    ~PhongProgram();
    bool init();

    void use() const;
    void setView(const Mat4 &v) const;
    void setProj(const Mat4 &p) const;
    void setLightDir(float x, float y, float z) const;
    void setCameraPos(float x, float y, float z) const;

    void drawMesh(const Mesh &m, const Mat4 &model, float r, float g,
                  float b) const;

    unsigned int id() const { return m_program; }

private:
    unsigned int m_program = 0;
    int m_uModel           = -1;
    int m_uView            = -1;
    int m_uProj            = -1;
    int m_uLight           = -1;
    int m_uColor           = -1;
    int m_uCam             = -1;
};
