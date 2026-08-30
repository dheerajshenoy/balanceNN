#pragma once

#include <SDL3/SDL.h>

#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

struct Vec3
{
    float x, y, z;
};

struct Vec2
{
    float x, y;
};

// Mesh: vertices + edges (index pairs). Wireframe only.
struct Mesh
{
    std::vector<Vec3> vertices;
    std::vector<std::pair<int, int>> edges;
};

inline Vec3
rotateY(Vec3 v, float a)
{
    float c = std::cos(a), s = std::sin(a);
    return {c * v.x + s * v.z, v.y, -s * v.x + c * v.z};
}

inline Vec3
rotateX(Vec3 v, float a)
{
    float c = std::cos(a), s = std::sin(a);
    return {v.x, c * v.y - s * v.z, s * v.y + c * v.z};
}

// Perspective-project a camera-space point to screen coords.
// fov: vertical field of view (radians). cx/cy: screen center. dist: focal
// length in pixels for the given fov and screen height.
inline Vec2
project(Vec3 v, float cx, float cy, float focal, float camZ)
{
    float z = v.z + camZ;
    if (z < 0.01f)
        z = 0.01f;
    return {cx + (v.x * focal) / z, cy - (v.y * focal) / z};
}

inline Mesh
makeCube(float size = 1.0f)
{
    float h = size * 0.5f;
    Mesh m;
    m.vertices = {
        {-h, -h, -h}, { h, -h, -h}, { h,  h, -h}, {-h,  h, -h},
        {-h, -h,  h}, { h, -h,  h}, { h,  h,  h}, {-h,  h,  h},
    };
    m.edges = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };
    return m;
}

// UV-sphere wireframe: `stacks` horizontal rings, `slices` vertical meridians.
inline Mesh
makeSphere(float radius = 1.0f, int stacks = 12, int slices = 16)
{
    Mesh m;
    constexpr float pi = std::numbers::pi_v<float>;

    // Vertices: (stacks+1) rings × slices verts (poles duplicated per slice for
    // index simplicity).
    for (int i = 0; i <= stacks; ++i)
    {
        float v     = static_cast<float>(i) / stacks;
        float phi   = v * pi;               // 0 .. pi
        float y     = radius * std::cos(phi);
        float rSin  = radius * std::sin(phi);
        for (int j = 0; j < slices; ++j)
        {
            float u     = static_cast<float>(j) / slices;
            float theta = u * 2.0f * pi;    // 0 .. 2pi
            m.vertices.push_back(
                {rSin * std::cos(theta), y, rSin * std::sin(theta)});
        }
    }

    auto idx = [slices](int i, int j) { return i * slices + (j % slices); };

    // Parallels (horizontal rings).
    for (int i = 1; i < stacks; ++i)
        for (int j = 0; j < slices; ++j)
            m.edges.emplace_back(idx(i, j), idx(i, j + 1));

    // Meridians (vertical arcs).
    for (int j = 0; j < slices; ++j)
        for (int i = 0; i < stacks; ++i)
            m.edges.emplace_back(idx(i, j), idx(i + 1, j));

    return m;
}

// Draw a wireframe mesh, applying rotation and translation, then projecting.
inline void
drawWireframe(SDL_Renderer *renderer, const Mesh &mesh, Vec3 translate,
              float rotXAngle, float rotYAngle, float cx, float cy,
              float focal, float camZ)
{
    std::vector<Vec2> projected;
    projected.reserve(mesh.vertices.size());
    for (const Vec3 &v : mesh.vertices)
    {
        Vec3 r = rotateX(rotateY(v, rotYAngle), rotXAngle);
        r.x += translate.x;
        r.y += translate.y;
        r.z += translate.z;
        projected.push_back(project(r, cx, cy, focal, camZ));
    }
    for (const auto &[a, b] : mesh.edges)
        SDL_RenderLine(renderer, projected[a].x, projected[a].y,
                       projected[b].x, projected[b].y);
}
