#pragma once

#include "render_common.hpp"

// A rectangular plate that can be tilted about world X and Z axes.
// Owns its mesh; must be constructed after a GL context is current.
class Plate
{
public:
    Plate(float width = 4.0f, float height = 0.2f, float depth = 4.0f);
    ~Plate();

    Plate(const Plate &)            = delete;
    Plate &operator=(const Plate &) = delete;

    // Tilt in radians. tiltX rotates about world X; tiltZ about world Z.
    void setTilt(float tiltX, float tiltZ)
    {
        m_tiltX = tiltX;
        m_tiltZ = tiltZ;
    }
    float tiltX() const { return m_tiltX; }
    float tiltZ() const { return m_tiltZ; }

    float width() const { return m_width; }
    float height() const { return m_height; }
    float depth() const { return m_depth; }
    float halfWidth() const { return m_width * 0.5f; }
    float halfDepth() const { return m_depth * 0.5f; }
    float topY() const { return m_height * 0.5f; }

    // Transform from plate-local space to world (tilt only — no scene rot).
    Mat4 tiltMatrix() const;

    // Draw with `parent` prepended (e.g., camera-orbit scene rotation).
    // Default color matches the plate; override for markers that share the
    // Plate mesh shape but need to stand out.
    void draw(const PhongProgram &prog, const Mat4 &parent,
              float r = 0.35f, float g = 0.4f, float b = 0.5f) const;

private:
    float m_width, m_height, m_depth;
    float m_tiltX = 0.0f;
    float m_tiltZ = 0.0f;
    Mesh m_mesh;
};
