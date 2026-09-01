#include "obs_normalizer.hpp"

#include <cmath>
#include <fstream>

ObsNormalizer::ObsNormalizer(std::size_t dim)
    : m_dim(dim), m_mean(dim, 0.0), m_M2(dim, 0.0)
{
}

void
ObsNormalizer::observe(const float *obs)
{
    ++m_count;
    for (std::size_t i = 0; i < m_dim; ++i)
    {
        double x     = static_cast<double>(obs[i]);
        double delta = x - m_mean[i];
        m_mean[i] += delta / static_cast<double>(m_count);
        double delta2 = x - m_mean[i];
        m_M2[i] += delta * delta2;
    }
}

void
ObsNormalizer::normalize(float *obs) const
{
    if (m_count < 2)
        return; // no variance estimate yet
    constexpr double eps = 1e-8;
    double invN          = 1.0 / static_cast<double>(m_count - 1);
    for (std::size_t i = 0; i < m_dim; ++i)
    {
        double var = m_M2[i] * invN;
        double sd  = std::sqrt(var) + eps;
        obs[i]     = static_cast<float>((static_cast<double>(obs[i]) - m_mean[i])
                                    / sd);
    }
}

std::vector<double>
ObsNormalizer::stddev() const
{
    std::vector<double> out(m_dim, 1.0);
    if (m_count < 2)
        return out;
    double invN = 1.0 / static_cast<double>(m_count - 1);
    for (std::size_t i = 0; i < m_dim; ++i)
        out[i] = std::sqrt(m_M2[i] * invN);
    return out;
}

bool
ObsNormalizer::save(const std::string &path) const
{
    std::ofstream f(path);
    if (!f)
        return false;
    f << m_dim << ' ' << m_count << '\n';
    f.precision(17);
    for (double v : m_mean)
        f << v << ' ';
    f << '\n';
    for (double v : m_M2)
        f << v << ' ';
    f << '\n';
    return static_cast<bool>(f);
}

bool
ObsNormalizer::load(const std::string &path)
{
    std::ifstream f(path);
    if (!f)
        return false;
    std::size_t dim = 0;
    long long count = 0;
    f >> dim >> count;
    if (!f || dim != m_dim)
        return false;
    std::vector<double> mean(dim), m2(dim);
    for (std::size_t i = 0; i < dim; ++i)
        f >> mean[i];
    for (std::size_t i = 0; i < dim; ++i)
        f >> m2[i];
    if (!f)
        return false;
    m_count = count;
    m_mean  = std::move(mean);
    m_M2    = std::move(m2);
    return true;
}
