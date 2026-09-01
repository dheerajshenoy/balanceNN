#pragma once

#include <cstddef>
#include <vector>

// Online mean/variance tracker for observation normalization.
// Uses Welford's algorithm (numerically stable — the naive
// sum-of-squares-then-divide approach loses precision as counts grow).
//
// Usage: call observe() with each raw observation seen during rollout
// collection (before feeding into the network), then normalize() any obs
// before inference. The stats keep updating forever; over time the
// distribution stabilizes.
//
// Not torch-dependent: pure C++ so it's unit-testable without libtorch.
class ObsNormalizer
{
public:
    explicit ObsNormalizer(std::size_t dim);

    std::size_t dim() const { return m_dim; }
    long long count() const { return m_count; }

    // Update running mean/variance with one observation of length dim().
    void observe(const float *obs);

    // In-place normalize: obs[i] = (obs[i] - mean[i]) / (std[i] + eps).
    // Safe to call before any observe(): with count()==0, returns obs
    // unchanged (mean=0, std=1).
    void normalize(float *obs) const;

    // Non-mutating accessors for saving / debugging.
    const std::vector<double> &mean() const { return m_mean; }
    // Per-dim standard deviation (recomputed from M2 each call — call
    // sparingly).
    std::vector<double> stddev() const;

private:
    std::size_t m_dim;
    long long m_count = 0;
    std::vector<double> m_mean; // running mean per dim
    std::vector<double> m_M2;   // sum of squared deviations per dim
};
