// Sparse4D detection-only decoder. Source contract: repo commit 0e30fe5a.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace sparse4d {
inline constexpr std::array<const char*, 13> kClasses{{
    "barrier", "bus", "car", "construction_vehicle", "motor", "bicycle",
    "animal", "pedestrian", "traffic_cone", "stopper", "trash_bin", "sign", "truck"}};
struct Detection {
    size_t query = 0;
    int label = 0;
    float score = 0;
    std::array<double, 7> box{};  // x,y,z,w,l,h,yaw; lidar frame, box center.
};
inline float sigmoid(float x) {
    if (x >= 0) return 1.f / (1.f + std::exp(-x));
    float e = std::exp(x);
    return e / (1.f + e);
}
inline std::vector<Detection> decode(const std::vector<float>& logits,
                                     const std::vector<float>& boxes,
                                     int top_k = 300, float threshold = .5f) {
    if (top_k <= 0 || threshold < 0 || threshold > 1 ||
        logits.empty() || logits.size() % 13 || boxes.size() != logits.size()/13*8)
        throw std::runtime_error("Invalid decoder shape/parameters: expected Q*13 and Q*8.");
    struct Candidate { float score; size_t index; };
    std::vector<Candidate> candidates;
    candidates.reserve(logits.size());
    for (size_t i=0; i<logits.size(); ++i) {
        if (!std::isfinite(logits[i]))
            throw std::runtime_error("Non-finite classification output.");
        candidates.push_back({sigmoid(logits[i]), i});
    }
    size_t k = std::min(static_cast<size_t>(top_k), candidates.size());
    // Tie order is deterministic here; torch.topk does not promise stable ties.
    std::partial_sort(candidates.begin(), candidates.begin()+k, candidates.end(),
                      [](const Candidate& a, const Candidate& b) {
                          return a.score == b.score ? a.index < b.index : a.score > b.score;
                      });
    std::vector<Detection> result;
    for (size_t i=0; i<k; ++i) {
        const auto& c = candidates[i];
        if (c.score < threshold) continue;
        Detection d;
        d.query = c.index / 13; d.label = static_cast<int>(c.index % 13); d.score = c.score;
        const float* b = boxes.data()+d.query*8;
        for (int j=0; j<8; ++j)
            if (!std::isfinite(b[j])) throw std::runtime_error("Non-finite box output.");
        d.box = {{b[0], b[1], b[2], std::exp(double(b[3])),
                  std::exp(double(b[4])), std::exp(double(b[5])), std::atan2(b[6],b[7])}};
        for (double v : d.box)
            if (!std::isfinite(v)) throw std::runtime_error("Box dimension overflow.");
        result.push_back(d);
    }
    return result;
}
using Point3 = std::array<double, 3>;
inline std::array<Point3,8> corners(const Detection& d) {
    constexpr int bits[8][3] = {{0,0,0},{0,0,1},{0,1,1},{0,1,0},
                               {1,0,0},{1,0,1},{1,1,1},{1,1,0}};
    std::array<Point3,8> result{};
    double c=std::cos(d.box[6]), s=std::sin(d.box[6]);
    for (int i=0;i<8;++i) {
        double x=(bits[i][0]-.5)*d.box[3], y=(bits[i][1]-.5)*d.box[4];
        result[i]={{x*c-y*s+d.box[0], x*s+y*c+d.box[1],
                    (bits[i][2]-.5)*d.box[5]+d.box[2]}};
    }
    return result;
}
inline constexpr int kEdges[12][2] = {
    {0,1},{0,3},{0,4},{1,2},{1,5},{3,2},{3,7},{4,5},{4,7},{2,6},{5,6},{6,7}};
inline std::array<double,2> fisheye(const Point3& p,
                                    const std::array<double,9>& k,
                                    const std::array<double,4>& distortion) {
    // Matches tool/visualization/utils.py, including the Z epsilon.
    double theta=std::atan2(std::hypot(p[0],p[1]),p[2]+1e-6);
    double phi=std::atan2(p[1],p[0]), t2=theta*theta;
    double td=theta*(1+t2*(distortion[0]+t2*(distortion[1]+t2*(distortion[2]+t2*distortion[3]))));
    return {{k[2]+k[0]*td*std::cos(phi), k[5]+k[4]*td*std::sin(phi)}};
}
} // namespace sparse4d

