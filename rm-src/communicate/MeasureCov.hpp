#pragma once
#include <array>
#include <eigen3/Eigen/Core>

// Stub: MeasureCov 用于 RerunVisualizer::viewCov()，主程序未调用
struct MeasureCov {
    std::array<double, 4> operator()(const Eigen::Matrix<double, 4, 1>&) {
        return {0.0, 0.0, 0.0, 0.0};
    }
};
