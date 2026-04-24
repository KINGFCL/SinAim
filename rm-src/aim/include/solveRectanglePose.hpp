#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace pose {

// ---------- 黄金分割法最小化 ----------
template<typename Func>
inline std::pair<double, double>
goldenSectionMin(double left, double right, Func&& cost, int iters = 16)
{
    const double phi = 0.6180339887498949025257388711906969547271728515625;
    
    // 初始化两个探测点
    double ml = left + (right - left) * (1.0 - phi);
    double mr = left + (right - left) * phi;
    
    double ml_cost = cost(ml);
    double mr_cost = cost(mr);

    for (int i = 0; i < iters; ++i) {
        if (ml_cost < mr_cost) {
            right = mr;          // 丢弃右侧区间
            mr = ml;             // 原来的左探测点，变成了新区间的右探测点 (坐标直接复用，无浮点漂移)
            mr_cost = ml_cost;   // 连同代价一起复用
            
            // 唯一需要计算的是新的左探测点
            ml = left + (right - left) * (1.0 - phi);
            ml_cost = cost(ml);
        } else {
            left = ml;           // 丢弃左侧区间
            ml = mr;             // 原来的右探测点，变成了新区间的左探测点
            ml_cost = mr_cost;
            
            // 唯一需要计算的是新的右探测点
            mr = left + (right - left) * phi;
            mr_cost = cost(mr);
        }
    }
    return { 0.5 * (left + right), right - left };
}

} // namespace pose

