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


struct PoseSolution {
    Eigen::Vector3d T;  // 矩形中心在 base 坐标下的平移向量
    double          yaw;    // 求得的 alpha (rad)
    double          reproj; // 重投影代价: sum_i (1 - cos(angle_i))
                            // 约等于 0.5 * sum(angle_i^2), 单位 rad^2
};


// ---------- SVD 求矩形中心方向 ON ----------
inline Eigen::Vector3d solveCenterDirection(const std::array<Eigen::Vector3d,4>& v)
{
    Eigen::Matrix<double,3,4> M;
    M.col(0) =  v[0].normalized();
    M.col(1) = -v[1].normalized();
    M.col(2) =  v[2].normalized();
    M.col(3) = -v[3].normalized();

    Eigen::JacobiSVD<Eigen::Matrix<double,3,4>>
        svd(M, Eigen::ComputeFullV);
    Eigen::Vector4d dprime = svd.matrixV().col(3); // 最小奇异值对应列

    if (dprime.sum() < 0) dprime = -dprime;        // 保证深度为正

    Eigen::Vector3d Tprime = 0.25 * ( dprime[0] * v[0] + dprime[1] * v[1] + dprime[2] * v[2] + dprime[3] * v[3] );

    return Tprime.normalized();
}

inline std::array< PoseSolution, 2>
solveRectanglePose(const std::array<Eigen::Vector3d,4>& v_in,
                   double beta, double gamma,
                   double W, double H,double accepted_reproj_cost = 5e-4) 
{
    Eigen::Vector3d center_cam = solveCenterDirection(v_in);
}
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
    return { 0.5 * (left + right), std::min(ml_cost, mr_cost)};
}

template<typename Func>
inline std::pair<double,double>
linearEnumMin(double left, double right, Func&& cost, size_t step_num)
{
    if (left > right) {
        std::swap(left, right);
    }

    step_num = std::max<std::size_t>(1, step_num);

    const double step = (right - left) / static_cast<double>(step_num);

    double best_x = left;
    double best_cost = std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i <= step_num; ++i) {
        double x = (i == step_num) ? right : left + step * static_cast<double>(i);
        double y = cost(x);

        if (y < best_cost) {
            best_cost = y;
            best_x = x;
        }
    }

    return {best_x, best_cost};
}


template<typename Func>
inline std::pair<double, double>
linearEnumMax(double left, double right, Func&& cost, std::size_t step_num)
{
    if (left > right) {
        std::swap(left, right);
    }

    step_num = std::max<std::size_t>(1, step_num);

    const double step = (right - left) / static_cast<double>(step_num);

    double best_x = left;
    double best_cost = -std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i <= step_num; ++i) {
        const double x = (i == step_num)
            ? right
            : left + step * static_cast<double>(i);

        const double y = cost(x);

        if (y > best_cost) {
            best_cost = y;
            best_x = x;
        }
    }

    return {best_x, best_cost};
}

} // namespace pose

