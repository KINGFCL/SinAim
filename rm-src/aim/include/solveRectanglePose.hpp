#pragma once

#include <Eigen/src/Core/Matrix.h>
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
inline Eigen::Vector3d solveCenterDirection(const Eigen::Matrix<double,3,4>& v)
{
    Eigen::Matrix<double,3,4> M;
    M.col(0) =  v.col(0).normalized();
    M.col(1) = -v.col(1).normalized();
    M.col(2) =  v.col(2).normalized();
    M.col(3) = -v.col(3).normalized();

    Eigen::JacobiSVD<Eigen::Matrix<double,3,4>>
        svd(M, Eigen::ComputeFullV);
    Eigen::Vector4d dprime = svd.matrixV().col(3); // 最小奇异值对应列

    if (dprime.sum() < 0) dprime = -dprime;        // 保证深度为正

    Eigen::Vector3d Tprime = 0.25 * ( dprime[0] * M.col(0) - dprime[1] * M.col(1) + dprime[2] * M.col(2) - dprime[3] * M.col(3) );

    return Tprime.normalized();
}


inline Eigen::Matrix3d EulerPitchRoll(double pitch, double roll)
{
    const double cp = std::cos(pitch);
    const double sp = std::sin(pitch);
    const double cr = std::cos(roll);
    const double sr = std::sin(roll);

    return Eigen::Matrix3d{
        { cp, sp * sr, sp * cr  },
        { 0.0,   cr,     -sr    },
        { -sp, cp * sr, cp * cr }
    };
}

inline Eigen::Matrix3d EulerYawPitchRoll(const Eigen::Matrix3d& R_raw, double yaw) {
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);

    // 直接构造并返回，避免拷贝后修改
    return Eigen::Matrix3d{
        {cy * R_raw(0, 0),  cy * R_raw(0, 1) - sy * R_raw(1, 1),  cy * R_raw(0, 2) - sy * R_raw(1, 2)},
        {sy * R_raw(0, 0),  sy * R_raw(0, 1) + cy * R_raw(1, 1),  sy * R_raw(0, 2) + cy * R_raw(1, 2)},
        {R_raw(2, 0),       R_raw(2, 1),                          R_raw(2, 2)}
    };
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



inline std::array< PoseSolution, 2>
solveRectanglePose(const Eigen::Matrix<double,3,4>& v_in,
                   const Eigen::Matrix3d& R_cam2base,
                   double pitch, double roll,
                   double W, double H,double accepted_reproj_cost = 5e-4) 
{
    Eigen::Vector3d center_cam = solveCenterDirection(v_in);
    const Eigen::Vector3d T_cam = 200.0 * center_cam;//假设在200cm处
    const Eigen::Matrix<double,3,4> P {
            {0.0,   0.0,    0.0,    0.0},
            {W*0.5, -W*0.5, -W*0.5, W*0.5},
            {H*0.5, H*0.5,  -H*0.5, -H*0.5}
    };
    const Eigen::Matrix3d R_raw = EulerPitchRoll(pitch, roll);

    const Eigen::Matrix<double,2,4> v_real = v_in.block<2,4>(0,0);

    auto yawcost = [&](double yaw) -> double {
        const Eigen::Matrix3d R_world2base = EulerYawPitchRoll(R_raw, yaw);
        const Eigen::Matrix3d R_world2cam = R_cam2base * R_world2base;
                
        Eigen::Matrix<double,3,4> P_cam = R_world2cam * P;
        
        P_cam.col(0) += T_cam;
        P_cam.col(1) += T_cam;
        P_cam.col(2) += T_cam;
        P_cam.col(3) += T_cam;

        double z0 = 1.0 / P_cam(2,0), z1 = 1.0 / P_cam(2,1), z2 = 1.0 / P_cam(2,2), z3 = 1.0 / P_cam(2,3);
        
        Eigen::Matrix<double,2,4> P_idea;
        P_idea.col(0) = z0 * P_cam.block<2,1>(0,0);
        P_idea.col(1) = z1 * P_cam.block<2,1>(0,1);
        P_idea.col(2) = z2 * P_cam.block<2,1>(0,2);
        P_idea.col(3) = z3 * P_cam.block<2,1>(0,3);

        //计算cost:
        Eigen::Vector2d M = P_idea.rowwise().mean(); // 计算 idea 的质心

        // 广播减法，求去质心后的向量
        Eigen::Matrix<double, 2, 4> U = v_real.colwise() - M;
        Eigen::Matrix<double, 2, 4> V = P_idea.colwise() - M;

        // 计算点积和与范数平方和，求最优缩放系数 s
        double sum_UV = U.cwiseProduct(V).sum();
        double norm_V2 = V.squaredNorm();
        double s = sum_UV / norm_V2;

        // 计算最终的对齐误差 E(s)
        // 直接根据定义计算残差矩阵 (U - s*V) 的范数平方
        return (U - s * V).squaredNorm();
    };

    std::pair<double, double> ans_left = linearEnumMin( )
}

} // namespace pose

