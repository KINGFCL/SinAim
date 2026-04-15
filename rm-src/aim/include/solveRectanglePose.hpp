// solveRectanglePose.hpp
// -----------------------------------------------------------------------------
// 单头文件: 已知矩形 4 角点在 A 系下的单位方向向量 (深度未知),
// 以及 B->A 内旋顺序 Z->X->Y 中 beta(X), gamma(Y) 已知, 求解:
//   - B->A 旋转矩阵 R_B2A
//   - 矩形中心在 A 下的平移 t_B2A
//   - 重投影误差 reproj (供外部按目标尺寸/形状筛选)
// 并针对远距离 Yaw 歧义, 返回 (顺时针 yaw, 逆时针 yaw) 两组候选解.
// -----------------------------------------------------------------------------
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
    Eigen::Matrix3d R_B2A;  // B -> A 旋转矩阵
    Eigen::Vector3d t_B2A;  // 矩形中心在 A 下的平移
    double          yaw;    // 求得的 alpha (rad)
    double          reproj; // 重投影代价: sum_i (1 - cos(angle_i))
                            // 约等于 0.5 * sum(angle_i^2), 单位 rad^2
};

// ---------- 基本旋转 ----------
inline Eigen::Matrix3d Rx(double a) {
    Eigen::Matrix3d R; double c = std::cos(a), s = std::sin(a);
    R << 1, 0, 0,
         0, c,-s,
         0, s, c;
    return R;
}
inline Eigen::Matrix3d Ry(double a) {
    Eigen::Matrix3d R; double c = std::cos(a), s = std::sin(a);
    R << c, 0, s,
         0, 1, 0,
        -s, 0, c;
    return R;
}
inline Eigen::Matrix3d Rz(double a) {
    Eigen::Matrix3d R; double c = std::cos(a), s = std::sin(a);
    R << c,-s, 0,
         s, c, 0,
         0, 0, 1;
    return R;
}

// 旧接口 (兼容外部直接调用): R_{B->A} = (Ry(gamma) Rx(beta) Rz(alpha))^T
inline Eigen::Matrix3d R_B2A_func(double alpha, double beta, double gamma) {
    return (Ry(gamma) * Rx(beta) * Rz(alpha)).transpose();
}

// ---------- 性能优化: 预计算固定部分 ----------
// 注意: R_{B->A} = Rz(alpha)^T * Rx(beta)^T * Ry(gamma)^T
//                = Rz(-alpha)  * (Ry(gamma) * Rx(beta))^T
// 把 (Ry(gamma) * Rx(beta))^T 预计算成 Ryx_T, 搜索循环里只算 Rz(-alpha).
// 每次 reprojCost 调用省下 2 次三角函数 + 2 次 3x3 矩阵乘法.
inline Eigen::Matrix3d precomputeRyxT(double beta, double gamma) {
    return (Ry(gamma) * Rx(beta)).transpose();
}

inline Eigen::Matrix3d R_B2A_fast(double alpha, const Eigen::Matrix3d& Ryx_T) {
    // Rz(alpha)^T == Rz(-alpha), 但直接构造转置版本更省一次取负
    Eigen::Matrix3d RzT;
    double c = std::cos(alpha), s = std::sin(alpha);
    // Rz(alpha)^T:
    //   [ c,  s, 0]
    //   [-s,  c, 0]
    //   [ 0,  0, 1]
    RzT <<  c,  s, 0,
           -s,  c, 0,
            0,  0, 1;
    return RzT * Ryx_T;
}

// =============================================================================
//  角点顺序约定 (务必严格遵守, 否则 SVD 与正交残差都会算错!)
// -----------------------------------------------------------------------------
//  v_in[0..3] 必须是矩形四个顶点 "按顺时针或逆时针" 依次排列的
//  单位方向向量 (A 系下), 例如:
//        v_in[0] = 左上
//        v_in[1] = 右上
//        v_in[2] = 右下
//        v_in[3] = 左下
//  这样才能同时满足:
//    (a) SVD 中 M = [v0, -v1, v2, -v3] 要求 0-2 / 1-3 为对角;
//    (b) 重投影代价里 B 系角点顺序 (LT,RT,RB,LB) 与之一一对应.
//  若外部传入顺序错乱, 代码不会报错, 但结果将完全错误.
// =============================================================================

// ---------- SVD 求矩形中心方向 ON ----------
inline Eigen::Vector3d solveCenterDirection(const std::array<Eigen::Vector3d,4>& v)
{
    Eigen::Matrix<double,3,4> M;
    M.col(0) =  v[0];
    M.col(1) = -v[1];
    M.col(2) =  v[2];
    M.col(3) = -v[3];

    Eigen::JacobiSVD<Eigen::Matrix<double,3,4>>
        svd(M, Eigen::ComputeFullV);
    Eigen::Vector4d dprime = svd.matrixV().col(3); // 最小奇异值对应列

    if (dprime.sum() < 0) dprime = -dprime;        // 保证深度为正

    Eigen::Vector3d Tprime = Eigen::Vector3d::Zero();
    for (int i = 0; i < 4; ++i) Tprime += dprime[i] * v[i];
    Tprime *= 0.25;

    return Tprime.normalized();
}

// ---------- 重投影角度代价 (使用预计算 Ryx_T) ----------
// 同时输出该 alpha 对应的 (R, t), 避免外层重复计算.
inline double reprojCostFast(double alpha,
                             const Eigen::Matrix3d& Ryx_T,
                             const Eigen::Vector3d& ON,
                             const std::array<Eigen::Vector3d,4>& v,
                             double W, double H,
                             Eigen::Matrix3d* RBA_out = nullptr,
                             Eigen::Vector3d* t_out = nullptr)
{
    Eigen::Matrix3d RBA = R_B2A_fast(alpha, Ryx_T);
    Eigen::Vector3d nA  = RBA.col(0);
    double denomConst = nA.dot(ON);
    if (std::abs(denomConst) < 1e-12) return 1e18;

    // 1) 射线 ↔ 平面求交
    std::array<Eigen::Vector3d,4> P;
    for (int i = 0; i < 4; ++i) {
        double d = nA.dot(v[i]);
        if (std::abs(d) < 1e-9) return 1e18;
        double k = denomConst / d;
        if (k <= 0) return 1e18;          // 必须在相机前方
        P[i] = k * v[i];
    }

    // 2) 用对角线恢复尺度
    double Lfake = (P[0] - P[2]).norm();
    double Ltrue = std::sqrt(W*W + H*H);
    if (Lfake < 1e-12) return 1e18;
    double S = Ltrue / Lfake;
    Eigen::Vector3d t = S * ON;

    // 3) 用 B 系理想角点重投影回 A 系比角度
    const std::array<Eigen::Vector3d,4> cB = {
        Eigen::Vector3d(0.0,  W/2,  H/2),
        Eigen::Vector3d(0.0, -W/2,  H/2),
        Eigen::Vector3d(0.0, -W/2, -H/2),
        Eigen::Vector3d(0.0,  W/2, -H/2)
    };
    double cost = 0.0;
    for (int i = 0; i < 4; ++i) {
        Eigen::Vector3d pred = (RBA * cB[i] + t).normalized();
        double c = pred.dot(v[i]);
        if (c > 1.0) c = 1.0;
        cost += (1.0 - c);
    }

    if (RBA_out) *RBA_out = RBA;
    if (t_out)   *t_out   = t;
    return cost;
}

// ---------- 黄金分割法最小化 ----------
template<typename Func>
inline std::pair<double,double>
trisectionMin(double left, double right, Func&& f, int iters = 16)
{
    const double phi = (std::sqrt(5.0) - 1.0) / 2.0;
    double ml_cost = 0.0, mr_cost = 0.0;
    int reserved = -1;
    for (int i = 0; i < iters; ++i) {
        double ml = left + (right - left) * (1.0 - phi);
        double mr = left + (right - left) * phi;
        if (reserved != 0) ml_cost = f(ml);
        if (reserved != 1) mr_cost = f(mr);
        if (ml_cost < mr_cost) {
            right = mr; mr_cost = ml_cost; reserved = 1;
        } else {
            left = ml;  ml_cost = mr_cost; reserved = 0;
        }
    }
    return { 0.5 * (left + right), right - left };
}

// ----------------------------------------------------------------------------
//  solveRectanglePose: 通过矩形四个顶点 "顺时针或逆时针" 依次排列的单
//                   位方向向量 (A 系下) 和目标尺寸、形状, 求得目标的 PoseSolution
// ----------------------------------------------------------------------------
//  Returns:
//      - 如果有 2 个解, 顺序为 [顺时针 yaw (yaw < 0), 逆时针 yaw (yaw >= 0)].
//      - 如果只有 1 个解, 直接返回该解。
//      - PoseSolution::reproj 字段为重投影代价, 调用者可据此筛选
//        "是否真的是目标尺寸和形状" (代价过大则不是)。
/*!
    @param v_in: 顺时针或逆时针依次排列的矩形四个顶点在 A 系下的单位方向向量
    @param beta: 目标的 beta 单位：弧度
    @param gamma: 目标的 gamma 单位：弧度
    @param W: 目标的 W
    @param H: 目标的 H
    @return: 按照逆时针，顺时针顺序输出
*/
inline std::vector<PoseSolution>
solveRectanglePose(const std::array<Eigen::Vector3d,4>& v_in,
                   double beta, double gamma,
                   double W, double H,double accepted_reproj_cost = 5e-4) 
{
    std::array<Eigen::Vector3d,4> v;
    for (int i = 0; i < 4; ++i) v[i] = v_in[i].normalized();

    Eigen::Vector3d ON = solveCenterDirection(v);

    // === 性能优化: 预计算固定部分 ===
    Eigen::Matrix3d Ryx_T = precomputeRyxT(beta, gamma);

    const double deg   = M_PI / 180.0;
    const double limit = 80.0 * deg;

    auto cost_fn = [&](double a) {
        return reprojCostFast(a, Ryx_T, ON, v, W, H);
    };

    // 在正/负两个半区间内分别最小化
    double yaw_standard = std::atan2(ON(1), ON(0));
    auto [a_pos, w_pos] = trisectionMin(yaw_standard,    yaw_standard+limit, cost_fn, 16);
    auto [a_neg, w_neg] = trisectionMin(yaw_standard-limit, yaw_standard,   cost_fn, 16);

    // 收尾时一并取出 R, t, 避免再算一次
    Eigen::Matrix3d R_pos, R_neg;
    Eigen::Vector3d t_pos, t_neg;
    double c_pos = reprojCostFast(a_pos, Ryx_T, ON, v, W, H, &R_pos, &t_pos);
    double c_neg = reprojCostFast(a_neg, Ryx_T, ON, v, W, H, &R_neg, &t_neg);

    auto valid = [&](double a, double c, double w_end) {
        return std::isfinite(c) && c < accepted_reproj_cost
            && std::abs(std::remainder(a - w_end, 2.0 * M_PI) ) > 0.5 * deg;   // 远离区间端点
    };

    std::vector<PoseSolution> out;
    auto push = [&](double a, double c,
                    const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
        PoseSolution s;
        s.R_B2A = R; s.t_B2A = t; s.yaw = std::remainder(a, 2.0 * M_PI); s.reproj = c;
        out.push_back(s);
    };

    bool ok_neg = valid(a_neg, c_neg, yaw_standard-limit);
    bool ok_pos = valid(a_pos, c_pos,  yaw_standard+limit);

    //按照逆时针，顺时针顺序输出
    if (ok_neg&&ok_pos)
    {
        push(a_neg, c_neg, R_neg, t_neg);
        push(a_pos, c_pos, R_pos, t_pos);
    } 

    // 排重: 两半区间收敛到同一根 (典型: yaw ≈ 0 的正对情形)
    if (out.size() == 2 &&
        std::abs(std::remainder(out[0].yaw - out[1].yaw, 2.0 * M_PI)) < 0.5 * deg) {
        // 留下重投影误差更小的那个
        if (out[1].reproj < out[0].reproj) out[0] = out[1];
        out.pop_back();
    }
    return out;
}

} // namespace pose
