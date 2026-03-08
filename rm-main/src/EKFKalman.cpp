#include "../include/EKFKalman.hpp"
#include <cmath>
#include "opencv2/core/cvdef.h"

void EKFKalman::Init()
{
    this->CovState = this->CovStateInit;
}

// ---------------------------------------------------------
// 单装甲板更新
// ---------------------------------------------------------
Eigen::Matrix<double, 11, 1> EKFKalman::operator()(
    const Eigen::Matrix<double, 11, 1>& State,
    const Eigen::Matrix<double, 4, 1>& View, 
    int armor_id, 
    double dt)
{
    // 调用多板更新的逻辑即可，避免代码重复
    std::vector<Eigen::Matrix<double, 4, 1>> Views = {View};
    std::vector<int> armor_ids = {armor_id};
    return this->operator()(State, Views, armor_ids, dt);
}

// ---------------------------------------------------------
// 多装甲板序贯更新核心算法
// ---------------------------------------------------------
Eigen::Matrix<double, 11, 1> EKFKalman::operator()(
    const Eigen::Matrix<double, 11, 1>& State,
    const std::vector<Eigen::Matrix<double, 4, 1>>& Views, 
    const std::vector<int>& armor_ids, 
    double dt)
{
    // ==========================================
    // 1. 预测阶段 (Predict) - 纯线性匀速模型
    // ==========================================
    double vxc = State(3), vyc = State(4), vzc = State(5), w = State(7);

    Eigen::Matrix<double, 11, 1> X_curr = State;
    X_curr(0) += vxc * dt;
    X_curr(1) += vyc * dt;
    X_curr(2) += vzc * dt;
    X_curr(6) += w * dt;
    X_curr(6) = std::remainder(X_curr(6), CV_PI * 2.0);

    // 状态转移雅可比 F
    Eigen::Matrix<double, 11, 11> F = Eigen::Matrix<double, 11, 11>::Identity();
    F(0, 3) = dt; F(1, 4) = dt; F(2, 5) = dt; F(6, 7) = dt;

    // 过程噪声 Q
    Eigen::Matrix<double, 11, 11> Q = Eigen::Matrix<double, 11, 11>::Zero();
    // 预先计算好关于 dt 的时间项
    auto a = dt * dt * dt * dt * 0.25;  // dt^4 / 4
    auto b = dt * dt * dt * 0.5;       // dt^3 / 2
    auto c = dt * dt;                // dt^2
    Q(0,0)=Q(1,1)=Q(2,2) = a * this->Var_a;
    Q(3,3)=Q(4,4)=Q(5,5) = c * this->Var_a;
    Q(6,6) = a * this->Var_alpha;
    Q(7,7) = c * this->Var_alpha;
    Q(8,8) = this->Var_r; 
    Q(9,9) = this->Var_l; 
    Q(10,10) = this->Var_h; 

    Q(0,3) = Q(3,0) = Q(1,4) = Q(4,1) = Q(2,5) = Q(5,2) = b * this->Var_a;     // X 与 Vx 绑定
    Q(6,7) = Q(7,6) = b * this->Var_alpha; // Yaw 与 w 绑定

    this->CovState = F * this->CovState * F.transpose() + Q;

    // ==========================================
    // 2. 序贯更新阶段 (Sequential Update)
    // ==========================================
    for (size_t i = 0; i < Views.size(); ++i) 
    {
        int id = armor_ids[i];
        const auto& View = Views[i];

        double pred_xc = X_curr(0), pred_yc = X_curr(1), pred_zc = X_curr(2);
        double pred_yaw = X_curr(6);
        double pred_r = X_curr(8), pred_l = X_curr(9), pred_h = X_curr(10);

        // 判断当前观测的是否为侧面板 (ID = 1 或 3)
        bool is_side = (id == 1 || id == 3);

        // 获取真实的半径和高度
        double cur_r = is_side ? (pred_r + pred_l) : pred_r;
        double cur_z = is_side ? (pred_zc + pred_h) : pred_zc;

        // 计算当前这块装甲板的理论角度
        double armor_angle = pred_yaw + id * (CV_PI / 2.0);
        armor_angle = std::remainder(armor_angle, CV_PI * 2.0);
        double cos_a = std::cos(armor_angle);
        double sin_a = std::sin(armor_angle);

        // 映射出这块装甲板的预测坐标 h(x)
        Eigen::Matrix<double, 4, 1> Z_predict;
        Z_predict(0) = pred_xc + cur_r * cos_a;
        Z_predict(1) = pred_yc + cur_r * sin_a;
        Z_predict(2) = cur_z;
        Z_predict(3) = armor_angle;

        // 计算雅可比 H
        Eigen::Matrix<double, 4, 11> H = Eigen::Matrix<double, 4, 11>::Zero();
        H(0, 0) = 1.0; 
        H(0, 6) = -cur_r * sin_a; // dx/d(yaw)
        H(0, 8) = cos_a;          // dx/d(r)
        H(0, 9) = is_side ? cos_a : 0.0; // dx/d(l)

        H(1, 1) = 1.0; 
        H(1, 6) = cur_r * cos_a;  // dy/d(yaw)
        H(1, 8) = sin_a;          // dy/d(r)
        H(1, 9) = is_side ? sin_a : 0.0; // dy/d(l)

        H(2, 2) = 1.0;            // dz/d(zc)
        H(2, 10) = is_side ? 1.0 : 0.0;  // dz/d(h)

        H(3, 6) = 1.0;            // d(theta)/d(yaw)

        // 计算残差 Y
        Eigen::Matrix<double, 4, 1> Y = View - Z_predict;
        Y(3) = std::remainder(Y(3), CV_PI * 2.0);

        // 计算卡尔曼增益 K
        Eigen::Matrix<double, 4, 4> S = H * this->CovState * H.transpose() + this->CovView;
        Eigen::Matrix<double, 11, 4> K = this->CovState * H.transpose() * S.inverse();

        // 更新状态向量
        X_curr = X_curr + K * Y;
        X_curr(6) = std::remainder(X_curr(6), CV_PI * 2.0); 

        // 更新协方差矩阵
        Eigen::Matrix<double, 11, 11> I_KH = Eigen::Matrix<double, 11, 11>::Identity() - K * H;
        this->CovState = I_KH * this->CovState * I_KH.transpose() + K * this->CovView * K.transpose();
    }

    return X_curr;
}