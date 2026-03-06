#include "../include/EKFKalman.hpp"
#include <cmath>
#include "opencv2/core/cvdef.h"
#include <eigen3/Eigen/src/Core/Matrix.h>
#include "../include/RerunVisualizer.hpp"

#define EKFDebug
#ifdef EKFDebug
extern RerunVisualizer viz;
#endif
void EKFKalman::Init()
{
    this->CovState = this->CovStateInit;
};


Eigen::Matrix<double, 8, 1> EKFKalman::operator()
    (
        const Eigen::Matrix<double, 8, 1>& State,
        const Eigen::Matrix<double, 4, 4>& CovArmor, // 当前装甲板专属的位置协方差
        const Eigen::Matrix<double, 4, 1>& View,     // 当前观测值 Z: [x, y, z, theta]
        double radius,                               // 旋转半径 r
        double dt,                                   // 两次处理的时间差
        double err                                   // (暂未使用)
    )
{
    // ---------------------------------------------------------------------
    // 0. 状态热切换 (State Swapping)
    // ---------------------------------------------------------------------
    // 将当前被观测装甲板的位置协方差注入到 EKF 的状态协方差中
    // 这样既保留了整车共享的速度协方差(右下角4x4)，又更新了当前装甲板的位置置信度(左上角4x4)
    this->CovState.block<4,4>(0,0) = CovArmor;

    // 提取旧状态
    const double &x = State(0), &y = State(1), &z = State(2), &theta = State(3);
    const double &vx = State(4), &vy = State(5), &vz = State(6), &w = State(7);

    // 提前计算三角函数以优化性能
    double theta_new = theta + w * dt;
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    double cos_theta_new = std::cos(theta_new);
    double sin_theta_new = std::sin(theta_new);

    // ---------------------------------------------------------------------
    // 1. 预测阶段 (Predict)
    // ---------------------------------------------------------------------
    
    // 1.1 非线性状态预测 X_{k|k-1} = f(X_{k-1}) (先旋转后平移模型)
    Eigen::Matrix<double, 8, 1> X_predict;
    X_predict(0) = x - radius * cos_theta + radius * cos_theta_new + vx * dt;
    X_predict(1) = y - radius * sin_theta + radius * sin_theta_new + vy * dt;
    X_predict(2) = z + vz * dt;
    X_predict(3) = theta_new;
    X_predict(4) = vx;
    X_predict(5) = vy;
    X_predict(6) = vz;
    X_predict(7) = w;

    // 1.2 计算状态转移雅可比矩阵 F = df/dX
    Eigen::Matrix<double, 8, 8> F = Eigen::Matrix<double, 8, 8>::Identity();
    F(0, 3) =  radius * (sin_theta - sin_theta_new);
    F(0, 4) =  dt;
    F(0, 7) = -radius * dt * sin_theta_new;

    F(1, 3) = -radius * (cos_theta - cos_theta_new);
    F(1, 5) =  dt;
    F(1, 7) =  radius * dt * cos_theta_new;

    F(2, 6) =  dt;
    F(3, 7) =  dt;

    // 1.3 计算过程噪声驱动雅可比矩阵 Gamma (代码中简写为 G)
    Eigen::Matrix<double, 8, 4> G = Eigen::Matrix<double, 8, 4>::Zero();
    
    double dt2_half = 0.5 * dt * dt;

    // 平移加速度对位置的影响
    G(0, 0) = dt2_half;
    G(1, 1) = dt2_half;
    G(2, 2) = dt2_half;
    
    // 角加速度对角度的影响
    G(3, 3) = dt2_half;

    // 【极其关键】：角加速度 alpha 对装甲板 x, y 位置的影响 (雅可比非线性映射)
    G(0, 3) = -radius * dt2_half * sin_theta_new;
    G(1, 3) =  radius * dt2_half * cos_theta_new;

    // 加速度对速度的影响
    G(4, 0) = dt;
    G(5, 1) = dt;
    G(6, 2) = dt;
    G(7, 3) = dt;


    // 1.4 预测协方差矩阵 P_{k|k-1} = F * P_{k-1} * F^T + Q

    this->CovState = F * this->CovState * F.transpose() + G * this->CovCourseSrc * G.transpose();
    
    //计算卡尔曼增益

    Eigen::Matrix<double, 8, 4> K = this->CovState.block<8,4>(0,0) * (this->CovState.block<4,4>(0,0) + this->CovView).inverse();


    // ---------------------------------------------------------------------
    // 2. 更新阶段 (Update)
    // ---------------------------------------------------------------------
    Eigen::Matrix<double, 4, 1> X_update = View - X_predict.block<4,1>(0,0);
    X_update(3) = std::remainder(X_update(3), CV_PI*2.0);

    Eigen::Matrix<double, 8, 1> ans =  X_predict + K * X_update;
    ans(3) = std::remainder(ans(3), CV_PI*2.0);

    // 2.2 更新协方差矩阵
    Eigen::Matrix<double, 8, 8> G_P = Eigen::Matrix<double, 8, 8>::Identity()- K*this->H;
    this->CovState = G_P * this->CovState * G_P.transpose() + K * this->CovView * K.transpose(); 

#ifdef EKFDebug
    viz.EKFKalmanUpdate(ans, CovArmor, View, CovState, K, radius, dt);
    viz.viewCov(View);
#endif

    return ans;
}

