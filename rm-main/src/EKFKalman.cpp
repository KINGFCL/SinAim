#include "../include/EKFKalman.hpp"
#include <cmath>
#include <cstddef>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Geometry/Quaternion.h>
#include <vector>
#include <algorithm>
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
    const cv::Quatd& quat, 
    double dt)
{
    std::vector<Eigen::Matrix<double, 4, 1>> Views = {View};
    std::vector<int> armor_ids = {armor_id};
    return this->operator()(State, Views, armor_ids, quat, dt);
}

// ---------------------------------------------------------
// 多装甲板序贯更新核心算法
// ---------------------------------------------------------
Eigen::Matrix<double, 11, 1> EKFKalman::operator()(
    const Eigen::Matrix<double, 11, 1>& State,
    const std::vector<Eigen::Matrix<double, 4, 1>>& Views, 
    const std::vector<int>& armor_ids,
    const cv::Quatd& quat, 
    double dt)
{
    //计算观测噪声矩阵
    Eigen::Quaterniond EigenQuat(quat.w, quat.x, quat.y, quat.z);
    Eigen::Matrix3d R_cam2world = EigenQuat.toRotationMatrix() * this->RCamera2Grip; // 从相机坐标系到世界坐标系的旋转矩阵

    this->CovView.block<3,3>(0,0) = 
        R_cam2world * this->CovViewCamera * R_cam2world.transpose(); // 将相机观测噪声转换到世界坐标系
    
    this->CovView(3,3) = this->Var_yaw; // yaw 观测噪声

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
    auto a = dt * dt * dt * dt * 0.25;  
    auto b = dt * dt * dt * 0.5;       
    auto c = dt * dt;                
    
    Q(0,0)=Q(1,1)=Q(2,2) = a * this->Var_a;
    Q(3,3)=Q(4,4)=Q(5,5) = c * this->Var_a;
    Q(6,6) = a * this->Var_alpha;
    Q(7,7) = c * this->Var_alpha;
    Q(8,8) = this->Var_r; 
    Q(9,9) = this->Var_l; 
    Q(10,10) = this->Var_h; 

    Q(0,3) = Q(3,0) = Q(1,4) = Q(4,1) = Q(2,5) = Q(5,2) = b * this->Var_a;     
    Q(6,7) = Q(7,6) = b * this->Var_alpha; 

    this->CovState = F * this->CovState * F.transpose() + Q;

    // ==========================================
    // 预先计算融合权重，并决定序贯更新的优先级
    // ==========================================
    std::vector<size_t> update_order(Views.size());
    std::vector<double> face_projs(Views.size(), 0.1);

    for (size_t i = 0; i < Views.size(); ++i) 
    {
        update_order[i] = i; // 初始化默认索引

        Eigen::Matrix<double, 3, 1> P_view{Views[i](0), Views[i](1), 0.0};
        double distance = P_view.norm();
        
        if (distance > 1e-3) {
            Eigen::Matrix<double, 3, 1> L_view = P_view / distance; 
            
            // 切线是 (cos, sin, 0)，逆时针转 90 度的法向量就是 (-sin, cos, 0)
            Eigen::Matrix<double, 3, 1> N_view(-std::sin(Views[i](3)), std::cos(Views[i](3)), 0.0);
            // 计算迎角投影，并限制最小值为 0.1 防止除 0
            face_projs[i] = std::max(std::abs(N_view.dot(L_view)), 0.1); 
        }
    }

    //如果有多块板，优先更新“最正对”的那一块
    if (Views.size() == 2) 
    {
        // 绝大多数多板情况只有 2 块板，直接一个 if 搞定，0 函数调用开销！
        if (face_projs[update_order[0]] < face_projs[update_order[1]]) {
            std::swap(update_order[0], update_order[1]);
        }
    }

    // ==========================================
    // 2. 序贯更新阶段 (Sequential Update)
    // ==========================================
    // 按照算好的最优顺序遍历
    size_t num_views = std::min(Views.size(), (size_t)2);
    for (size_t order_idx = 0; order_idx < num_views; ++order_idx) 
    {
        size_t i = update_order[order_idx]; // 获取实际在 Views 里的索引
        int id = armor_ids[i];
        const auto& View = Views[i];
        double face_proj = face_projs[i]; // 取出预先算好的迎角投影

        double pred_xc = X_curr(0), pred_yc = X_curr(1), pred_zc = X_curr(2);
        double pred_yaw = X_curr(6);
        double pred_r = X_curr(8), pred_l = X_curr(9), pred_h = X_curr(10);

        bool is_side = (id == 1 || id == 3);
        double cur_r = is_side ? (pred_r + pred_l) : pred_r;
        double cur_z = is_side ? (pred_zc + pred_h) : pred_zc;

        double armor_angle = pred_yaw + id * (CV_PI / 2.0);
        armor_angle = std::remainder(armor_angle, CV_PI * 2.0);
        double cos_a = std::cos(armor_angle);
        double sin_a = std::sin(armor_angle);

        Eigen::Matrix<double, 4, 1> Z_predict;
        Z_predict(0) = pred_xc + cur_r * cos_a;
        Z_predict(1) = pred_yc + cur_r * sin_a;
        Z_predict(2) = cur_z;
        Z_predict(3) = armor_angle;

        Eigen::Matrix<double, 4, 11> H = Eigen::Matrix<double, 4, 11>::Zero();
        H(0, 0) = 1.0; H(0, 6) = -cur_r * sin_a; H(0, 8) = cos_a; H(0, 9) = is_side ? cos_a : 0.0; 
        H(1, 1) = 1.0; H(1, 6) = cur_r * cos_a;  H(1, 8) = sin_a; H(1, 9) = is_side ? sin_a : 0.0; 
        H(2, 2) = 1.0; H(2, 10) = is_side ? 1.0 : 0.0;  
        H(3, 6) = 1.0;            

        Eigen::Matrix<double, 4, 1> Y = View - Z_predict;
        Y(3) = std::remainder(Y(3), CV_PI * 2.0);

        // ==========================================
        // 自适应观测噪声矩阵 (AEKF)
        // ==========================================
        Eigen::Matrix<double, 4, 4> Dynamic_R = this->CovView;
        double scale = 1.0 / face_proj; // 投影越小，惩罚系数越大
        
        Dynamic_R(0, 0) *= scale;             // X: 线性惩罚
        Dynamic_R(1, 1) *= scale;             // Y: 线性惩罚
        Dynamic_R(2, 2) *= (scale * scale);   // Z(距离): 大侧偏极度不准，平方惩罚
        Dynamic_R(3, 3) *= (scale * scale);   // Yaw(角度): 大侧偏极度不准，平方惩罚

        // 计算卡尔曼增益 K (使用动态 R 矩阵)
        Eigen::Matrix<double, 4, 4> S = H * this->CovState * H.transpose() + Dynamic_R;
        Eigen::Matrix<double, 11, 4> K = this->CovState * H.transpose() * S.inverse();

        // 更新状态向量
        X_curr = X_curr + K * Y;
        X_curr(6) = std::remainder(X_curr(6), CV_PI * 2.0); 

        // 更新协方差矩阵 (使用动态 R 矩阵)
        Eigen::Matrix<double, 11, 11> I_KH = Eigen::Matrix<double, 11, 11>::Identity() - K * H;
        this->CovState = I_KH * this->CovState * I_KH.transpose() + K * Dynamic_R * K.transpose();
    }

    return X_curr;
}