#include "../include/EKFKalman.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Geometry/Quaternion.h>
#include <opencv2/core/types.hpp>
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
Eigen::Matrix<double, 14, 1> EKFKalman::operator()(
    const Eigen::Matrix<double, 14, 1>& State,
    const Eigen::Matrix<double, 4, 1>& View,
    const cv::Point3d& SCS, 
    int armor_id,
    const cv::Quatd& quat, 
    double dt)
{
    //计算观测噪声矩阵
    Eigen::Matrix3d JacobianS2C = this->getJacobianSphericalToCartesian(SCS);
    Eigen::Matrix3d CovViewCameraCCS = JacobianS2C * this->CovViewCamera * JacobianS2C.transpose(); // 相机坐标系下的观测噪声
    Eigen::Quaterniond EigenQuat(quat.w, quat.x, quat.y, quat.z);
    Eigen::Matrix3d R_cam2world = EigenQuat.toRotationMatrix() * this->RCamera2Grip; // 从相机坐标系到世界坐标系的旋转矩阵

    this->CovView.block<3,3>(0,0) = 
        R_cam2world * CovViewCameraCCS * R_cam2world.transpose(); // 将相机观测噪声转换到世界坐标系
    
    this->CovView(3,3) = this->Var_yaw; // yaw 观测噪声

    // ==========================================
    // 1. 预测阶段 (Predict) - 纯线性匀速模型
    // ==========================================
    const double& vxc = State(3), vyc = State(4), vzc = State(5), w = State(7);

    Eigen::Matrix<double, 14, 1> X_curr = State;
    X_curr(0) += vxc * dt;
    X_curr(1) += vyc * dt;
    X_curr(2) += vzc * dt;
    X_curr(6) += w * dt;
    X_curr(6) = std::remainder(X_curr(6), CV_PI * 2.0);

    // 状态转移雅可比 F
    Eigen::Matrix<double, 14, 14> F = Eigen::Matrix<double, 14, 14>::Identity();
    F(0, 3) = dt; F(1, 4) = dt; F(2, 5) = dt; F(6, 7) = dt;

    // 过程噪声 Q
    Eigen::Matrix<double, 14, 14> Q = Eigen::Matrix<double, 14, 14>::Zero();
    auto a = dt * dt * dt * dt * 0.25;  
    auto b = dt * dt * dt * 0.5;       
    auto c = dt * dt;                
    
    Q(0,0)=Q(1,1)=Q(2,2) = a * this->Var_a;
    Q(3,3)=Q(4,4)=Q(5,5) = c * this->Var_a;
    Q(6,6) = a * this->Var_alpha;
    Q(7,7) = c * this->Var_alpha;
    // Q(8,8) = this->Var_r; 
    // Q(9,9) = this->Var_l; 
    // Q(10,10) = this->Var_h; 

    Q(0,3) = Q(3,0) = Q(1,4) = Q(4,1) = Q(2,5) = Q(5,2) = b * this->Var_a;     
    Q(6,7) = Q(7,6) = b * this->Var_alpha; 

    this->CovState = F * this->CovState * F.transpose() + Q;

    //获得观测矩阵H
    Eigen::Matrix<double, 4, 14> H = this->getStateToViewJacobian(X_curr, armor_id);

    //计算卡尔曼增益
    Eigen::Matrix<double, 14, 4> K = this->CovState * H.transpose() * (H * this->CovState * H.transpose() + this->CovView).inverse();

    //更新状态
    int& id = armor_id;
    const double& pred_xc = X_curr(0);
    const double& pred_yc = X_curr(1);
    const double& pred_zc = X_curr(2);
    const double& pred_yaw = X_curr(6);
    const double& pred_r = X_curr(8);
    const double& pred_l = X_curr(9);
    const double& pred_h = X_curr(10);

    bool is_side = (id == 1 || id == 3);

    double cur_r = is_side ? (pred_r + pred_l) : pred_r;
    double cur_z = is_side ? (pred_zc + pred_h) : pred_zc;

    double armor_angle = ( (id == 0) ? pred_yaw : pred_yaw + X_curr(10 + id,0) );
    armor_angle = std::remainder(armor_angle, CV_PI * 2.0);
    double cos_a = std::cos(armor_angle);
    double sin_a = std::sin(armor_angle);

    Eigen::Matrix<double, 4, 1> View_curr;
    View_curr(0) = pred_xc + cur_r * cos_a;
    View_curr(1) = pred_yc + cur_r * sin_a;
    View_curr(2) = cur_z;
    View_curr(3) = armor_angle;

    Eigen::Matrix<double, 4, 1> Y = View - View_curr;
    Y(3) = std::remainder(Y(3), CV_PI * 2.0);

    Eigen::Matrix<double, 14, 1> X_next = X_curr + K * Y;

    X_next(6) = std::remainder(X_next(6), CV_PI * 2.0);

    Eigen::Matrix<double, 14, 14> I_KH = Eigen::Matrix<double, 14, 14>::Identity() - K * H;
    this->CovState = I_KH * this->CovState * I_KH.transpose() + K * this->CovView * K.transpose();
    return X_next;
}


Eigen::Matrix<double, 14, 1> EKFKalman::operator()(
    const Eigen::Matrix<double, 14, 1>& State,
    const Eigen::Matrix<double, 10, 1>& Views,
    const cv::Point3d& SCS1,
    const cv::Point3d& SCS2, 
    int armor_id,
    const cv::Quatd& quat, 
    double dt)
{
    //计算观测噪声矩阵
    Eigen::Matrix3d JacobianS2C1 = this->getJacobianSphericalToCartesian(SCS1);
    Eigen::Matrix3d JacobianS2C2 = this->getJacobianSphericalToCartesian(SCS2);

    Eigen::Matrix3d CovViewCameraCCS1 = JacobianS2C1 * this->CovViewCamera * JacobianS2C1.transpose(); // 相机坐标系下的观测噪声
    Eigen::Matrix3d CovViewCameraCCS2 = JacobianS2C2 * this->CovViewCamera * JacobianS2C2.transpose(); // 相机坐标系下的观测噪声
    
    Eigen::Quaterniond EigenQuat(quat.w, quat.x, quat.y, quat.z);
    Eigen::Matrix3d R_cam2world = EigenQuat.toRotationMatrix() * this->RCamera2Grip; // 从相机坐标系到世界坐标系的旋转矩阵

    this->CovViews.block<3,3>(0,0) = 
        R_cam2world * CovViewCameraCCS1 * R_cam2world.transpose(); // 将相机观测噪声转换到世界坐标系

    this->CovViews.block<3,3>(4,4) = 
        R_cam2world * CovViewCameraCCS2 * R_cam2world.transpose();
    
    this->CovViews(3,3) = this->CovViews(7,7) = this->Var_yaw; // yaw 观测噪声

    this->CovViews(8,8) = this->CovViews(2,2) + this->CovViews(6,6); // h 观测噪声

    this->CovViews(9,9) = this->Var_dtheta; // dtheta 观测噪声


    // ==========================================
    // 1. 预测阶段 (Predict) - 纯线性匀速模型
    // ==========================================
    const double& vxc = State(3), vyc = State(4), vzc = State(5), w = State(7);

    Eigen::Matrix<double, 14, 1> X_curr = State;
    X_curr(0) += vxc * dt;
    X_curr(1) += vyc * dt;
    X_curr(2) += vzc * dt;
    X_curr(6) += w * dt;
    X_curr(6) = std::remainder(X_curr(6), CV_PI * 2.0);

    // 状态转移雅可比 F
    Eigen::Matrix<double, 14, 14> F = Eigen::Matrix<double, 14, 14>::Identity();
    F(0, 3) = dt; F(1, 4) = dt; F(2, 5) = dt; F(6, 7) = dt;

    // 过程噪声 Q
    Eigen::Matrix<double, 14, 14> Q = Eigen::Matrix<double, 14, 14>::Zero();
    auto a = dt * dt * dt * dt * 0.25;  
    auto b = dt * dt * dt * 0.5;       
    auto c = dt * dt;                
    
    Q(0,0)=Q(1,1)=Q(2,2) = a * this->Var_a;
    Q(3,3)=Q(4,4)=Q(5,5) = c * this->Var_a;
    Q(6,6) = a * this->Var_alpha;
    Q(7,7) = c * this->Var_alpha;
    // Q(8,8) = this->Var_r; 
    // Q(9,9) = this->Var_l; 
    // Q(10,10) = this->Var_h; 

    Q(0,3) = Q(3,0) = Q(1,4) = Q(4,1) = Q(2,5) = Q(5,2) = b * this->Var_a;     
    Q(6,7) = Q(7,6) = b * this->Var_alpha; 

    this->CovState = F * this->CovState * F.transpose() + Q;

    //获得观测矩阵H
    Eigen::Matrix<double, 10, 14> H = this->getStateToViewsJacobian(X_curr, armor_id);

    //计算卡尔曼增益
    Eigen::Matrix<double, 14, 10> K = this->CovState * H.transpose() * (H * this->CovState * H.transpose() + this->CovViews).inverse();

    //更新状态
    int id1 = armor_id;
    int id2 = (id1 + 1) % 4;

    const double& pred_xc = X_curr(0);
    const double& pred_yc = X_curr(1);
    const double& pred_zc = X_curr(2);
    const double& pred_yaw = X_curr(6);
    const double& pred_r = X_curr(8);
    const double& pred_l = X_curr(9);
    const double& pred_h = X_curr(10);

    bool is_side1 = (id1 == 1 || id1 == 3);
    bool is_side2 = (id2 == 1 || id2 == 3);

    double cur_r1 = is_side1 ? (pred_r + pred_l) : pred_r;
    double cur_z1 = is_side1 ? (pred_zc + pred_h) : pred_zc;

    double cur_r2 = is_side2 ? (pred_r + pred_l) : pred_r;
    double cur_z2 = is_side2 ? (pred_zc + pred_h) : pred_zc;

    double armor_angle1 = ( (id1 == 0) ? pred_yaw : pred_yaw + X_curr(10 + id1,0) );
    armor_angle1 = std::remainder(armor_angle1, CV_PI * 2.0);
    double cos_a1 = std::cos(armor_angle1);
    double sin_a1 = std::sin(armor_angle1);

    double armor_angle2 = ( (id2 == 0) ? pred_yaw : pred_yaw + X_curr(10 + id2,0) );
    armor_angle2 = std::remainder(armor_angle2, CV_PI * 2.0);
    double cos_a2 = std::cos(armor_angle2);
    double sin_a2 = std::sin(armor_angle2);

    Eigen::Matrix<double, 4, 1> View_curr1;
    View_curr1(0) = pred_xc + cur_r1 * cos_a1;
    View_curr1(1) = pred_yc + cur_r1 * sin_a1;
    View_curr1(2) = cur_z1;
    View_curr1(3) = armor_angle1;

    Eigen::Matrix<double, 4, 1> View_curr2;
    View_curr2(0) = pred_xc + cur_r2 * cos_a2;
    View_curr2(1) = pred_yc + cur_r2 * sin_a2;
    View_curr2(2) = cur_z2;
    View_curr2(3) = armor_angle2;

    Eigen::Matrix<double, 10, 1> Views_curr;
    Views_curr.block<4,1>(0,0) = View_curr1;
    Views_curr.block<4,1>(4,0) = View_curr2;

    Views_curr(8,0) = ( (id1 == 0 || id1 == 2) ? pred_h : -pred_h );

    if(id1 == 0)
    {
        Views_curr(9,0) = X_curr(11,0);
    }
    else if(id1 == 3)
    {
        Views_curr(9,0) = -X_curr(13,0);
    }
    else {
        Views_curr(9,0) = X_curr(10+id2,0) - X_curr(10 + id1,0);
    }
    

    Eigen::Matrix<double, 10, 1> Y = Views -Views_curr;
    Y(3,0) = std::remainder(Y(3), CV_PI * 2.0);
    Y(7,0) = std::remainder(Y(7), CV_PI * 2.0);
    Y(9,0) = std::remainder(Y(9), CV_PI * 2.0);

    Eigen::Matrix<double, 14, 1> X_next = X_curr + K * Y;

    X_next(6) = std::remainder(X_next(6), CV_PI * 2.0);
    X_next(11) = std::remainder(X_next(11), CV_PI * 2.0);
    X_next(12) = std::remainder(X_next(12), CV_PI * 2.0);
    X_next(13) = std::remainder(X_next(13), CV_PI * 2.0);

    Eigen::Matrix<double, 14, 14> I_KH = Eigen::Matrix<double, 14, 14>::Identity() - K * H;
    this->CovState = I_KH * this->CovState * I_KH.transpose() + K * this->CovViews * K.transpose();
    return X_next;
}



/**
 * @brief 计算从相机球坐标系到笛卡尔坐标系的雅可比矩阵
 * * @param SCS 球坐标点 (x: 半径 r, y: 极角 theta, z: 方位角 phi)
 * 基于 OpenCV 相机系约定：Z向前，X向右，Y向下
 * @return Eigen::Matrix3d 返回 3x3 的雅可比矩阵 J
 */
Eigen::Matrix<double, 3, 3> EKFKalman::getJacobianSphericalToCartesian(const cv::Point3d& SCS) {
    // SCS.x = r (距离)
    // SCS.y = theta (极角，与Z轴夹角)
    // SCS.z = phi (方位角，XY平面夹角)
    const double& r     = SCS.x;
    const double& theta = SCS.y;
    const double& phi   = SCS.z;

    double st = std::sin(theta);
    double ct = std::cos(theta);
    double sp = std::sin(phi);
    double cp = std::cos(phi);

    Eigen::Matrix<double, 3, 3> J;

    // 第一行：∂x/∂r, ∂x/∂theta, ∂x/∂phi
    J(0, 0) = st * cp;               // sin(theta) * cos(phi)
    J(0, 1) = r * ct * cp;           // r * cos(theta) * cos(phi)
    J(0, 2) = -r * st * sp;          // -r * sin(theta) * sin(phi)

    // 第二行：∂y/∂r, ∂y/∂theta, ∂y/∂phi
    J(1, 0) = st * sp;               // sin(theta) * sin(phi)
    J(1, 1) = r * ct * sp;           // r * cos(theta) * sin(phi)
    J(1, 2) = r * st * cp;           // r * sin(theta) * cos(phi)

    // 第三行：∂z/∂r, ∂z/∂theta, ∂z/∂phi
    J(2, 0) = ct;                    // cos(theta)
    J(2, 1) = -r * st;               // -r * sin(theta)
    J(2, 2) = 0.0;                   // z轴坐标与phi无关

    return J;
}

Eigen::Matrix<double, 4, 14> EKFKalman::getStateToViewJacobian(const Eigen::Matrix<double, 14, 1>& X_predict, int armor_id)
{
    Eigen::Matrix<double, 4, 14> H = Eigen::Matrix<double, 4, 14>::Zero();

    const double& yaw_0 = X_predict(6);
    const double& r = X_predict(8);
    const double& l = X_predict(9);

    bool is_side = (armor_id == 1 || armor_id == 3);
    double r_i = is_side ? (r + l) : r;

    // 获取当前板的物理角度 (虽然这里用到了 delta_theta，但不影响我们切断它的求导链条！)
    double theta_i = (armor_id == 0) ? yaw_0 : yaw_0 + X_predict(10 + armor_id);
    
    double cos_theta = std::cos(theta_i);
    double sin_theta = std::sin(theta_i);

    // 1. 位置对中心坐标求导
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    H(2, 2) = 1.0;

    // 2. 位置对结构参数求导
    H(0, 8) = cos_theta;                  // 对 r
    H(1, 8) = sin_theta;                  // 对 r
    H(0, 9) = is_side ? cos_theta : 0.0;  // 对 l
    H(1, 9) = is_side ? sin_theta : 0.0;  // 对 l
    H(2, 10) = is_side ? 1.0 : 0.0;       // 对 h

    // 3. 对全局 yaw_0 求导
    H(0, 6) = -r_i * sin_theta;
    H(1, 6) =  r_i * cos_theta;
    H(3, 6) = 1.0;

    return H;
}

Eigen::Matrix<double, 10, 14> EKFKalman::getStateToViewsJacobian(const Eigen::Matrix<double, 14, 1>& X_predict, int armor_id)
{
    Eigen::Matrix<double, 10, 14> H2 = Eigen::Matrix<double, 10, 14>::Zero();

    int id1 = armor_id;
    int id2 = (armor_id + 1)%4;
    H2.block<4, 14>(0, 0) = this->getStateToViewJacobian(X_predict, id1);
    H2.block<4, 14>(4, 0) = this->getStateToViewJacobian(X_predict, id2);
    
    // 第 9 维：高度差约束 h 
    // (预测值就是状态量 h, 索引为 10，对其导数为 1)
    H2(8, 10) = 1.0;

    // 第 10 维：相对偏角差约束 (DeltaTheta_id2 - DeltaTheta_id1)
    // (只对这俩特定的装配偏角有非 0 导数，彻底和 yaw_0 解耦！)
    if(id1 == 0)
    {
        H2(9, 10 + id2) = 1.0;
    }
    else if(id1 == 3)
    {
        H2(9, 13) = -1.0;
    }
    else 
    {
        H2(9, 10 + id1) = -1.0;
        H2(9, 10 + id2) = 1.0;
    }


    return H2;
}