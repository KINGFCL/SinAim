#include "LKFKalman.hpp"


void LKFKalman::Init() {
    this->CovState = this->CovStateInit;
}



Eigen::Matrix<double, 6, 1> LKFKalman::operator()(const Eigen::Matrix<double, 6, 1>& State, const Eigen::Vector3d View, const Eigen::Vector3d& SCS, double dt) {
    
    // 计算观测噪声矩阵
    Eigen::Matrix<double, 3, 3> J = this->getJacobianSphericalToCartesian(SCS);
    this->CovView.noalias() = J * this->CovViewSrc * J.transpose();

    // 计算预测矩阵
    this->F(0, 3) = dt; // X轴：位置 += 速度_x * dt
    this->F(1, 4) = dt; // Y轴：位置 += 速度_y * dt
    this->F(2, 5) = dt; // Z轴：位置 += 速度_z * dt


    // 计算过程噪声矩阵
    Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
    double c = dt * dt;                
    double a = c * c * 0.25;  
    double b = c * dt * 0.5;       
    
    // 严格区分 XY 轴和 Z 轴
    Q(0,0) = Q(1,1) = a * this->Var_a_xy; // X, Y 位置
    Q(2,2) = a * this->Var_a_z;           // Z 位置 (极小)
    
    Q(3,3) = Q(4,4) = c * this->Var_a_xy; // X, Y 速度
    Q(5,5) = c * this->Var_a_z;           // Z 速度 (极小)
    
    Q(0,3) = Q(3,0) = Q(1,4) = Q(4,1) = b * this->Var_a_xy; // X,Y 协方差
    Q(2,5) = Q(5,2) = b * this->Var_a_z;  

    //先验状态协方差矩阵

    Eigen::Matrix<double, 6, 6> CovState_pred;
    CovState_pred.noalias() = this->F * this->CovState * this->F.transpose() + Q;

    // 计算卡尔曼增益矩阵
    Eigen::Matrix<double, 6, 3> K = this->CovState * this->H.transpose() * (this->H * this->CovState * this->H.transpose() + this->CovView).inverse();

    // 更新后验状态协方差矩阵
    Eigen::Matrix<double, 6, 6> I_KH = Eigen::Matrix<double, 6, 6>::Identity() - K * H;
    this->CovState = I_KH * this->CovState * I_KH.transpose() + K * this->CovView * K.transpose();

    // 更新后验状态估计量
    Eigen::Matrix<double, 6, 1> State_pred = this->F * State;
    return State_pred + K * (View - this->H * State_pred);

}



/**
 * @brief 计算从相机球坐标系到笛卡尔坐标系的雅可比矩阵
 * * @param SCS 球坐标点 (x: 半径 r, y: 极角 theta, z: 方位角 phi)
 * @return Eigen::Matrix3d 返回 3x3 的雅可比矩阵 J
 */
Eigen::Matrix<double, 3, 3> LKFKalman::getJacobianSphericalToCartesian(const Eigen::Vector3d& SCS) {
    // SCS.x = r (距离)
    // SCS.y = theta (极角，与Z轴夹角)
    // SCS.z = phi (方位角，XY平面夹角)
    const double& r     = SCS(0);
    const double& theta = SCS(1);
    const double& phi   = SCS(2);

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