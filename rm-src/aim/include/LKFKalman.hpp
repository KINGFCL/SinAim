#pragma once

#include <Eigen/Core>
#include <Eigen/Dense> 
#include <cmath>

/**
 * @brief 线性化/扩展卡尔曼滤波器 (Linearized/Extended Kalman Filter)
 * * 该滤波器使用等速(CV)模型，状态量为 6D (x, y, z, vx, vy, vz)。
 * 观测输入为笛卡尔坐标系下的 3D 位置，但观测噪声模型基于相机/雷达的球坐标系(r, theta, phi)定义，
 * 通过雅可比矩阵(Jacobian)动态投影到笛卡尔坐标系中。
 * 内部采用了约瑟夫形式(Joseph form)更新协方差矩阵，以保证数值稳定性。
 */
class LKFKalman
{
private:
    /** * @brief 初始状态协方差矩阵 (P0)
     * 对角线元素分别对应: [xc, yc, zc, vxc, vyc, vzc] 的初始方差
     */
    const Eigen::Matrix<double, 6, 6> CovStateInit = (Eigen::Matrix<double, 6, 1>() << 
        100, 100, 100,       // xc, yc, zc 位置方差 (单位: cm^2)
        10000, 10000, 10000  // vxc, vyc, vzc 速度方差 (单位: (cm/s)^2)
    ).finished().asDiagonal();

    // ================== 观测噪声参数 ==================
    const double Var_r = 100.0;       ///< 测距方差 (单位: cm^2)
    const double Var_dtheta = 0.0001; ///< 测角方差 (单位: rad^2)
    
    /** @brief 球坐标系下的传感器固有观测噪声协方差 (R_spherical) */
    const Eigen::Matrix3d CovViewSrc = (Eigen::Matrix<double, 3, 1>() <<
        Var_r, Var_dtheta, Var_dtheta
    ).finished().asDiagonal();

    // ================== 过程噪声参数 ==================
    const double Var_a_xy = 10000.0;  ///< XY 平面加速度扰动方差 (单位: (cm/s^2)^2)
    const double Var_a_z = 10.0;      ///< Z 轴加速度扰动方差 (假定 Z 轴运动较平缓)

public:
    /** @brief 默认构造函数 */
    LKFKalman() = default;

    /**
     * @brief 初始化/重置滤波器状态协方差
     * 应该在开始新的追踪任务前调用
     */
    void Init();

    /**
     * @brief 执行卡尔曼滤波的预测与更新 (Predict & Update)
     * * @param State 当前先验状态向量 [x, y, z, vx, vy, vz]^T
     * @param View  传感器观测到的笛卡尔坐标系位置 [x, y, z]^T
     * @param SCS   传感器观测到的球坐标系数据 [r, theta, phi]^T (用于计算雅可比矩阵)
     * @param dt    距离上一次更新的时间间隔 (单位: s)
     * @return Eigen::Matrix<double, 6, 1> 更新后的后验最优状态估计向量
     */
    Eigen::Matrix<double, 6, 1> operator()(const Eigen::Matrix<double, 6, 1>& State, 
                                           const Eigen::Vector3d View, 
                                           const Eigen::Vector3d& SCS, 
                                           double dt);

    Eigen::Matrix<double, 6, 1> operator()(const Eigen::Matrix<double, 6, 1>& State,double dt);


private:
    /**
     * @brief 计算从球坐标系到笛卡尔坐标系的雅可比矩阵 (Jacobian)
     * @param SCS 球坐标点向量 (x: 半径 r, y: 极角 theta, z: 方位角 phi)
     * @return Eigen::Matrix<double, 3, 3> 3x3 雅可比矩阵
     */
    Eigen::Matrix<double, 3, 3> getJacobianSphericalToCartesian(const Eigen::Vector3d& SCS);

    /// @brief 内部维护的后验状态协方差矩阵 (P)
    Eigen::Matrix<double, 6, 6> CovState = Eigen::Matrix<double, 6, 6>::Identity();
    
    /// @brief 映射到笛卡尔坐标系下的动态观测噪声协方差矩阵 (R)
    Eigen::Matrix<double, 3, 3> CovView = Eigen::Matrix<double, 3, 3>::Identity();
    
    /** * @brief 观测矩阵 (H)
     * 仅观测位置 [x, y, z]，不直接观测速度
     */
    const Eigen::Matrix<double, 3, 6> H{{1, 0, 0, 0, 0, 0},
                                        {0, 1, 0, 0, 0, 0},
                                        {0, 0, 1, 0, 0, 0}};
                                        
    /** * @brief 状态转移矩阵 (F)
     * 默认初始化为单位阵，运行中动态更新速度项 (F(0,3)=dt 等)
     */
    Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Identity();
};