#pragma once
#include "Eigen/Core"
class LKFKalman
{
private:
    const Eigen::Matrix<double, 6, 6> CovStateInit = (Eigen::Matrix<double, 6, 1>() << 
        100, 100, 100,       // xc, yc, zc 位置方差
        10000, 10000, 10000 // vxc, vyc, vzc 速度方差
    ).finished().asDiagonal();

    //观测噪声源
    const double Var_r = 100.0, Var_dtheta = 0.0001; //单位：cm，rad
    const Eigen::Matrix3d CovViewSrc = (Eigen::Matrix<double, 3, 1>() <<
        Var_r, Var_dtheta, Var_dtheta
    ).finished().asDiagonal();

    // 过程噪声参数
    const double Var_a_xy = 10000.0,  Var_a_z = 10.0;//单位：cm/s^2

    LKFKalman() = default;
public:
    void Init();
    Eigen::Matrix<double, 6, 1> operator()(const Eigen::Matrix<double, 6, 1>& State, const Eigen::Vector3d View, const Eigen::Vector3d& SCS, double dt);
private:
    Eigen::Matrix<double, 3, 3> getJacobianSphericalToCartesian(const Eigen::Vector3d& SCS);
    Eigen::Matrix<double, 6, 6> CovState = Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::Matrix<double, 3, 3> CovView = Eigen::Matrix<double, 3, 3>::Identity();
    
    const Eigen::Matrix<double, 3, 6> H{{1,0,0,0,0,0},
                                        {0,1,0,0,0,0},
                                        {0,0,1,0,0,0}};
    Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Identity();


};