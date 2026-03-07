#ifndef EKFKALMAN_HPP_INCLUDE
#define EKFKALMAN_HPP_INCLUDE
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <vector>

class EKFKalman
{
public:
    EKFKalman() = default;
    void Init();

    // 过程噪声参数
    const double Var_a = 10000.0, Var_alpha = 1.0;
    const double Var_r = 0.01, Var_l = 0.01, Var_h = 0.01; // 结构参数收敛噪声极小

    // 单装甲板更新
    Eigen::Matrix<double, 11, 1> operator()(
        const Eigen::Matrix<double, 11, 1>& State,
        const Eigen::Matrix<double, 4, 1>& View, 
        int armor_id,                            
        double dt);

    // 多装甲板序贯更新
    Eigen::Matrix<double, 11, 1> operator()(
        const Eigen::Matrix<double, 11, 1>& State,
        const std::vector<Eigen::Matrix<double, 4, 1>>& Views, 
        const std::vector<int>& armor_ids,                         
        double dt);    

    Eigen::Matrix<double, 11, 11> CovState;

    // 初始化协方差
    const Eigen::Matrix<double, 11, 11> CovStateInit = (Eigen::Matrix<double, 11, 1>() << 
        100, 100, 100,       // xc, yc, zc 位置方差
        10000, 10000, 10000, // vxc, vyc, vzc 速度方差
        0.01, 36,           // theta, w 角度与角速度方差
        10, 10, 10           // r, l, h 几何结构初始方差
    ).finished().asDiagonal();

    // 测量噪声 R
    const Eigen::Matrix<double, 4, 4> CovView = (Eigen::Matrix<double, 4, 1>() << 
        25, 25, 25, 0.001 // x, y, z, theta 的观测噪声
    ).finished().asDiagonal();

};

#endif // EKFKALMAN_HPP_INCLUDE