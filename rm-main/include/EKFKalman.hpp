#ifndef EKFKALMAN_HPP_INCLUDE
#define EKFKALMAN_HPP_INCLUDE
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <opencv2/core/quaternion.hpp>
#include <vector>

class EKFKalman
{
public:
//状态向量 State 为 11 维: [xc, yc, zc, vxc, vyc, vzc, theta_0, w, r,l,h]
    EKFKalman() = default;
    void Init();

    // 过程噪声参数
    const double Var_yaw = 0.6; // 观测yaw值方差
    const double Var_a = 10000.0, Var_alpha = 0.6;
    const double Var_r = 0.01, Var_l = 0.01, Var_h = 0.01; // 结构参数收敛噪声极小

    // 单装甲板更新
    Eigen::Matrix<double, 11, 1> operator()(
        const Eigen::Matrix<double, 11, 1>& State,
        const Eigen::Matrix<double, 4, 1>& View, 
        int armor_id,
        const cv::Quatd& quat,                            
        double dt);

    // 多装甲板序贯更新
    Eigen::Matrix<double, 11, 1> operator()(
        const Eigen::Matrix<double, 11, 1>& State,
        const std::vector<Eigen::Matrix<double, 4, 1>>& Views, 
        const std::vector<int>& armor_ids,
        const cv::Quatd& quat,                         
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
    Eigen::Matrix<double, 4, 4> CovView = Eigen::Matrix<double, 4, 4>::Zero(); // 4维观测: [x, y, z, yaw] 
    const Eigen::Matrix<double, 3, 3> CovViewCamera = (Eigen::Matrix<double, 4, 1>() << 
        10, 10, 100 // 相机器的x, y, z 的观测噪声
    ).finished().asDiagonal();

    const Eigen::Matrix<double, 3, 3> RCamera2Grip// 从相机坐标系到手坐标系的旋转矩阵
    {
        -0.009102138195790865, 0.006927977296756926, -0.9999345749652024,
        0.999898703383153, -0.01087943808061275, -0.009177189097925642,
        -0.01094230565071617, -0.9999168170191085, -0.00682824936719284
    };
};

#endif // EKFKALMAN_HPP_INCLUDE