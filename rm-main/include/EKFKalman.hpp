#ifndef EKFKALMAN_HPP_INCLUDE
#define EKFKALMAN_HPP_INCLUDE
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <opencv2/core/quaternion.hpp>
#include <vector>

class EKFKalman
{
public:

    /*
    状态向量 State 为 14 维: [xc, yc, zc, vxc, vyc, vzc, theta_0, w, r,l,h,d_theta_1,d_theta_2,d_theta_3]
    */
    EKFKalman() = default;

    void Init();

    // 过程噪声参数
    const double Var_yaw = 0.6, Var_dtheta = 0.001,Var_h_view = 10.0; // 观测yaw值方差
    const double Var_a = 10000.0, Var_alpha = 0.6;
    const double Var_r = 0.01, Var_l = 0.01, Var_h = 0.01; // 结构参数收敛噪声极小

    /*
        * 单装甲板更新
        状态向量 State 为 14 维: [xc, yc, zc, vxc, vyc, vzc, theta_0, w, r,l,h,d_theta_1,d_theta_2,d_theta_3]
        观测向量 View 为 4 维: [x_i, y_i, z_i, yaw_i]
    */ 
    Eigen::Matrix<double, 14, 1> operator()(
        const Eigen::Matrix<double, 14, 1>& State,
        const Eigen::Matrix<double, 4, 1>& View, 
        const cv::Point3d& SCS,
        int armor_id,
        const cv::Quatd& quat,                            
        double dt);

    /*
        多装甲板序更新
        状态向量 State 为 14 维: [xc, yc, zc, vxc, vyc, vzc, theta_0, w, r,l,h,d_theta_1,d_theta_2,d_theta_3]
        观测向量 View 为 10 维: [x_i, y_i, z_i, yaw_i, x_i+1, y_i+1, z_i+1, yaw_i+1, z_i+1 - z_i , (d_theta_i+1 - d_theta_i)]
    */ 
    Eigen::Matrix<double, 14, 1> operator()(
        const Eigen::Matrix<double, 14, 1>& State,
        const Eigen::Matrix<double, 10, 1>& Views,
        const cv::Point3d& SCS1,
        const cv::Point3d& SCS2,
        int armor_id,
        const cv::Quatd& quat,                         
        double dt);
        
    Eigen::Matrix<double, 3, 3> getJacobianSphericalToCartesian(const cv::Point3d& SCS);

    Eigen::Matrix<double, 4, 14> getStateToViewJacobian(const Eigen::Matrix<double, 14, 1>& X_predict,int armor_id);
    
    /*
    观测向量 View 为 8 维: [x_i, y_i, z_i, yaw_i, x_i+1, y_i+1, z_i+1, yaw_i+1, h , (d_theta_i+1 - d_theta_i)]
    */
    Eigen::Matrix<double, 10, 14> getStateToViewsJacobian(const Eigen::Matrix<double, 14, 1>& X_predict,int armor_id);

    Eigen::Matrix<double, 14, 14> CovState;

    // 初始化协方差
    const Eigen::Matrix<double, 14, 14> CovStateInit = (Eigen::Matrix<double, 14, 1>() << 
        100, 100, 100,       // xc, yc, zc 位置方差
        10000, 10000, 10000, // vxc, vyc, vzc 速度方差
        0.01, 36,           // theta, w 角度与角速度方差
        10, 10, 10,           // r, l, h 几何结构初始方差
        0.0016, 0.0016, 0.0016  //d_theta_1,d_theta_2,d_theta_3
    ).finished().asDiagonal();

    // 测量噪声 R
    Eigen::Matrix<double, 4, 4> CovView = Eigen::Matrix<double, 4, 4>::Zero(); // 4维观测: [x, y, z, yaw]
    Eigen::Matrix<double, 10, 10> CovViews = Eigen::Matrix<double, 10, 10>::Zero(); 
    
    Eigen::Matrix<double, 3, 3> CovViewCamera = (Eigen::Matrix<double, 3, 1>() << 
        100, 0.0004, 0.0004  // 相机中球坐标系下的方差，r,theta,phi 的观测噪声
    ).finished().asDiagonal();

    const Eigen::Matrix<double, 3, 3> RCamera2Grip// 从相机坐标系到手坐标系的旋转矩阵
    {
        -0.009102138195790865, 0.006927977296756926, -0.9999345749652024,
        0.999898703383153, -0.01087943808061275, -0.009177189097925642,
        -0.01094230565071617, -0.9999168170191085, -0.00682824936719284
    };
};

#endif // EKFKALMAN_HPP_INCLUDE