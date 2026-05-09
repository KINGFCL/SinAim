#ifndef EKFKALMAN_HPP_INCLUDE
#define EKFKALMAN_HPP_INCLUDE
#include <cstddef>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#define EKFKalmanDebug
#ifdef EKFKalmanDebug
#include <functional>
using EKFDebugFn = void(*)(const Eigen::Matrix<double,14,1>&,
                            const Eigen::Matrix<double,4,1>&, double);
inline EKFDebugFn g_ekf_debug_cb = nullptr;
#endif


class EKFKalman
{
public:

    /*
    状态向量 State 为 14 维: [xc, yc, zc, vxc, vyc, vzc, theta_0, w, r,l,h,d_theta_1,d_theta_2,d_theta_3]
    */
    EKFKalman(const Eigen::Matrix3d& RCamera2Grip): RCamera2Grip(RCamera2Grip) {};

    void Init();
    
    // [新增] 鲁棒卡尔曼超参数
    const double Max_Robust_Scale = 2.0; // 最大放大系数
    const double Threshold_Pos = 1000.0;       // 位置误差容忍阈值 (单位: cm)
    const double Threshold_Angle = 0.8;      // 角度误差容忍阈值 (单位: rad)
    const double Threshold_Angle_diff = 0.2; // 角度差误差容忍阈值 (单位: rad)
    //观测噪声
    const double Var_r = 100.0, Var_yaw = 0.003, Var_dtheta = 0.01; // 观测yaw值方差

    Eigen::Matrix3d RCamera2Grip;
    
    // 过程噪声参数
    const double Var_a_xy = 10000.0,  Var_a_z = 100.0, Var_alpha = 10;
    
    // 初始化协方差
    const Eigen::Matrix<double, 14, 14> CovStateInit = (Eigen::Matrix<double, 14, 1>() << 
        100, 100, 100,       // xc, yc, zc 位置方差
        10000, 10000, 10000, // vxc, vyc, vzc 速度方差
        0.01, 5,           // theta, w 角度与角速度方差
        0.0, 0, 0,           // r, l, h 几何结构初始方差
        0.00, 0.00, 0.00  //d_theta_1,d_theta_2,d_theta_3
    ).finished().asDiagonal();

    Eigen::Matrix<double, 3, 3> CovViewCamera = (Eigen::Matrix<double, 3, 1>() << 
    this->Var_r, 0.009, 0.009  // 相机中球坐标系下的方差，r,theta,phi 的观测噪声
    ).finished().asDiagonal();
    
    // 结构参数收敛噪声极小
    // const double Var_r = 0.01, Var_l = 0.01, Var_h = 0.01; 

    /*
        * 单装甲板更新
        状态向量 State 为 14 维: [xc, yc, zc, vxc, vyc, vzc, theta_0, w, r,l,h,d_theta_1,d_theta_2,d_theta_3]
        观测向量 View 为 4 维: [x_i, y_i, z_i, yaw_i]
    */ 
    Eigen::Matrix<double, 14, 1> operator()(
        const Eigen::Matrix<double, 14, 1>& State,
        const Eigen::Matrix<double, 4, 1>& View, 
        const Eigen::Vector3d& SCS,
        const Eigen::Quaterniond& gripper_to_world,
        double delta_angle,
        size_t armor_id,                           
        double dt);

    /*
        多装甲板序更新
        状态向量 State 为 14 维: [xc, yc, zc, vxc, vyc, vzc, theta_0, w, r,l,h,d_theta_1,d_theta_2,d_theta_3]
        观测向量 View 为 10 维: [x_i, y_i, z_i, yaw_i, x_i+1, y_i+1, z_i+1, yaw_i+1, z_i+1 - z_i , (d_theta_i+1 - d_theta_i)]
    */ 
    Eigen::Matrix<double, 14, 1> operator()(
        const Eigen::Matrix<double, 14, 1>& State,
        const Eigen::Matrix<double, 10, 1>& Views,
        const Eigen::Vector3d& SCS1,
        const Eigen::Vector3d& SCS2,
        const Eigen::Quaterniond& gripper_to_world,
        double delta_angle1,
        double delta_angle2,
        size_t armor_id,                        
        double dt);

    Eigen::Matrix<double, 14, 1> operator()(
    const Eigen::Matrix<double, 14, 1>& State,
    double dt);

    double GetAngleRobustScale(double angle_error) const;
    double GetDistanceRobustScale(double distance_error) const;
    double GetAngleDiffRobustScale(double angle_diff_error)const;
    
        
    Eigen::Matrix<double, 3, 3> getJacobianSphericalToCartesian(const Eigen::Vector3d& SCS);

    Eigen::Matrix<double, 4, 14> getStateToViewJacobian(const Eigen::Matrix<double, 14, 1>& X_predict, size_t armor_id);
    
    /*
    观测向量 View 为 8 维: [x_i, y_i, z_i, yaw_i, x_i+1, y_i+1, z_i+1, yaw_i+1, h , (d_theta_i+1 - d_theta_i)]
    */
    Eigen::Matrix<double, 10, 14> getStateToViewsJacobian(const Eigen::Matrix<double, 14, 1>& X_predict, size_t armor_id);

    Eigen::Matrix<double, 14, 14> CovState;


    // 测量噪声 R
    Eigen::Matrix<double, 4, 4> CovView = Eigen::Matrix<double, 4, 4>::Zero(); // 4维观测: [x, y, z, yaw]
    Eigen::Matrix<double, 10, 10> CovViews = Eigen::Matrix<double, 10, 10>::Zero(); 
    


};

#endif // EKFKALMAN_HPP_INCLUDE