#ifndef EKFKALMAN_HPP_INCLUDE
#define EKFKALMAN_HPP_INCLUDE
#include "eigen3/Eigen/Dense"

//数据单位:  距离cm,时间s,角度rad
class EKFKalman
{
public:
    EKFKalman() = default;

    void Init();

    Eigen::Matrix<double, 8, 1> operator()(const Eigen::Matrix<double, 8, 1>& State,
                                           const Eigen::Matrix<double, 4, 4>& CovArmor, 
                                           const Eigen::Matrix<double, 4, 1>& View,
                                           double radius, 
                                           double dt,
                                           double err);


    Eigen::Matrix<double, 8, 8> CovState;
    const Eigen::Matrix<double, 8, 8> CovStateInit
    {
        {100, 0,  0,  0,      0,      0,      0,      0},
        {0,  100, 0,  0,      0,      0,      0,      0},
        {0,  0,  100, 0,      0,      0,      0,      0},
        {0,  0,  0,  0.0009, 0,      0,      0,      0},
        {0,  0,  0,  0,      250000, 0,      0,      0},
        {0,  0,  0,  0,      0,      250000, 0,      0},
        {0,  0,  0,  0,      0,      0,      250000, 0},
        {0,  0,  0, 0,       0,      0,      0,      36}
        
    };

    //平移加速度和旋转加速度v_x,v_y,v_z,w
    const Eigen::Matrix<double, 4, 4> CovCourseSrc
    {
        {100,0,    0,    0},
        {0,    100,0,    0},
        {0,    0,    100,0},
        {0,    0,    0,    0.0001}
    };

    const Eigen::Matrix<double, 4, 4> CovView
    {
        {100,    0,    0,    0},
        {0,    100,    0,    0},
        {0,    0,    100,    0},
        {0,    0,    0,    0.01}
    };
    const Eigen::Matrix<double, 4, 4> CovUnView
    {
        {10,    0,    0,    0},
        {0,    10,    0,    0},
        {0,    0,    10,    0},
        {0,    0,    0,    0.001}
    };

    //观测矩阵
    const Eigen::Matrix<double, 4, 8> H
    {
        {1,    0,    0,    0,    0,    0,    0,    0},
        {0,    1,    0,    0,    0,    0,    0,    0},
        {0,    0,    1,    0,    0,    0,    0,    0},
        {0,    0,    0,    1,    0,    0,    0,    0}
    };

};
   


#endif // KALMAN_HPP