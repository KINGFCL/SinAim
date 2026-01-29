#ifndef TRACKER_HPP
#define TRACKER_HPP
#include "Kamal.hpp"
#include "Target.hpp"
#include <eigen3/Eigen/Core>

class Tracker 
{
public:
    Tracker(const Target& target);

    /*!
        @return x, y, z, v_x, v_y, v_z, w, angel  
    */
    const Eigen::Matrix<double, 8, 1>& operator()(const Eigen::Matrix<double, 4, 1>& measurement,double dt, double error = 1);

private:

    const Target& target;

    bool is_first_ = true;
    Kalman<8, 4> kalman_
    {
    (Eigen::Matrix<double, 4, 8>() << 
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1).finished()
    };

    //传感器噪声源协方差矩阵
    Eigen::Matrix<double, 4, 4> R_src;

    //过程噪声源协方差矩阵
    Eigen::Matrix<double, 4, 4> Q_src;

    //初始状态向量协方差矩阵
    Eigen::Matrix<double, 8, 8> P_0; 
};


#endif // TRACKER_HPP