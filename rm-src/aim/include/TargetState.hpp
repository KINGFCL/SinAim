#ifndef INCLUDE_TARGET_STATE_HPP
#define INCLUDE_TARGET_STATE_HPP
#include "Target.hpp"

struct RobotState
{
    Robot::KalmanMode Mode;
    //整车速度v_x,v_y,v_z,w
    Eigen::Matrix<double, 4, 1> Speed;

    /* 4个装甲板位置：
        ID:   0        1          2          3

        0     x        x          x          x
        1     y        y          y          y
        2     z        z          z          z
        3   theta    theta      theta      theta
        4   radius   radius     radius     radius
    */

    /*
        theta radius h 
    */
    Eigen::Matrix<double, 3, 4> Armors;
    Eigen::Matrix<double, 4, 4> ArmorsPosi;
    double l_diff = 0, h_diff = 0;
    double d_theta_1, d_theta_2, d_theta_3;
    
    //中心点坐标
    Eigen::Matrix<double,3,1> center;

    std::chrono::steady_clock::time_point StateTime;

    RobotState(const Robot& robot, const std::chrono::steady_clock::time_point& t);
    
    /*!
    @param dt 单位: s
    */
    Eigen::Matrix<double, 4, 4> Predict(double dt)const;

    void Predict(const std::chrono::steady_clock::time_point& t);
};

#endif