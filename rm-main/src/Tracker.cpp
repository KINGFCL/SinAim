#include "../include/Tracker.hpp"
#include <eigen3/Eigen/src/Core/Matrix.h>

Tracker::Tracker(const Target& target) : is_first_(true),target(target) 
{
    //构建观测噪声协方差矩阵,距离: mm，角度: rad
    this->R_src <<
        2500,   0,      0,     0,
        0,      2500,   0,     0,
        0,      0,      2500,  0,
        0,      0,      0,     0.01;

    //构建过程噪声源矩阵,加速度: mm/s^2, 角加速度: rad/s^2
    this->Q_src << 
        250000,  0,       0,      0,
        0,       250000,  0,      0,
        0,       0,       250000, 0,
        0,       0,       0,      1;

    //初始化首次状态向量矩阵
    this-> P_0 << 
        2500,   0,      0,     0,       0,       0,       0,    0,
        0,      2500,   0,     0,       0,       0,       0,    0,
        0,      0,      2500,  0,       0,       0,       0,    0,
        0,      0,      0,     4000000, 0,       0,       0,    0,
        0,      0,      0,     0,       4000000, 0,       0,    0,
        0,      0,      0,     0,       0,       4000000, 0,    0,        
        0,      0,      0,     0,       0,       0,       1,    0,
        0,      0,      0,     0,       0,       0,       0,    0.01;         
}


const Eigen::Matrix<double, 8, 1>& Tracker::operator()(const Eigen::Matrix<double, 4, 1>& measurement,double dt,double error)
{
    //如果是第一次更新先初始化
    if(is_first_)
    {
        is_first_ = false;

        //初始化状态向量
            //更新位置坐标
            kalman_.StateVector().block<3,1>(0,0) = measurement.block<3,1>(0,0);
            
            //更新朝向角度
            kalman_.StateVector()(7,0) = measurement(3,0);
        
        //初始化状态向量协方差
            kalman_.Covariance_State() = this->P_0;
            
        return kalman_.StateVector();
    }

    //构建状态转移矩阵
    double angel = kalman_.StateVector()(6,0)*dt;
    Eigen::Matrix<double, 8, 8> A;
    
    A << 
        1, 0, 0, dt, 0,  0,
        0, 1, 0, 0,  dt, 0,
        0, 0, 1, 0,  0,  dt,
        0, 0, 0, 1,  0,  0,
        0, 0, 0, 0,  1,  0,
        0, 0, 0, 0,  0,  1;
    
    //状态向量转移偏移向量
    Eigen::Matrix<double, 8, 1> B;
    B << 1;


    //构建过程噪声协方差矩阵
    Eigen::Matrix<double, 8, 8> Q;
        //过程噪声驱动矩阵
        Eigen::Matrix<double, 8, 4> G_Q;
        G_Q <<
            0.5*dt*dt,  0,         0,
            0,          0.5*dt*dt, 0,
            0,          0,         0.5*dt*dt,
            dt,         0,         0,
            0,          dt,        0,
            0,          0,         dt;
    
    Q = G_Q * this->Q_src * G_Q.transpose();

    //更新状态向量为先验状态向量
    kalman_.StateVector() = A * kalman_.StateVector();

    //更新状态向量协方差矩阵
    auto& P = kalman_.Covariance_State();
    P = A * P * A.transpose() + Q;

    //更新观测向量协方差矩阵
    auto& R = kalman_.Covariance_Measurement();
    R = this->R_src * error;
    
    return this->kalman_.update(measurement);
}