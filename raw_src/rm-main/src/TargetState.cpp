#include "../include/TargetState.hpp"
#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Core>

// 修复构造函数：从 robot 实例中准确拷贝当前所有状态，初始化 StateTime，并推演当前时间的位姿
RobotState::RobotState(const Robot& robot, const std::chrono::steady_clock::time_point& t) 
    : Speed(robot.Speed),
      Armors(robot.Armors),
      l_diff(robot.l_diff),
      h_diff(robot.h_diff),
      d_theta_1(robot.d_theta_1),
      d_theta_2(robot.d_theta_2),
      d_theta_3(robot.d_theta_3),
      center(robot.center),
      StateTime(t) 
{
    // 初始化时刻（dt = 0）更新一次 ArmorsPosi 矩阵
    this->ArmorsPosi = this->Predict(0.0);
}

Eigen::Matrix<double, 4, 4> RobotState::Predict(double dt)const
{
    const double& w = this->Speed(3,0);

    Eigen::Matrix<double, 4, 4> ans;

    ans(3,0) = std::remainder(this->Armors(0,0) + w*dt, 2.0 * CV_PI);
    ans(3,1) = std::remainder(this->Armors(0,1) + w*dt, 2.0 * CV_PI);
    ans(3,2) = std::remainder(this->Armors(0,2) + w*dt, 2.0 * CV_PI);
    ans(3,3) = std::remainder(this->Armors(0,3) + w*dt, 2.0 * CV_PI);

    //旋转后的位置
    ans.block<2,1>(0,0) = Eigen::Matrix<double,2,1>{this->center(0) + this->Armors(1,0)*std::cos(ans(3,0)), this->center(1) + this->Armors(1,0)*std::sin(ans(3,0))};
    ans.block<2,1>(0,1) = Eigen::Matrix<double,2,1>{this->center(0) + this->Armors(1,1)*std::cos(ans(3,1)), this->center(1) + this->Armors(1,1)*std::sin(ans(3,1))};
    ans.block<2,1>(0,2) = Eigen::Matrix<double,2,1>{this->center(0) + this->Armors(1,2)*std::cos(ans(3,2)), this->center(1) + this->Armors(1,2)*std::sin(ans(3,2))};
    ans.block<2,1>(0,3) = Eigen::Matrix<double,2,1>{this->center(0) + this->Armors(1,3)*std::cos(ans(3,3)), this->center(1) + this->Armors(1,3)*std::sin(ans(3,3))};
    
    //加上平移
    ans.block<1,4>(2,0) = this->Armors.block<1,4>(2,0);
    auto move = this->Speed.block<3,1>(0,0)*dt;
    ans.block<3,1>(0,0) += move;
    ans.block<3,1>(0,1) += move;
    ans.block<3,1>(0,2) += move;
    ans.block<3,1>(0,3) += move;

    return ans;
}

// 补全缺失的根据时间点推演方法
void RobotState::Predict(const std::chrono::steady_clock::time_point& t)
{
    // 计算时间差（秒）
    std::chrono::duration<double> diff = t - this->StateTime;
    double dt = diff.count();
    
    // 1. 基于当前的内部基准状态计算推演后的 ArmorsPosi 四块装甲板具体位姿
    this->ArmorsPosi = this->Predict(dt);
    
    // 2. 将核心状态(时间、中心点、装甲板角度)前进到新的时间点 t
    this->StateTime = t;
    
    // 更新中心点坐标
    this->center += this->Speed.block<3,1>(0,0) * dt;
    
    // 更新四块装甲板的基础朝向角
    double w = this->Speed(3,0);
    this->Armors(0,0) = std::remainder(this->Armors(0,0) + w * dt, 2.0 * CV_PI);
    this->Armors(0,1) = std::remainder(this->Armors(0,1) + w * dt, 2.0 * CV_PI);
    this->Armors(0,2) = std::remainder(this->Armors(0,2) + w * dt, 2.0 * CV_PI);
    this->Armors(0,3) = std::remainder(this->Armors(0,3) + w * dt, 2.0 * CV_PI);

    //更新四块装甲板的 Z 轴高度
    const double& vz = this->Speed(2,0);
    this->Armors(2,0) += vz * dt;
    this->Armors(2,1) += vz * dt;
    this->Armors(2,2) += vz * dt;
    this->Armors(2,3) += vz * dt;
}