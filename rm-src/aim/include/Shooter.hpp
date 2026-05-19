#ifndef INCLUD_SHOOTER_CLASS_HPP
#define INCLUD_SHOOTER_CLASS_HPP


#include <eigen3/Eigen/Geometry>
#include "Target.hpp"
#include "TargetState.hpp"
#include <array>
#include <cmath>
#include "iostream"

class Shooter
{

private:

const Eigen::Matrix<double, 3, 1> toward;
const double toward_pitch,toward_yaw;
const double GRAVITY =  9.8, BULLET_SPEED  = 19.3; //重力加速度9.8m/s^2 弹速 22.8m/s;
const double BULLET_SPEED_2 = 2.0 * BULLET_SPEED * BULLET_SPEED;

double low_speed_delay_time_, high_speed_delay_time_, decision_v_speed_;

public:
    struct ShooterConfig
    {
        double toward_yaw, toward_pitch;
        double low_speed_delay_time_shooter, high_speed_delay_time_shooter, decision_v_speed_shooter;
    };

    //只留一个构造函数，避免冲突
    // Shooter(const Eigen::Matrix<double, 3, 1>& Vector):
    //         toward(Vector.normalized()), 
    //         toward_pitch(std::asin(toward(2,0))),
    //         toward_yaw(std::atan2(toward(1,0), toward(0,0)))
    //         {}
    
    // Shooter(double toward_yaw, double toward_pitch):
    // toward( std::cos(toward_pitch)*std::cos(toward_yaw), std::cos(toward_pitch)*std::sin(toward_yaw) , std::sin(toward_pitch) ),
    // toward_pitch(toward_pitch),
    // toward_yaw(toward_yaw){};

    explicit Shooter(const ShooterConfig& config):
    low_speed_delay_time_(config.low_speed_delay_time_shooter), 
    high_speed_delay_time_(config.high_speed_delay_time_shooter), 
    decision_v_speed_(config.decision_v_speed_shooter),
    toward( std::cos(config.toward_pitch)*std::cos(config.toward_yaw), std::cos(config.toward_pitch)*std::sin(config.toward_yaw) , std::sin(config.toward_pitch) ),
    toward_pitch(config.toward_pitch),
    toward_yaw(config.toward_yaw){};
    
    std::array<double, 2> operator () (const std::unique_ptr<RobotState>& target_ptr) const 
    {
        if(target_ptr == nullptr) return {0.0,0.0};
        RobotState copy_target = *target_ptr;
        double delay_time = std::abs( (target_ptr->Speed.block<3,1>(0,0)*0.01).norm() ) > decision_v_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;
        copy_target.Predict(std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6)));
        double fly_time = this->FlyTime(copy_target.center);
        Eigen::Vector3d aim = copy_target.center + copy_target.Speed.block<3, 1>(0, 0) * fly_time;
        return this->operator()(aim);
    }


    /**
    @brief 计算射击目标，枪口需要转动的pitch和yaw(单位：rad)
    @param ShootPosi 射击位置的坐标(单位：cm)
    @return a array that array[0]: pitch, array[1]: yaw(单位：rad)
    */
    std::array<double, 2> operator () (const Eigen::Matrix<double, 3, 1>& ShootPosi) const
    {

        // 1. 计算target向量 Yaw (偏航角)

        // atan2(y, x) 算出的是向量在水平面投影与 X轴 的夹角
        double shoot_yaw = std::atan2(ShootPosi(1,0), ShootPosi(0,0));


        // 2. 计算target向量 Pitch (俯仰角）
        //通过查表进行计算，算出的是向量与水平面的夹角
        
        //计算水平距离(单位：m)
        const double distance_2 = (ShootPosi(0,0) * ShootPosi(0,0) + ShootPosi(1,0) * ShootPosi(1,0))/10000.0;
        const double distance = std::sqrt(distance_2);
        const double high = ShootPosi(2,0)/100.0; 

        double A = (GRAVITY * distance_2 ) / BULLET_SPEED_2;
        double B = -distance;
        double C = high + A;
        double delta = distance_2 - 4.0 * A * C;

        double ideal_pitch_rad = 0.0;
        if (delta >= 0.0f) {
            // 取小根，即较平缓的弹道
            double u = (-B - std::sqrt(delta)) / (2.0 * A);
            ideal_pitch_rad = std::atan(u);
        } else {
            // 目标超出物理极限射程，返回直线瞄准角度作为降级处理
            ideal_pitch_rad = std::atan2(high, distance);
        }

        double shoot_pitch = ideal_pitch_rad;

        // 3. 计算差值 (需要的旋转量)
        double delta_yaw   = shoot_yaw - this->toward_yaw;
        double delta_pitch = shoot_pitch - this->toward_pitch;

        // std::cout<<"toward "<<toward_pitch<<" shoot pitch "<<shoot_pitch<<" ideal pitch "<<ideal_pitch_rad<<" air_comp "<<air_comp_rad<<"\n";
// std::cout<<shoot_pitch<<"\n";
        // 5. 角度归一化 (关键步骤)
        // 处理跨越 ±180 度的情况，保证走最短路径
        // 例如：从 -170度 转到 +170度，应该是转 -20度，而不是 +340度
        delta_yaw = std::remainder(delta_yaw, 2.0 * M_PI);
        
        // Pitch 一般受限在 ±90 度以内，通常不需要归一化，但为了通用性可以加上
        delta_pitch = std::remainder(delta_pitch, 2.0 * M_PI);

        return {delta_pitch, delta_yaw};
    }

    /**
    @brief 计算从发射到击中目标的飞行时间
    @param Posi 射击位置的坐标(单位：cm)
    @return 返回弹丸需要飞行的时间(单位：s)
    */
    double FlyTime(const Eigen::Matrix<double, 3, 1>& Posi) const
    {
        // 1. 单位换算：将传入的 cm 转换为物理计算标准的 m
        const double distance_2 = (Posi(0,0) * Posi(0,0) + Posi(1,0)  * Posi(1,0) ) / 10000.0;
        const double distance = std::sqrt(distance_2);
        const double high = Posi(2,0)  / 100.0;

        // 2. 估算目标点的理想俯仰角 pitch_rad (必须先算出角度，才能求水平速度分量)
        double A = (GRAVITY * distance_2) / BULLET_SPEED_2;
        double B = -distance;
        double C = high + A;
        double delta = distance_2 - 4.0 * A * C;

        // 3. 计算考虑空气阻力的真实飞行时间
        double ideal_pitch_rad = 0.0;
        if (delta >= 0.0) {
            // 取小根，优先走直射平缓弹道
            ideal_pitch_rad = std::atan((-B - std::sqrt(delta)) / (2.0 * A));
        } else {
            // 超出射程，退化为直线仰角
            ideal_pitch_rad = std::atan2(high, distance);
        }
        
        // 提取枪口的水平初速度方向余弦，做极小值保护防止意外除以 0
        double cos_theta = std::cos(ideal_pitch_rad);
        if (cos_theta < 0.01) cos_theta = 0.01; 

        // 套用降维后的平射时间解析解
        double t = distance / (this->BULLET_SPEED * cos_theta);
        
        return t;
    }


// 辅助函数：将角度限制在 [-PI, PI] 之间
    double NormalizeAngle(double angle) const
    {
        return std::remainder(angle, 2.0 * M_PI);
    }

    Eigen::Matrix<double, 3, 1>  GunDirection (const Eigen::Quaterniond& gripper_to_world) const
    {
        return gripper_to_world * this->toward;
    }

};



#endif