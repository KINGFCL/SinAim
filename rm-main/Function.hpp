#ifndef INCLUDE_FUNCTION_HPP
#define INCLUDE_FUNCTION_HPP

#include "Aim"
#include "Camera"
#include "Identify"
#include "Communicate"
#include "Shooter.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <deque>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <memory>
#include <opencv2/core/types.hpp>
#include <thread>
#include <vector>

//IMU与图像配对线程逻辑
namespace rm
{
    void IMUAndImageMatchFunction(io::HikCamera& Hik, io::RTSerial<Packet>& ser,FastQueue<FrameData>& Frames);
    
    template <std::size_t BufferSize>
    void IMUAndImageMatchFunction(io::HikCamera& Hik, io::LibXRSerial<BufferSize>& ser, FastQueue<FrameData>& Frames);

    void MPCPlanFunction(MPC::Planner& planner, FastQueue<std::unique_ptr<RobotState>>& RobotStates, io::RTSerial<Packet>& ser);

    void MPCPlanFunction(MPC::Planner &planner, FastQueue<std::unique_ptr<RobotState>> &RobotStates, io::RTSerial<Packet> &ser, const Shooter& shooter);

    template <std::size_t BufferSize>
    void MPCPlanFunction(MPC::Planner &planner, FastQueue<std::unique_ptr<RobotState>> &RobotStates, io::LibXRSerial<BufferSize> &ser, const Shooter& shooter);
    template <std::size_t BufferSize>
    void MPCPlanFunction(MPC::Planner &planner, FastQueue<std::unique_ptr<RobotState>> &RobotStates, FastQueue<std::unique_ptr<OutPustState>> &OutPustStates, io::LibXRSerial<BufferSize> &ser, const Shooter& shooter);

    void SendMessageToRobot(io::RTSerial<Packet>& ser, float pitch, float yaw, bool fire);

    template <std::size_t BufferSize>
    void SendMessageToRobot(io::LibXRSerial<BufferSize>& ser, float pitch, float yaw, bool fire);

    template <std::size_t BufferSize>
    void SendMessageToRobot(io::LibXRSerial<BufferSize>& ser, const MPC::Plan& plan, bool fire);
    Eigen::Matrix<double, 4, 1> ChooseBestAimArmor(const Eigen::Matrix<double, 4, 4>& aims,
                                                   const Eigen::Matrix<double,4, 1>& Speed,
                                                   const Eigen::Matrix<double, 3, 1>& Gun);
    std::deque<CVArmor> FilterCenterArmor(const std::deque<std::array<ArmorPosi,2>>& armors_posis, const cv::Point3d& Gun, int num = 1);
    double SolveDt(const std::chrono::steady_clock::time_point& start, const std::chrono::steady_clock::time_point& end, double pic);

    std::vector<YoloArmor> MatchYoloAndOpenCV(const std::vector<CVArmor>& armors,const std::vector<YoloArmor>& yolo_armors);
    bool CheckFireCondition(const cv::Quatd& gripper_to_world,
                            const std::array<double, 2>& Pitch_Yaw,
                            const Eigen::Matrix<double, 4,1>& aim,
                            const Eigen::Matrix<double, 3, 1>& Gun,
                            double pitch_thresh = 0.003, double yaw_thresh = 0.003, double dist_thresh = M_PI/4);

    // Demo 可视化：将 solver 解算结果投影回图像
    // armors_2: Solver 输出（每个 CVArmor 对应 [small, big] 两个 ArmorPosi）
    // armors:   ResNet 分类结果
    // quat:     当前帧姿态四元数
    // solver_config: 相机内参 + 外参
    cv::Mat DrawSolverArmors(const cv::Mat& image,
                             const std::vector<std::array<ArmorPosi, 2>>& armors_2,
                             const std::vector<ArmorPosi>& armors,
                             const cv::Quatd& quat,
                             const Solver::SolverConfig& solver_config,
                             Tracker::State state,
                             const Robot* current_robot);
}

template <std::size_t BufferSize>
void rm::IMUAndImageMatchFunction(io::HikCamera& Hik, io::LibXRSerial<BufferSize>& ser, FastQueue<FrameData>& Frames)
{
    while (true) {
        io::HikCamera::ImageData HikData;

        Hik.read(HikData);
        if (HikData.image.empty()) continue;

        std::chrono::steady_clock::time_point time;
        Eigen::Quaterniond gripper_to_world;
        while (true) {
            bool ret = ser.ReadData(gripper_to_world, time);
            if (!ret) break;
            // std::cout << "have uart\n";

            const auto& t = (static_cast<double>((HikData.time - time).count())) * 1e-6;

            if (t > 8) continue;
            if (t < 5) break;

            FrameData frame(HikData.image, gripper_to_world, HikData.time);
            Frames.push(frame);
            break;
        }
    }
}

template <std::size_t BufferSize>
void rm::MPCPlanFunction(MPC::Planner& planner,
                         FastQueue<std::unique_ptr<RobotState>>& RobotStates,
                         FastQueue<std::unique_ptr<OutPustState>>& OutPustStates,
                         io::LibXRSerial<BufferSize>& ser,
                         const Shooter& shoot)
{
    auto next_time = std::chrono::steady_clock::now();
    const auto PERIOD = std::chrono::milliseconds(10);

    while (true) {
        next_time += PERIOD;
        // std::cout << "cout have live\n";
        auto now = std::chrono::steady_clock::now();
        if (now >= next_time) {
            std::cerr << "[Warning] MPC Loop Missed Deadline!\n";
            next_time = now;
            continue;
        }

        while (RobotStates.size() > 1) {
            RobotStates.pop();
        }

        const std::unique_ptr<RobotState>* robot_ptr = RobotStates.peek();
        if (robot_ptr != nullptr && *robot_ptr != nullptr) {
            Robot::KalmanMode mode = (*robot_ptr)->Mode;
            if (mode == Robot::KalmanMode::EKF) {
                MPC::Plan plan = planner.plan(*robot_ptr, 19.3);
                rm::SendMessageToRobot(ser, plan, plan.fire);
            } else {
                std::array<double, 2> pitch_and_yaw = shoot(*robot_ptr);
                MPC::Plan plan;
                plan.pitch = pitch_and_yaw[0];
                plan.yaw = pitch_and_yaw[1];
                rm::SendMessageToRobot(ser, plan, true);
            }
            std::this_thread::sleep_until(next_time);
            continue;
        }

        while (OutPustStates.size() > 1) {
            OutPustStates.pop();
        }

        const std::unique_ptr<OutPustState>* outpust_ptr = OutPustStates.peek();
        if (outpust_ptr != nullptr && *outpust_ptr != nullptr) {
            MPC::Plan plan = planner.plan(*outpust_ptr, 19.0);
            rm::SendMessageToRobot(ser, plan, plan.fire);
            std::this_thread::sleep_until(next_time);
            continue;
        }

        rm::SendMessageToRobot(ser, 0.0, 0.0, false);
        std::this_thread::sleep_until(next_time);
    }
}

template <std::size_t BufferSize>
void rm::MPCPlanFunction(MPC::Planner& planner,
                         FastQueue<std::unique_ptr<RobotState>>& RobotStates,
                         io::LibXRSerial<BufferSize>& ser,
                         const Shooter& shoot)
{
    auto next_time = std::chrono::steady_clock::now();
    const auto PERIOD = std::chrono::milliseconds(10);

    while (true) {
        next_time += PERIOD;

        auto now = std::chrono::steady_clock::now();
        if (now >= next_time) {
            std::cerr << "[Warning] MPC Loop Missed Deadline!\n";
            next_time = now;
            continue;
        }

        while (RobotStates.size() > 1) {
            RobotStates.pop();
        }

        const std::unique_ptr<RobotState>* target_ptr = RobotStates.peek();
        if (target_ptr == nullptr || *target_ptr == nullptr) {
            rm::SendMessageToRobot(ser, 0.0, 0.0, false);
            std::this_thread::sleep_until(next_time);
            continue;
        }

        Robot::KalmanMode mode = (*target_ptr)->Mode;
        if (mode == Robot::KalmanMode::EKF) {
            MPC::Plan plan = planner.plan(*target_ptr, 22.0);
            rm::SendMessageToRobot(ser, plan, plan.fire);
            std::this_thread::sleep_until(next_time);
        } else {
            double fly_time = shoot.FlyTime((*target_ptr)->center);
            Eigen::Vector3d aim = (*target_ptr)->center + (*target_ptr)->Speed.block<3, 1>(0, 0) * fly_time;

            std::array<double, 2> pitch_and_yaw = shoot(aim);
            MPC::Plan plan;
            plan.pitch = pitch_and_yaw[0];
            plan.yaw = pitch_and_yaw[1];
            rm::SendMessageToRobot(ser, plan, true);
            std::this_thread::sleep_until(next_time);
        }
    }
}

template <std::size_t BufferSize>
void rm::SendMessageToRobot(io::LibXRSerial<BufferSize>& ser, float pitch, float yaw, bool fire)
{
    MPC::Plan plan;
    plan.pitch = pitch;
    plan.yaw = yaw;
    rm::SendMessageToRobot(ser, plan, fire);
}

template <std::size_t BufferSize>
void rm::SendMessageToRobot(io::LibXRSerial<BufferSize>& ser, const MPC::Plan& plan, bool fire)
{
    io::mcu::HostGimbalTarget target;
    io::mcu::HostFireNotify fire_notify;

    target.yaw = plan.yaw;
    target.pit = plan.pitch;

    target.yaw_dot = plan.yaw_vel;
    target.pit_dot = plan.pitch_vel;

    target.yaw_ddot = plan.yaw_acc;
    target.pit_ddot = plan.pitch_acc;

    fire_notify.isfire = fire;

    // std::cout << "yaw: " << target.yaw << "pitch: " << target.pit << "\n";

    ser.WriteData(target, fire_notify);
}


#endif
