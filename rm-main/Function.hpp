#ifndef INCLUDE_FUNCTION_HPP
#define INCLUDE_FUNCTION_HPP

#include "Aim"
#include "Camera"
#include "Identify"
#include "Communicate"
#include "Shooter.hpp"

#include <cmath>
#include <deque>
#include <eigen3/Eigen/Core>
#include <opencv2/core/types.hpp>
#include <vector>

//IMU与图像配对线程逻辑
namespace rm
{
    void IMUAndImageMatchFunction(io::HikCamera& Hik, io::RTSerial<Packet>& ser,FastQueue<FrameData>& Frames);
    void IMUAndImageMatchFunction(io::HikCamera& Hik, io::LibXRSerial<>& ser, FastQueue<FrameData>& Frames);

    void MPCPlanFunction(MPC::Planner& planner, FastQueue<std::unique_ptr<RobotState>>& RobotStates, io::RTSerial<Packet>& ser);

    void MPCPlanFunction(MPC::Planner &planner, FastQueue<std::unique_ptr<RobotState>> &RobotStates, io::RTSerial<Packet> &ser, const Shooter& shooter);
    void MPCPlanFunction(MPC::Planner &planner, FastQueue<std::unique_ptr<RobotState>> &RobotStates, io::LibXRSerial<> &ser, const Shooter& shooter);

    void SendMessageToRobot(io::RTSerial<Packet>& ser, float pitch, float yaw, bool fire);
    void SendMessageToRobot(io::LibXRSerial<>& ser, float pitch, float yaw, bool fire);
    void SendMessageToRobot(io::LibXRSerial<>& ser, const MPC::Plan& plan, bool fire);
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


#endif
