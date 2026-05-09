#pragma once
#include <string>
#include "Armor.hpp"
#include "CVDetector.hpp"
#include "HikCamera.hpp"
#include "MvObsoleteInterfaces.h"
#include "Shooter.hpp"
#include "Target.hpp"
#include "planner.hpp"
#include "yaml.hpp"
#include "Aim"  // for Solver::SolverConfig

// 从 yaml 文件读取 Solver::SolverConfig
// yaml 格式示例：
//   camera_matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
//   distortion_coeffs: [k1, k2, p1, p2, k3]
//   R_Cam_to_gripper: [r00, r01, r02, r10, r11, r12, r20, r21, r22]
//   T_Cam_to_gripper: [tx, ty, tz]   # 单位：cm
//   reproj_threshold: 1.0
inline Solver::SolverConfig LoadSolverConfig(const std::string& config_path)
{
    YAML::Node yaml = tools::load(config_path);

    Solver::SolverConfig cfg;

    auto cam = tools::read<std::vector<double>>(yaml, "camera_matrix");
    auto dist = tools::read<std::vector<double>>(yaml, "distortion_coeffs");
    auto R = tools::read<std::vector<double>>(yaml, "R_Cam_to_gripper");
    auto T = tools::read<std::vector<double>>(yaml, "T_Cam_to_gripper");

    std::copy_n(cam.begin(),  9, cfg.camera_matrix.begin());
    std::copy_n(dist.begin(), 5, cfg.distortion_coeffs.begin());
    std::copy_n(R.begin(),    9, cfg.R_Cam_to_gripper.begin());
    std::copy_n(T.begin(),    3, cfg.T_Cam_to_gripper.begin());

    if (yaml["reproj_threshold"])
        cfg.reproj_threshold = yaml["reproj_threshold"].as<double>();

    return cfg;
}

inline MPC::Planner::PlannerConfig LoadPlannerConfig(const std::string& config_path)
{
    YAML::Node yaml = tools::load(config_path);

    MPC::Planner::PlannerConfig cfg;

    cfg.yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 180.0 * M_PI;
    cfg.pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 180.0 * M_PI;
    cfg.fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
    cfg.decision_speed_ = tools::read<double>(yaml, "decision_speed");
    cfg.high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
    cfg.low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");

    cfg.max_yaw_acc_ = tools::read<double>(yaml, "max_yaw_acc");
    cfg.Q_yaw_ = tools::read<std::vector<double>>(yaml, "Q_yaw");
    cfg.R_yaw_ = tools::read<std::vector<double>>(yaml, "R_yaw");

    cfg.max_pitch_acc_ = tools::read<double>(yaml, "max_pitch_acc");
    cfg.Q_pitch_ = tools::read<std::vector<double>>(yaml, "Q_pitch");
    cfg.R_pitch_ = tools::read<std::vector<double>>(yaml, "R_pitch");

    return cfg;
}

inline Shooter::ShooterConfig LoadShooterConfig(const std::string& config_path)
{
    YAML::Node yaml = tools::load(config_path);

    Shooter::ShooterConfig cfg;

    cfg.toward_yaw = tools::read<double>(yaml, "yaw_offset") / 180.0 * M_PI;
    cfg.toward_pitch = tools::read<double>(yaml, "pitch_offset") / 180.0 * M_PI;

    cfg.low_speed_delay_time_shooter = tools::read<double>(yaml, "low_speed_delay_time_shooter");
    cfg.high_speed_delay_time_shooter = tools::read<double>(yaml, "high_speed_delay_time_shooter");
    cfg.decision_v_speed_shooter = tools::read<double>(yaml, "decision_v_speed_shooter");

    return cfg;
}

inline CVDetector::CVDetectorConfig LoadCVDetectorConfig(const std::string& config_path)
{
    YAML::Node yaml = tools::load(config_path);

    CVDetector::CVDetectorConfig cfg;

    cfg.color = tools::read<int>(yaml, "color") ;
    cfg.roi_width = tools::read<int>(yaml, "roi_width");
    cfg.roi_height = tools::read<int>(yaml, "roi_height");

    return cfg;
}

inline io::HikCamera::HikCameraConfig LoadHikCameraConfig(const std::string& config_path)
{
    YAML::Node yaml = tools::load(config_path);

    io::HikCamera::HikCameraConfig cfg;

    cfg.exposure_ms = tools::read<int>(yaml, "exposure_ms");
    cfg.gain = tools::read<int>(yaml, "gain");
    cfg.autocap = tools::read<bool>(yaml, "autocap");

    return cfg;
}
inline Robot::RobotConfig LoadRobotConfig(const std::string& config_path)
{
    YAML::Node yaml = tools::load(config_path);

    Robot::RobotConfig cfg;

    cfg.gripper_to_world_matrix = tools::read<std::array<double, 9>>(yaml, "R_Cam_to_gripper");

    return cfg;
}

