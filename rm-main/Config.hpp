#pragma once
#include <string>
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
