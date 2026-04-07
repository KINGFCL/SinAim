#include "PoseDetector.hpp"
std::vector<std::array<bool,2>> 
        PoseDetector::InCamera(std::vector<std::array<ArmorPosi,2>>& armors_posis, 
                                std::vector<cv::Mat>& armors_pattern) const
{
    std::vector<std::array<bool,2>> result(armors_posis.size(), {false, false});
    std::vector<std::array<ArmorPosi,2>> ans_posis;
    std::vector<cv::Mat> ans_pattern;
    ans_posis.reserve(armors_posis.size());
    ans_pattern.reserve(armors_pattern.size());

    for (size_t i = 0; i < armors_posis.size(); i++) {
        if (armors_pattern[i].empty()) continue; // 跳过无效图案

        auto &posismall = armors_posis[i][0];
        auto &posibig = armors_posis[i][1];

        result[ans_posis.size()][0] = InRange(posismall, Type::CAMERA_RANGE, cv::Point3d(0,0,1));
        result[ans_posis.size()][1] = InRange(posibig, Type::CAMERA_RANGE, cv::Point3d(0,0,1));

        if( !(result[ans_posis.size()][0] || result[ans_posis.size()][1]) ) {
            continue; // 如果两者都不在范围内，跳过
        }

        ans_posis.push_back(armors_posis[i]);
        ans_pattern.push_back(armors_pattern[i]);
    }
    armors_posis = std::move(ans_posis);
    armors_pattern = std::move(ans_pattern);
    result.resize(armors_posis.size());

    return result;

}

std::vector<std::array<bool,2>> 
        PoseDetector::InWorld(std::vector<std::array<ArmorPosi,2>>& armors_posis, 
                                std::vector<cv::Mat>& armors_pattern, 
                                const std::vector<std::array<bool, 2>>& PosePassHax, 
                                const Eigen::Matrix<double, 3, 1>& Gun) const
{
    std::vector<std::array<bool,2>> result(armors_posis.size(), {false, false});
    std::vector<std::array<ArmorPosi,2>> ans_posis;
    std::vector<cv::Mat> ans_pattern;
    ans_posis.reserve(armors_posis.size());
    ans_pattern.reserve(armors_pattern.size());
    const cv::Point3d Gun_cv = cv::Point3d(Gun(0), Gun(1), 0); // 枪管方向向量

    for (size_t i = 0; i < armors_posis.size(); i++) {
        if (armors_pattern[i].empty()) continue; // 跳过无效图案

        auto &posismall = armors_posis[i][0];
        auto &posibig = armors_posis[i][1];


        result[ans_posis.size()][0] = InRange(posismall, Type::WORLD_RANGE, Gun_cv) && PosePassHax[i][0];
        result[ans_posis.size()][1] = InRange(posibig, Type::WORLD_RANGE, Gun_cv) && PosePassHax[i][1];

        if( !(result[ans_posis.size()][0] || result[ans_posis.size()][1]) ) {
            continue; // 如果两者都不在范围内，跳过
        }

        ans_posis.push_back(armors_posis[i]);
        ans_pattern.push_back(armors_pattern[i]);
    }
    armors_posis = std::move(ans_posis);
    armors_pattern = std::move(ans_pattern);
    result.resize(armors_posis.size());

    return result;

}

void PoseDetector::operator()(std::vector<ArmorPosi>& armors_posis, const cv::Quatd& gripper_to_world) const
{

}

