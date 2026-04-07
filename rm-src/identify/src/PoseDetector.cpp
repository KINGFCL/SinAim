#include "PoseDetector.hpp"
#include <cmath>
#include <opencv2/core/mat.hpp>

PoseDetector::PoseDetector(double err_threshold) : err_threshold(err_threshold){}

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

        result[ans_posis.size()][0] = IsInCameraRange(posismall);
        result[ans_posis.size()][1] = IsInCameraRange(posibig);

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
    Eigen::Matrix<double, 3, 1> Gun_xy = Eigen::Matrix<double, 3, 1>(Gun[0], Gun[1], 0).normalized(); // 枪管方向向量
    const cv::Point3d Gun_cv(Gun_xy[0], Gun_xy[1], 0); // 转换为 OpenCV 的 Point3d


    for (size_t i = 0; i < armors_posis.size(); i++) {
        if (armors_pattern[i].empty()) continue; // 跳过无效图案

        auto &posismall = armors_posis[i][0];
        auto &posibig = armors_posis[i][1];


        result[ans_posis.size()][0] = IsInWorldRange(posismall, Gun, Type::WORLD_RANGE) && PosePassHax[i][0];
        result[ans_posis.size()][1] = IsInWorldRange(posibig, Gun, Type::WORLD_RANGE) && PosePassHax[i][1];

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

bool PoseDetector::IsInCameraRange(const ArmorPosi& posi) const
{
    //检查重投影
    if (posi.error > this->err_threshold) return false;

    // 检查距离
    if (posi.SCS.x > this->camera_range.distance_max) return false;

    // 计算偏航角、俯仰角和滚转角
    //pitch:
    double pitch = std::asin(posi.face.y);
    if(pitch > this->camera_range.pitch_max || pitch < this->camera_range.pitch_min) return false;

    //yaw:
    double yaw = std::atan2(posi.face.x, posi.face.z);
    yaw = std::abs(yaw);
    if(yaw > this->camera_range.yaw_max) return false;

    //roll:
    double roll = std::asin(posi.toward.y);
    roll = std::abs(roll);
    if(roll > this->camera_range.roll_max) return false;

    return true; // 满足所有条件，认为在范围内
}

bool PoseDetector::IsInWorldRange(const ArmorPosi& posi, Eigen::Matrix<double, 3, 1> Gun, Type type) const
{

    // 检查距离
    if (cv::norm(posi.posi) > this->world_range.distance_max) return false;

    const Range* range_ptr = nullptr;

    switch (type) {
        case Type::WORLD_RANGE:
            range_ptr = &this->world_range;
            break;
        case Type::OUTPOST_RANGE:
            range_ptr = &this->outpost_range;
            break;
        case Type::BASE_RANGE:
            range_ptr = &this->base_range;
            break;
        case Type::ROBOT_RANGE:
            range_ptr = &this->robot_range;
            break;
        default:
            return false;
    }

    const Range& current_range = *range_ptr;
    //高度检查
    if (posi.posi.z > current_range.high_max || posi.posi.z < current_range.high_min) return false;

    // 计算偏航角、俯仰角和滚转角
    //pitch:
    double pitch = std::asin(-posi.face.z);
    if(pitch > current_range.pitch_max || pitch < current_range.pitch_min) return false;

    //yaw:
    auto Gun_xy = Eigen::Matrix<double, 3, 1>(Gun[0], Gun[1], 0).normalized(); // 枪管方向向量  
    auto face_xy = Eigen::Matrix<double, 3, 1>(posi.face.x, posi.face.y, 0).normalized();
    double yaw = std::acos(Gun_xy.dot(face_xy));
    if(yaw > current_range.yaw_max) return false;

    //roll:
    double roll = std::asin(posi.toward.z);
    roll = std::abs(roll);
    if(roll > current_range.roll_max) return false;

    return true; // 满足所有条件，认为在范围内
}


void PoseDetector::operator()(std::vector<ArmorPosi>& armors_posis, const cv::Quatd& gripper_to_world) const
{

}

