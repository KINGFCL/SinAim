#include "PoseDetector.hpp"
#include <cmath>
#include <opencv2/core/mat.hpp>

PoseDetector::PoseDetector(double err_threshold) : err_threshold(err_threshold) {}

std::vector<std::array<bool, 2>> 
PoseDetector::InCamera(std::vector<std::array<ArmorPosi, 2>>& armors_posis, 
                       std::vector<cv::Mat>& armors_pattern) const
{
    std::vector<std::array<bool, 2>> ans_result;
    std::vector<std::array<ArmorPosi, 2>> ans_posis;
    std::vector<cv::Mat> ans_pattern;

    // 预分配内存，避免 vector 动态扩容带来的开销
    ans_result.reserve(armors_posis.size());
    ans_posis.reserve(armors_posis.size());
    ans_pattern.reserve(armors_pattern.size());

    for (size_t i = 0; i < armors_posis.size(); ++i) {
        if (armors_pattern[i].empty()) continue; 

        // 提取判断逻辑，解耦状态获取与数组推入
        std::array<bool, 2> valid_flags = {
            IsInCameraRange(armors_posis[i][0]),
            IsInCameraRange(armors_posis[i][1])
        };

        if (!valid_flags[0] && !valid_flags[1]) {
            continue; 
        }

        // 使用 std::move 榨干性能，避免深拷贝装甲板数据和 Mat 图像
        ans_posis.push_back(std::move(armors_posis[i]));
        ans_pattern.push_back(std::move(armors_pattern[i]));
        ans_result.push_back(valid_flags);
    }

    // 更新原始引用
    armors_posis = std::move(ans_posis);
    armors_pattern = std::move(ans_pattern);

    return ans_result;
}

std::vector<std::array<bool, 2>> 
PoseDetector::InWorld(std::vector<std::array<ArmorPosi, 2>>& armors_posis, 
                      std::vector<cv::Mat>& armors_pattern, 
                      const std::vector<std::array<bool, 2>>& PosePassHax, 
                      const Eigen::Matrix<double, 3, 1>& Gun) const
{
    std::vector<std::array<bool, 2>> ans_result;
    std::vector<std::array<ArmorPosi, 2>> ans_posis;
    std::vector<cv::Mat> ans_pattern;

    ans_result.reserve(armors_posis.size());
    ans_posis.reserve(armors_posis.size());
    ans_pattern.reserve(armors_pattern.size());

    for (size_t i = 0; i < armors_posis.size(); ++i) {
        if (armors_pattern[i].empty()) continue; 

        std::array<bool, 2> valid_flags = {
            IsInWorldRange(armors_posis[i][0], Gun, Type::WORLD_RANGE) && PosePassHax[i][0],
            IsInWorldRange(armors_posis[i][1], Gun, Type::WORLD_RANGE) && PosePassHax[i][1]
        };

        if (!valid_flags[0] && !valid_flags[1]) {
            continue; 
        }

        ans_posis.push_back(std::move(armors_posis[i]));
        ans_pattern.push_back(std::move(armors_pattern[i]));
        ans_result.push_back(valid_flags);
    }

    armors_posis = std::move(ans_posis);
    armors_pattern = std::move(ans_pattern);

    return ans_result;
}


bool PoseDetector::IsInWorldRange(const ArmorPosi& posi, Eigen::Matrix<double, 3, 1> Gun, Type type) const
{
    if (cv::norm(posi.posi) > this->world_range.distance_max) return false;

    const Range* range_ptr = nullptr;
    switch (type) {
        case Type::WORLD_RANGE:   range_ptr = &this->world_range;   break;
        case Type::OUTPOST_RANGE: range_ptr = &this->outpost_range; break;
        case Type::BASE_RANGE:    range_ptr = &this->base_range;    break;
        case Type::ROBOT_RANGE:   range_ptr = &this->robot_range;   break;
        default: return false;
    }

    const Range& current_range = *range_ptr;
    
    if (posi.posi.z > current_range.high_max || posi.posi.z < current_range.high_min) return false;

    double pitch = std::asin(std::clamp(-posi.face.z, -1.0, 1.0));
    if (pitch > current_range.pitch_max || pitch < current_range.pitch_min) return false;

    // 使用 Vector3d 使 Eigen 的表达更清晰
    Eigen::Vector3d Gun_xy(Gun[0], Gun[1], 0.0);  
    Eigen::Vector3d face_xy(posi.face.x, posi.face.y, 0.0);
    
    // 性能优化：使用 squaredNorm 替代 norm，避免底层开方运算
    if (Gun_xy.squaredNorm() < 1e-12 || face_xy.squaredNorm() < 1e-12) return false; 
    
    Gun_xy.normalize();
    face_xy.normalize();

    double yaw = std::acos(std::clamp(Gun_xy.dot(face_xy), -1.0, 1.0)); 
    if (yaw > current_range.yaw_max) return false;

    double roll = std::abs(std::asin(std::clamp(posi.toward.z, -1.0, 1.0)));
    if (roll > current_range.roll_max) return false;

    return true; 
}

void PoseDetector::operator()(std::vector<ArmorPosi>& armors_posis) const
{

    for (auto& armor_posi : armors_posis) {

        if(armor_posi.type == ArmorPosi::Type::Unknow) continue; // 如果类型未知，不进行范围检查
        //检查高度
        switch (armor_posi.type) {
            case ArmorPosi::Type::base:
                if (armor_posi.posi.z > this->base_range.high_max || armor_posi.posi.z < this->base_range.high_min) {
                    armor_posi.type = ArmorPosi::Type::Unknow; // 不满足高度条件，类型设为未知
                    continue; 
                }
                break;
            case ArmorPosi::Type::outpost:
                if (armor_posi.posi.z > this->outpost_range.high_max || armor_posi.posi.z < this->outpost_range.high_min) {
                    armor_posi.type = ArmorPosi::Type::Unknow; // 不满足高度条件，类型设为未知
                    continue; 
                }
                break;
            default:
                if (armor_posi.posi.z > this->robot_range.high_max || armor_posi.posi.z < this->robot_range.high_min) {
                    armor_posi.type = ArmorPosi::Type::Unknow; // 不满足高度条件，类型设为未知
                    continue;
                }
                break; 
        }
        
    }
}

