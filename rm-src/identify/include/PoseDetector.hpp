#ifndef POSEDETECTOR_HPP
#define POSEDETECTOR_HPP
#include "Armor.hpp"
#include "opencv2/core/quaternion.hpp"
#include <eigen3/Eigen/Core>
#include <opencv2/core/types.hpp>
#include <vector>
class PoseDetector
{
public:
    enum class Type : int{
        CAMERA_RANGE = 0,
        WORLD_RANGE = 1,
        OUTPOST_RANGE = 2,
        BASE_RANGE = 3,
        ROBOT_RANGE = 4
    };
    PoseDetector()=default;
    
    std::vector< std::array<bool,2> > InCamera (std::vector<std::array<ArmorPosi,2>>& armors_posis, std::vector<cv::Mat>& armors_pattern) const;

    std::vector< std::array<bool,2> > InWorld (std::vector<std::array<ArmorPosi,2>>& armors_posis, std::vector<cv::Mat>& armors_pattern, const std::vector<std::array<bool, 2>>& PosePassHax, const Eigen::Matrix<double, 3, 1>& Gun) const;

    bool InRange(const ArmorPosi& posi, Type type,cv::Point3d gripper) const;
    void operator()(std::vector<ArmorPosi>& armors_posis, const cv::Quatd& gripper_to_world) const;
private:
    struct Range
    {
        double distance_max;
        double high_max, high_min;

        double yaw_max, pitch_max, pitch_min, roll_max;
    };
    const double err_threshold = 1.0; // 从投影阈值，需根据实际情况调整

    const Range camera_range{
        .distance_max = 800.0,  // 最大可见距离，单位：厘米
        .high_max = 1e10,  // 最大可见高度，单位：厘米,1e10表示无限
        .high_min = -1e10, // 最小可见高度，单位：厘米, -1e10表示无限
        .yaw_max = 80.0/180*M_PI,       // 最大可见偏航角，单位：rad
        .pitch_max = 60.0/180*M_PI,     // 最大可见俯仰角，单位：rad
        .pitch_min = -10.0/180*M_PI,     // 最大可见俯仰角，单位：rad
        .roll_max = 45.0/180*M_PI       // 最大可见滚转角，单位：rad
    };

    const Range world_range{
        .distance_max = 800.0,  // 最大可见距离，单位：厘米
        .high_max = 200.0,  // 最大可见高度，单位：厘米,1e10表示无限
        .high_min = -50.0, // 最小可见高度，单位：厘米, -1e10表示无限
        .yaw_max = 80.0/180*M_PI,       // 最大可见偏航角，单位：rad
        .pitch_max = 35.0/180*M_PI,     // 最大可见俯仰角，单位：rad
        .pitch_min = -10.0/180*M_PI,     // 最大可见俯仰角，单位：rad
        .roll_max = 45.0/180*M_PI       // 最大可见滚转角，单位：rad
    };

    const Range outpost_range{
        .distance_max = 800.0,  // 最大可见距离，单位：厘米
        .high_max = 200.0,  // 最大可见高度，单位：厘米
        .high_min = 100.0, // 最小可见高度，单位：厘米
        .yaw_max = 80.0/180*M_PI,       // 最大可见偏航角，单位：rad
        .pitch_max = 60.0/180*M_PI,     // 最大可见俯仰角，单位：rad
        .pitch_min = -10.0/180*M_PI,     // 最大可见俯仰角，单位：rad
        .roll_max = 45.0/180*M_PI       // 最大可见滚转角，单位：rad
    };

    const Range base_range{
        .distance_max = 800.0,  // 最大可见距离，单位：厘米
        .high_max = 200.0,  // 最大可见高度，单位：厘米
        .high_min = 100.0, // 最小可见高度，单位：厘米 
        .yaw_max = 80.0/180*M_PI,       // 最大可见偏航角，单位：rad
        .pitch_max = 35.0/180*M_PI,     // 最大可见俯仰角，单位：rad
        .pitch_min = -10.0/180*M_PI,     // 最大可见俯仰角，单位：rad
        .roll_max = 45.0/180*M_PI       // 最大可见滚转角，单位：rad
    };

    const Range robot_range{
        .distance_max = 800.0,  // 最大可见距离，单位：厘米
        .high_max = 50.0,  // 最大可见高度，单位：厘米
        .high_min = -50.0, // 最小可见高度，单位：厘米
        .yaw_max = 80.0/180*M_PI,       // 最大可见偏航角，单位：rad
        .pitch_max = 35.0/180*M_PI,     // 最大可见俯仰角，单位：rad
        .pitch_min = -10.0/180*M_PI,     // 最大可见俯仰角，单位：rad
        .roll_max = 45.0/180*M_PI       // 最大可见滚转角，单位：rad
    };

};

#endif