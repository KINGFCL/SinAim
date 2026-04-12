#ifndef SOLVER_CLASS_INCLUDE
#define SOLVER_CLASS_INCLUDE
#include "Armor.hpp"
#include "string"
#include <array>
#include <deque>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
class Solver
{
public:
    struct SolverConfig{
        std::array<double, 9> camera_matrix;
        std::array<double, 5> distortion_coeffs;
        std::array<double, 9> R_Cam_to_gripper;
        std::array<double, 3> T_Cam_to_gripper;
        double err_threshold = 1.0;
    };

    explicit Solver( const SolverConfig& config );
    
    //解算传入的所有装甲板并返回相机坐标系下的结果
    std::vector< std::array<ArmorPosi,2> > operator () (const std::deque<CVArmor>& armors);
    
    //解算传入的所有装甲板并返回相机坐标系下的结果
    std::vector< std::array<ArmorPosi,2> > operator () (const std::vector<CVArmor>& armors);

    //解算传入的所有装甲板并返回世界坐标系下的结果
    std::vector< std::array<ArmorPosi,2> > operator () (const std::vector<CVArmor>& armors, const Eigen::Quaterniond& gripper_to_world);

    //解算传入的所有装甲板并返回相机坐标系下的结果
    std::vector< ArmorPosi > operator () (const std::vector<YoloArmor>& armors);

    //解算传入的所有装甲板并返回世界坐标系下的结果
    std::vector< ArmorPosi > operator () (const std::vector<YoloArmor>& armors, const Eigen::Quaterniond& gripper_to_world);
    
    //解算单个装甲板的位置返回相机坐标系下的结果
    std::array<ArmorPosi,2> operator () (const CVArmor& armor);

    //解算单个装甲板的位置返回世界坐标系下的结果
    std::array<ArmorPosi,2> operator () (const CVArmor& armor, const Eigen::Quaterniond& gripper_to_world);

    //坐标系变换
    void ConverToWorld(std::array<ArmorPosi,2>& armor_posis, const Eigen::Quaterniond& gripper_to_world);
    void ConverToWorld(std::vector< std::array<ArmorPosi,2> >& armors_posis, const Eigen::Quaterniond& gripper_to_world);

    void ConverToWorld(ArmorPosi& armor_posi, const Eigen::Quaterniond& gripper_to_world);
    void ConverToWorld(std::vector<ArmorPosi>& armors_posi, const Eigen::Quaterniond& gripper_to_world);    


    bool IsInCameraRange(const Eigen::Matrix<double, 3, 1>& center,const Eigen::Matrix3d& R,const double error) const;
    bool IsInWorldRange(const ArmorPosi& armor) const;
    bool IsInWorldRange(const Eigen::Matrix<double, 3, 1>& center,const Eigen::Matrix3d& R) const;
    //选择在世界系下距离枪管最近的num个装甲板
    // void Filter(std::vector< std::array<ArmorPosi,2> >& armors_posis,
    //             const Eigen::Matrix<double, 3, 1>& Gun,
    //             const size_t num = 1);
                
    // void FilterAndConverToWorld(std::vector<ArmorPosi>& armors_posi,
    //             const Eigen::Quaterniond& gripper_to_world,
    //             const Eigen::Matrix<double, 3, 1>& Gun,
    //             const size_t num = 1);

    void ansShow(const Eigen::Matrix<double, 3, 1>& posi,cv::Mat& image);
    //void ansShow(const ArmorPosi& armor,cv::Mat& image);

private:
    cv::Mat_<double> cameraMatrix;
    cv::Mat_<double> distCoeffs;

    Eigen::Matrix<double, 3, 3> R_Cam_to_gripper;
    Eigen::Matrix<double, 3, 1> T_Cam_to_gripper;

    std::vector<cv::Point3d> 
        objectBigArmorP, objectSmallArmorP;
    
    const double err_threshold; // 从投影阈值，需根据实际情况调整
    
        struct Range
    {
        double distance_max;
        double high_max, high_min;

        double yaw_max, pitch_max, pitch_min, roll_max;
    };

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
        .yaw_max = 2*M_PI,       // 最大可见偏航角，单位：rad, 2*M_PI表示无限
        .pitch_max = 35.0/180*M_PI,     // 最大可见俯仰角，单位：rad
        .pitch_min = -10.0/180*M_PI,     // 最大可见俯仰角，单位：rad
        .roll_max = 45.0/180*M_PI       // 最大可见滚转角，单位：rad
    };

    const Range outpost_range{
        .distance_max = 800.0,  // 最大可见距离，单位：厘米
        .high_max = 200.0,  // 最大可见高度，单位：厘米,1e10表示无限
        .high_min = -50.0, // 最小可见高度，单位：厘米, -1e10表示无限
        .yaw_max = 2*M_PI,       // 最大可见偏航角，单位：rad, 2*M_PI表示无限
        .pitch_max = 35.0/180*M_PI,     // 最大可见俯仰角，单位：rad
        .pitch_min = -10.0/180*M_PI,     // 最大可见俯仰角，单位：rad
        .roll_max = 45.0/180*M_PI       // 最大可见滚转角，单位：rad
    };

    const Range robot_range{
        .distance_max = 800.0,  // 最大可见距离，单位：厘米
        .high_max = 200.0,  // 最大可见高度，单位：厘米,1e10表示无限
        .high_min = -50.0, // 最小可见高度，单位：厘米, -1e10表示无限
        .yaw_max = 2*M_PI,       // 最大可见偏航角，单位：rad, 2*M_PI表示无限
        .pitch_max = 35.0/180*M_PI,     // 最大可见俯仰角，单位：rad
        .pitch_min = -10.0/180*M_PI,     // 最大可见俯仰角，单位：rad
        .roll_max = 45.0/180*M_PI       // 最大可见滚转角，单位：rad
    };

};

#endif