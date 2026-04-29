#ifndef SOLVER_CLASS_INCLUDE
#define SOLVER_CLASS_INCLUDE
#include "Armor.hpp"
#include <array>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
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
        double reproj_threshold = 1;
    };

    explicit Solver( const SolverConfig& config );

    //解算传入的所有装甲板并返回世界坐标系下的结果
    std::vector< std::array<ArmorPosi,2> > operator () (const std::vector<CVArmor>& armors, const Eigen::Quaterniond& gripper_to_world);

private:
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;

    Eigen::Matrix<double, 3, 3> R_Cam_to_gripper;
    Eigen::Matrix<double, 3, 1> T_Cam_to_gripper;

    struct Range{
        double max_high = 50.0; // 最大高度，单位：厘米
        double min_high = -20.0; // 最小高度，单位：厘米
        double max_distence = 800.0; // 最大距离，单位：厘米
    } const range;

    const double reproj_threshold;

    // 角点式物体坐标系，原点在左上角
    std::vector<cv::Point3f> objectSmallArmorP{
        {0,0,0},{13.5f,0,0},{13.5f,5.5f,0},{0,5.5f,0}};
    std::vector<cv::Point3f> objectBigArmorP{
        {0,0,0},{23.0f,0,0},{23.0f,5.5f,0},{0,5.5f,0}};

    // 装甲板中心在物体坐标系中的位置
    cv::Mat_<double> SmallArmorCenter = (cv::Mat_<double>(3,1) << 6.75, 2.75, 0.0);
    cv::Mat_<double> BigArmorCenter   = (cv::Mat_<double>(3,1) << 11.5, 2.75, 0.0);

};

#endif