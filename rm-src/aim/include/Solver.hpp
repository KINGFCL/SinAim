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
        double reproj_threshold = 1.0;
    };

    explicit Solver( const SolverConfig& config );

    //解算传入的所有装甲板并返回世界坐标系下的结果
    std::vector< std::array<ArmorPosi,2> > operator () (const std::vector<CVArmor>& armors, const Eigen::Quaterniond& gripper_to_world);

private:
    cv::Mat_<double> cameraMatrix;
    cv::Mat_<double> distCoeffs;

    Eigen::Matrix<double, 3, 3> R_Cam_to_gripper;
    Eigen::Matrix<double, 3, 1> T_Cam_to_gripper;

    struct Range{
        double max_high = 50.0; // 最大高度，单位：厘米
        double min_high = -20.0; // 最小高度，单位：厘米
        double max_distence = 800.0; // 最大距离，单位：厘米
    } const range;

    const double reproj_threshold;// 从投影阈值，需根据实际情况调整
    
    const double w_big = 23.0;// 大装甲板宽度，单位：厘米

    const double w_small = 13.5; // 小装甲板宽度，单位：厘米

    const double h = 5.5;// 灯条长度，单位：厘米

    const double beta = 0.0;
    const double gamma = 15.0 * M_PI / 180.0;

};

#endif