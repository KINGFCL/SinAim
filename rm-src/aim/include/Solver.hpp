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

    void ansShow(const Eigen::Matrix<double, 3, 1>& posi,cv::Mat& image);
    //void ansShow(const ArmorPosi& armor,cv::Mat& image);

private:
    cv::Mat_<double> cameraMatrix;
    cv::Mat_<double> distCoeffs;

    Eigen::Matrix<double, 3, 3> R_Cam_to_gripper;
    Eigen::Matrix<double, 3, 1> T_Cam_to_gripper;

    const double reproj_threshold;// 从投影阈值，需根据实际情况调整
    
    const double w_big = 23.0;
    const double w_small = 13.5;
    const double h = 5.5;

    const double beta = 0.0;
    const double gamma = 15.0 * M_PI / 180.0;

};

#endif