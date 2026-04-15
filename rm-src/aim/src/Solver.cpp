#include "Solver.hpp"
#include "Armor.hpp"
#include "solveRectanglePose.hpp"
// #include <array>
#include <Eigen/src/Core/Matrix.h>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <numeric>

// #define SolverDebug
#ifdef SolverDebug
#include "RerunVisualizer.hpp"
extern RerunVisualizer viz;
#endif

namespace SASize {
    constexpr double LIGHTBAR_LENGTH = 5.50; // 灯条长度，单位：厘米
    constexpr double SMALL_ARMOR_WIDTH = 13.50; // 小装甲板宽度，单位：厘米
    constexpr double BIG_ARMOR_WIDTH = 23.0;   // 大装甲板宽度，单位：厘米
}

Solver::Solver( const SolverConfig& config ):
    cameraMatrix(config.camera_matrix),
    distCoeffs(config.distortion_coeffs),
    R_Cam_to_gripper(config.R_Cam_to_gripper),
    T_Cam_to_gripper(config.T_Cam_to_gripper),
    reproj_threshold(config.reproj_threshold)
{}

std::vector< std::array<ArmorPosi,2> > Solver::operator () (const std::vector<CVArmor>& armors, const Eigen::Quaterniond& gripper_to_world)
{
    std::vector< std::array<ArmorPosi,2> > results;
    if(armors.empty()) return results;
    results.reserve(armors.size());

    Eigen::Matrix3d R_gripper_to_world = gripper_to_world.toRotationMatrix();

    Eigen::Matrix3d R_cam2world =  R_gripper_to_world * this->R_Cam_to_gripper;

    Eigen::Vector3d photocenter_world = R_gripper_to_world * this->T_Cam_to_gripper;

    // 1. 去畸变, 输出归一化平面坐标 (fx=fy=1, cx=cy=0 的理想坐标)
    std::vector<cv::Point2f> undistorted;
    std::array<Eigen::Vector3d, 4> v_in;
    for(const auto& armor:armors)
    {
        cv::undistortPoints(armor.Lightcorners, undistorted, cameraMatrix, distCoeffs);
        
        v_in[0] = R_cam2world * Eigen::Vector3d(undistorted[0].x,undistorted[0].y,1.0);
        v_in[1] = R_cam2world * Eigen::Vector3d(undistorted[1].x,undistorted[1].y,1.0);
        v_in[2] = R_cam2world * Eigen::Vector3d(undistorted[2].x,undistorted[2].y,1.0);
        v_in[3] = R_cam2world * Eigen::Vector3d(undistorted[3].x,undistorted[3].y,1.0);

        std::vector<pose::PoseSolution> solution_small =
            pose::solveRectanglePose(v_in, beta, gamma, this->w_small, this->h);

        std::vector<pose::PoseSolution> solution_big =
            pose::solveRectanglePose(v_in, beta, gamma, this->w_big, this->h);

        Eigen::Matrix<double, 3, 2> center_small;
        Eigen::Matrix<double, 3, 2> center_big;

        std::array<double, 2> yaw_small, yaw_big;
        std::array<double, 2> reproj_small, reproj_big;

        bool isInRange_small = true, isInRange_big = true;

        switch (solution_small.size()) {
            case 1:
                center_small.block<3,1>(0,0) = solution_small[0].t_B2A + photocenter_world;
                center_small.block<3,1>(0,1) = solution_small[0].t_B2A + photocenter_world;
                yaw_small[0] = solution_small[0].yaw;
                yaw_small[1] = solution_small[0].yaw;
                reproj_small[0] = solution_small[0].reproj;
                reproj_small[1] = solution_small[0].reproj;

                //范围判断：
                if(reproj_small[0] > this->reproj_threshold)
                {
                    isInRange_small = false;
                    break;
                }
                if(center_small.col(0).z() > this->range.max_high || center_small.col(0).z() < this->range.min_high)
                {
                    isInRange_small = false;
                    break;
                }
                if(center_small.col(0).norm() > this->range.max_distence)
                {
                    isInRange_small = false;
                    break;
                }
                break;
            case 2:
                center_small.block<3,1>(0,0) = solution_small[0].t_B2A + photocenter_world;
                center_small.block<3,1>(0,1) = solution_small[1].t_B2A + photocenter_world;
                yaw_small[0] = solution_small[0].yaw;
                yaw_small[1] = solution_small[1].yaw;
                reproj_small[0] = solution_small[0].reproj;
                reproj_small[1] = solution_small[1].reproj;

                //范围判断：
                if(reproj_small[0] > this->reproj_threshold || reproj_small[1] > this->reproj_threshold)
                {                    
                    isInRange_small = false;
                    break;
                }
                if(center_small.col(0).z() > this->range.max_high || center_small.col(0).z() < this->range.min_high ||
                   center_small.col(1).z() > this->range.max_high || center_small.col(1).z() < this->range.min_high)
                {
                    isInRange_small = false;
                    break;
                }
                if(center_small.col(0).norm() > this->range.max_distence || center_small.col(1).norm() > this->range.max_distence)
                {
                    isInRange_small = false;
                    break;
                }
                break;

            default:
                isInRange_small = false;
                break;
        }

        switch (solution_small.size()) {
            case 1:
                center_big.block<3,1>(0,0) = solution_big[0].t_B2A + photocenter_world;
                center_big.block<3,1>(0,1) = solution_big[0].t_B2A + photocenter_world;
                yaw_big[0] = solution_big[0].yaw;
                yaw_big[1] = solution_big[0].yaw;
                reproj_big[0] = solution_big[0].reproj;
                reproj_big[1] = solution_big[0].reproj;
                //范围判断：
                if(reproj_big[0] > this->reproj_threshold)
                {                    
                    isInRange_big = false;
                    break;
                }
                if(center_big.col(0).z() > this->range.max_high || center_big.col(0).z() < this->range.min_high)
                {                    
                    isInRange_big = false;
                    break;
                }
                if(center_big.col(0).norm() > this->range.max_distence)
                {                    
                    isInRange_big = false;    
                    break;
                }
                break;
            case 2:
                center_big.block<3,1>(0,0) = solution_big[0].t_B2A + photocenter_world;
                center_big.block<3,1>(0,1) = solution_big[1].t_B2A + photocenter_world;
                yaw_big[0] = solution_big[0].yaw;
                yaw_big[1] = solution_big[1].yaw;
                reproj_big[0] = solution_big[0].reproj;
                reproj_big[1] = solution_big[1].reproj;
                //范围判断：
                if(reproj_big[0] > this->reproj_threshold || reproj_big[1] > this->reproj_threshold)
                {                    
                    isInRange_big = false;
                    break;
                }
                if(center_big.col(0).z() > this->range.max_high || center_big.col(0).z() < this->range.min_high ||
                   center_big.col(1).z() > this->range.max_high || center_big.col(1).z() < this->range.min_high)
                {                    
                    isInRange_big = false;   
                    break;
                }                
                if(center_big.col(0).norm() > this->range.max_distence || center_big.col(1).norm() > this->range.max_distence)
                {                    
                    isInRange_big = false;
                    break;
                }
                break;

            default:
                isInRange_big = false;
                break;
        }

        ArmorPosi armor_small, armor_big;
        if(isInRange_small)
        {
            armor_small = ArmorPosi(center_small, yaw_small, reproj_small, isInRange_small);
        }
        if(isInRange_big)
        {
            armor_big = ArmorPosi(center_big, yaw_big, reproj_big, isInRange_big);
        }
        results.emplace_back(armor_small,armor_big);
    }

    return results;
}

