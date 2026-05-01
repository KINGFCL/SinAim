#include "Solver.hpp"
#include "Armor.hpp"
#include "eigen3/Eigen/Dense"
#include "iostream"
#include <array>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>

// #define SolverDebug
#ifdef SolverDebug
#include "RerunVisualizer.hpp"
extern RerunVisualizer viz;
#endif


Solver::Solver( const SolverConfig& config ):
    cameraMatrix(cv::Mat(3,3,CV_64FC1,const_cast<double*>(config.camera_matrix.data())).clone() ),
    distCoeffs(cv::Mat(5,1,CV_64FC1,const_cast<double*>(config.distortion_coeffs.data())).clone()),
    R_Cam_to_gripper( Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> >(config.R_Cam_to_gripper.data()) ),
    T_Cam_to_gripper(Eigen::Map<const Eigen::Matrix<double, 3, 1> >(config.T_Cam_to_gripper.data()) ),
    reproj_threshold(config.reproj_threshold)
{
    double hs = h * 0.5;
    double ws = w_small * 0.5;
    double wb = w_big * 0.5;
    objectSmallArmorP = { {-ws, -hs, 0}, {ws, -hs, 0}, {ws, hs, 0}, {-ws, hs, 0} };
    objectBigArmorP   = { {-wb, -hs, 0}, {wb, -hs, 0}, {wb, hs, 0}, {-wb, hs, 0} };
}

std::vector< std::array<ArmorPosi,2> > Solver::operator () (const std::vector<CVArmor>& armors, const Eigen::Quaterniond& gripper_to_world)
{
    std::vector< std::array<ArmorPosi,2> > results;
    if(armors.empty()) return results;
    results.reserve(armors.size());

    Eigen::Matrix3d R_gripper_to_world = gripper_to_world.toRotationMatrix();
    Eigen::Matrix3d R_cam2world = R_gripper_to_world * this->R_Cam_to_gripper;
    Eigen::Vector3d photocenter_world = R_gripper_to_world * this->T_Cam_to_gripper;

    for(const auto& armor : armors)
    {
        Eigen::Matrix<double,3,2> center_small, center_big, center_cam_small, center_cam_big;
        std::array<double,2> yaw_small, yaw_big, reproj_small, reproj_big;
        bool isInRange_small = true, isInRange_big = true;

        // 小装甲板
        {
            std::vector<cv::Mat> rvecs, tvecs;
            std::vector<double> reprojErr;
            cv::solvePnPGeneric(objectSmallArmorP, armor.Lightcorners, cameraMatrix, distCoeffs,
                                rvecs, tvecs, false, cv::SOLVEPNP_IPPE,
                                cv::noArray(), cv::noArray(), reprojErr);
            for(int s = 0; s < 2; s++) {
                cv::Mat R_cv;
                cv::Rodrigues(rvecs[s], R_cv);
                Eigen::Matrix3d R_arm2cam(Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_cv.ptr<double>()));
                Eigen::Vector3d T_cam(Eigen::Map<Eigen::Vector3d>(tvecs[s].ptr<double>()));
                Eigen::Vector3d toward_world = R_cam2world * R_arm2cam.col(2);

                // 2D 叉积判断法向量在视线左/右侧：
                // cross2d < 0 → 法向量在视线右侧
                // cross2d > 0 → 法向量在视线左侧
                double cross2d = T_cam.x() * R_arm2cam(1,2) - T_cam.y() * R_arm2cam(0,2);
                int idx = (cross2d > 0) ? 0 : 1;
                center_cam_small.col(idx) = T_cam;
                center_small.col(idx) = R_cam2world * T_cam + photocenter_world;
                yaw_small[idx] = std::atan2(-toward_world.y(), -toward_world.x());
                reproj_small[idx] = reprojErr[s];
            }
        }
        if(center_small.col(0).z() > range.max_high || center_small.col(0).z() < range.min_high ||
           center_small.col(1).z() > range.max_high || center_small.col(1).z() < range.min_high ||
           (reproj_small[0] > reproj_threshold && reproj_small[1] > reproj_threshold) ||
           center_small.col(0).norm() > range.max_distence || center_small.col(1).norm() > range.max_distence)
            isInRange_small = false;

        // 大装甲板
        {
            std::vector<cv::Mat> rvecs, tvecs;
            std::vector<double> reprojErr;
            cv::solvePnPGeneric(objectBigArmorP, armor.Lightcorners, cameraMatrix, distCoeffs,
                                rvecs, tvecs, false, cv::SOLVEPNP_IPPE,
                                cv::noArray(), cv::noArray(), reprojErr);
            for(int s = 0; s < 2; s++) {
                cv::Mat R_cv;
                cv::Rodrigues(rvecs[s], R_cv);
                Eigen::Matrix3d R_arm2cam(Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_cv.ptr<double>()));
                Eigen::Vector3d T_cam(Eigen::Map<Eigen::Vector3d>(tvecs[s].ptr<double>()));

                Eigen::Vector3d toward_world = R_cam2world * R_arm2cam.col(2);
                double cross2d = T_cam.x() * R_arm2cam(1,2) - T_cam.y() * R_arm2cam(0,2);
                int idx = (cross2d > 0) ? 0 : 1;
                center_cam_big.col(idx) = T_cam;
                center_big.col(idx) = R_cam2world * T_cam + photocenter_world;
                yaw_big[idx] = std::atan2(-toward_world.y(), -toward_world.x());
                reproj_big[idx] = reprojErr[s];
            }
        }
        if(center_big.col(0).z() > range.max_high || center_big.col(0).z() < range.min_high ||
           center_big.col(1).z() > range.max_high || center_big.col(1).z() < range.min_high ||
           (reproj_big[0] > reproj_threshold && reproj_big[1] > reproj_threshold) ||
           center_big.col(0).norm() > range.max_distence || center_big.col(1).norm() > range.max_distence)
            isInRange_big = false;

        std::array<ArmorPosi,2> armor_result;
        if(isInRange_small)
            armor_result[0] = ArmorPosi(center_small, center_cam_small, photocenter_world, yaw_small, reproj_small, true);
        if(isInRange_big)
            armor_result[1] = ArmorPosi(center_big, center_cam_big, photocenter_world, yaw_big, reproj_big, true);
        results.push_back(armor_result);
    }

    return results;
}
