#include "Solver.hpp"
#include "Armor.hpp"
#include "eigen3/Eigen/Dense"
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
}

// 用 Z 轴消歧选出一个 PnP 解，返回 (R_cv, T_cv, reproj_err)
static std::tuple<cv::Mat, cv::Mat, double> disambiguate(
    const std::vector<cv::Mat>& rvecs,
    const std::vector<cv::Mat>& tvecs,
    const std::vector<double>& reprojErr)
{
    double Z_data[3]{0, 0, 10};
    cv::Mat Z_vector(cv::Size(1,3), CV_64FC1, Z_data);

    cv::Mat r_0, r_1;
    cv::Rodrigues(rvecs.front(), r_0);
    cv::Rodrigues(rvecs.back(),  r_1);

    cv::Mat Z_cam_0 = r_0 * Z_vector;

    if(Z_cam_0.at<double>(2,0) > 0)
        return {r_0, tvecs.front(), reprojErr.front()};
    else
        return {r_1, tvecs.back(),  reprojErr.back()};
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
        std::array<ArmorPosi,2> armor_result;
        bool isInRange_small = true, isInRange_big = true;

        // 解算小装甲板
        Eigen::Vector3d center_small;
        double yaw_small, reproj_small;
        {
            std::vector<cv::Mat> rvecs, tvecs;
            std::vector<double> reprojErr;
            cv::solvePnPGeneric(objectSmallArmorP, armor.Lightcorners, cameraMatrix, distCoeffs,
                                rvecs, tvecs, false, cv::SOLVEPNP_IPPE,
                                cv::noArray(), cv::noArray(), reprojErr);

            auto [R_cv, T_cv, err] = disambiguate(rvecs, tvecs, reprojErr);

            cv::Mat P_cam = R_cv * SmallArmorCenter + T_cv;
            Eigen::Vector3d T_cam(P_cam.at<double>(0,0), P_cam.at<double>(1,0), P_cam.at<double>(2,0));
            center_small = R_cam2world * T_cam + photocenter_world;

            Eigen::Matrix3d R_armor;
            for(int r = 0; r < 3; r++)
                for(int c = 0; c < 3; c++)
                    R_armor(r,c) = R_cv.at<double>(r,c);
            // 装甲板法线（Z轴）在世界系的方向，即从机器人中心指向装甲板的方向
            Eigen::Vector3d normal_world = R_cam2world * (R_armor * Eigen::Vector3d(0,0,1));
            yaw_small = std::atan2(normal_world.y(), normal_world.x());
            reproj_small = err;
        }
        if(center_small.z() > range.max_high || center_small.z() < range.min_high ||
           center_small.norm() > range.max_distence)
            isInRange_small = false;

        // 解算大装甲板
        Eigen::Vector3d center_big;
        double yaw_big, reproj_big;
        {
            std::vector<cv::Mat> rvecs, tvecs;
            std::vector<double> reprojErr;
            cv::solvePnPGeneric(objectBigArmorP, armor.Lightcorners, cameraMatrix, distCoeffs,
                                rvecs, tvecs, false, cv::SOLVEPNP_IPPE,
                                cv::noArray(), cv::noArray(), reprojErr);

            auto [R_cv, T_cv, err] = disambiguate(rvecs, tvecs, reprojErr);

            cv::Mat P_cam = R_cv * BigArmorCenter + T_cv;
            Eigen::Vector3d T_cam(P_cam.at<double>(0,0), P_cam.at<double>(1,0), P_cam.at<double>(2,0));
            center_big = R_cam2world * T_cam + photocenter_world;

            Eigen::Matrix3d R_armor;
            for(int r = 0; r < 3; r++)
                for(int c = 0; c < 3; c++)
                    R_armor(r,c) = R_cv.at<double>(r,c);
            Eigen::Vector3d normal_world = R_cam2world * (R_armor * Eigen::Vector3d(0,0,1));
            yaw_big = std::atan2(normal_world.y(), normal_world.x());
            reproj_big = err;
        }
        if(center_big.z() > range.max_high || center_big.z() < range.min_high ||
           center_big.norm() > range.max_distence)
            isInRange_big = false;

        if(isInRange_small)
            armor_result[0] = ArmorPosi(center_small, photocenter_world, yaw_small, reproj_small, true);
        if(isInRange_big)
            armor_result[1] = ArmorPosi(center_big, photocenter_world, yaw_big, reproj_big, true);
        results.push_back(armor_result);
    }

    return results;
}
