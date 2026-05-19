#include "Solver.hpp"
#include "Armor.hpp"
#include "eigen3/Eigen/Dense"
#include "iostream"
#include <Eigen/src/Core/Matrix.h>
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

        std::array<double, 2> cos2_pitch_small;
        std::array<double, 2> cos2_pitch_big;

        std::array<double, 2> cos2_roll_small;
        std::array<double, 2> cos2_roll_big;

        // 小装甲板
        {
            std::vector<cv::Mat> rvecs, tvecs;
            std::vector<double> reprojErr;
            cv::solvePnPGeneric(objectSmallArmorP, armor.Lightcorners, cameraMatrix, distCoeffs,
                                rvecs, tvecs, false, cv::SOLVEPNP_IPPE,
                                cv::noArray(), cv::noArray(), reprojErr);
            int idx = 3;
            for(int s = 0; s < 2; s++) {
                cv::Mat R_cv;
                cv::Rodrigues(rvecs[s], R_cv);
                Eigen::Matrix3d R_arm2cam(Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_cv.ptr<double>()));
                Eigen::Vector3d T_cam(Eigen::Map<Eigen::Vector3d>(tvecs[s].ptr<double>()));
                Eigen::Vector3d toward_world = R_cam2world * R_arm2cam.col(2);
                Eigen::Vector3d Side_world = R_cam2world * R_arm2cam.col(0);

                // 2D 叉积判断法向量在视线左/右侧：
                // cross2d < 0 → 法向量在视线右侧
                // cross2d > 0 → 法向量在视线左侧
                Eigen::Vector3d T_base = R_cam2world * T_cam;
                double cross2d = T_base(0) * toward_world(1) - T_base(1) * toward_world(0);

                if(idx == 3) idx = (cross2d < 0) ? 0 : 1;
                else{ idx = 1-idx; }
                center_cam_small.col(idx) = T_cam;
                center_small.col(idx) = R_cam2world * T_cam + photocenter_world;
                yaw_small[idx] = std::atan2(-toward_world.y(), -toward_world.x());
                reproj_small[idx] = reprojErr[s];
                // 小装甲板角度判断
                cos2_pitch_small[idx] = toward_world(0) * toward_world(0) + toward_world(1) * toward_world(1);

                cos2_roll_small[idx] = Side_world(0) * Side_world(0) + Side_world(1) * Side_world(1);
            }
        }
        if(center_small.col(0).z() > range.max_high || center_small.col(0).z() < range.min_high ||
           center_small.col(1).z() > range.max_high || center_small.col(1).z() < range.min_high ||
           (reproj_small[0] > reproj_threshold && reproj_small[1] > reproj_threshold) ||
            (cos2_pitch_small[0] < range.cos2_pitch && cos2_pitch_small[1] < range.cos2_pitch)||
            (cos2_roll_small[0] < range.cos2_roll && cos2_roll_small[1] < range.cos2_roll) ||
           center_small.col(0).norm() > range.max_distence || center_small.col(1).norm() > range.max_distence)
            isInRange_small = false;


        // 大装甲板
        {
            std::vector<cv::Mat> rvecs, tvecs;
            std::vector<double> reprojErr;
            cv::solvePnPGeneric(objectBigArmorP, armor.Lightcorners, cameraMatrix, distCoeffs,
                                rvecs, tvecs, false, cv::SOLVEPNP_IPPE,
                                cv::noArray(), cv::noArray(), reprojErr);
            int idx = 3;
            for(int s = 0; s < 2; s++) {
                cv::Mat R_cv;
                cv::Rodrigues(rvecs[s], R_cv);
                Eigen::Matrix3d R_arm2cam(Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_cv.ptr<double>()));
                Eigen::Vector3d T_cam(Eigen::Map<Eigen::Vector3d>(tvecs[s].ptr<double>()));

                Eigen::Vector3d toward_world = R_cam2world * R_arm2cam.col(2);
                Eigen::Vector3d Side_world = R_cam2world * R_arm2cam.col(0);

                Eigen::Vector3d T_base = R_cam2world * T_cam;
                double cross2d = T_base(0) * toward_world(1) - T_base(1) * toward_world(0);
                if(idx == 3) idx = (cross2d < 0) ? 0 : 1;
                else{ idx = 1-idx; }
                center_cam_big.col(idx) = T_cam;
                center_big.col(idx) = R_cam2world * T_cam + photocenter_world;
                yaw_big[idx] = std::atan2(-toward_world.y(), -toward_world.x());
                reproj_big[idx] = reprojErr[s];

                // 大装甲板角度判断
                cos2_pitch_big[idx] = toward_world(0) * toward_world(0) + toward_world(1) * toward_world(1);

                cos2_roll_big[idx] = Side_world(0) * Side_world(0) + Side_world(1) * Side_world(1);
            }
        }
        if(center_big.col(0).z() > range.max_high || center_big.col(0).z() < range.min_high ||
           center_big.col(1).z() > range.max_high || center_big.col(1).z() < range.min_high ||
           (reproj_big[0] > reproj_threshold && reproj_big[1] > reproj_threshold) ||
            (cos2_pitch_big[0] < range.cos2_pitch && cos2_pitch_big[1] < range.cos2_pitch)||
            (cos2_roll_big[0] < range.cos2_roll && cos2_roll_big[1] < range.cos2_roll) ||
           center_big.col(0).norm() > range.max_distence || center_big.col(1).norm() > range.max_distence)
            isInRange_big = false; // 大装甲板解算失败不代表小装甲板解算失败，仍然保留小装甲板的结果

        std::array<ArmorPosi,2> armor_result;
        if(isInRange_small)
            armor_result[0] = ArmorPosi(center_small, center_cam_small, photocenter_world, yaw_small, reproj_small, true);
        if(isInRange_big)
            armor_result[1] = ArmorPosi(center_big, center_cam_big, photocenter_world, yaw_big, reproj_big, true);
        results.push_back(armor_result);
    }

    return results;
}

std::vector<ArmorPosi> Solver::operator () (const std::vector<YoloArmor>& armors, const Eigen::Quaterniond& gripper_to_world)
{
    std::vector<ArmorPosi> results;
    if (armors.empty()) return results;
    results.reserve(armors.size());

    const Eigen::Matrix3d R_gripper_to_world = gripper_to_world.toRotationMatrix();
    const Eigen::Matrix3d R_cam2world = R_gripper_to_world * this->R_Cam_to_gripper;
    const Eigen::Vector3d photocenter_world = R_gripper_to_world * this->T_Cam_to_gripper;

    auto decode_yolo_type = [](int id, ArmorPosi::Type& type, bool& is_big) -> bool {
        switch (id) {
            case 0: case 1: case 2:
                type = ArmorPosi::Type::guard;
                is_big = false;
                return true;
            case 3: case 4: case 5:
                type = ArmorPosi::Type::hero;
                is_big = true;
                return true;
            case 6: case 7: case 8:
                type = ArmorPosi::Type::two;
                is_big = false;
                return true;
            case 9: case 10: case 11:
                type = ArmorPosi::Type::three;
                is_big = false;
                return true;
            case 12: case 13: case 14:
                type = ArmorPosi::Type::four;
                is_big = false;
                return true;
            case 18: case 19: case 20:
                type = ArmorPosi::Type::outpost;
                is_big = false;
                return true;
            case 21: case 22: case 23: case 24:
                type = ArmorPosi::Type::base;
                is_big = true;
                return true;
            case 25: case 26: case 27: case 28:
                type = ArmorPosi::Type::base;
                is_big = false;
                return true;
            case 29: case 30: case 31:
                type = ArmorPosi::Type::three;
                is_big = false;
                return true;
            case 32: case 33: case 34:
                type = ArmorPosi::Type::four;
                is_big = false;
                return true;
            default:
                return false;
        }
    };

    auto solve_one = [&](const std::vector<cv::Point3d>& object_points,
                         const std::vector<cv::Point2f>& image_points,
                         ArmorPosi& armor) -> bool
    {
        if (image_points.size() < 4) {
            return false;
        }

        std::vector<cv::Mat> rvecs, tvecs;
        std::vector<double> reprojErr;
        if (!cv::solvePnPGeneric(object_points, image_points, cameraMatrix, distCoeffs,
                                 rvecs, tvecs, false, cv::SOLVEPNP_IPPE,
                                 cv::noArray(), cv::noArray(), reprojErr)) {
            return false;
        }

        if (rvecs.size() < 2 || tvecs.size() < 2 || reprojErr.size() < 2) {
            return false;
        }

        Eigen::Matrix<double, 3, 2> center_world = Eigen::Matrix<double, 3, 2>::Zero();
        Eigen::Matrix<double, 3, 2> center_cam = Eigen::Matrix<double, 3, 2>::Zero();
        std::array<double, 2> yaw{};
        std::array<double, 2> reproj{};

        int idx = 3;
        for (int s = 0; s < 2; ++s) {
            cv::Mat R_cv;
            cv::Rodrigues(rvecs[s], R_cv);
            Eigen::Matrix3d R_arm2cam(
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_cv.ptr<double>()));
            Eigen::Vector3d T_cam(Eigen::Map<Eigen::Vector3d>(tvecs[s].ptr<double>()));
            Eigen::Vector3d toward_world = R_cam2world * R_arm2cam.col(2);

            Eigen::Vector3d T_base = R_cam2world * T_cam;
            double cross2d = T_base(0) * toward_world(1) - T_base(1) * toward_world(0);
            if (idx == 3) {
                idx = (cross2d < 0) ? 0 : 1;
            } else {
                idx = 1 - idx;
            }

            center_cam.col(idx) = T_cam;
            center_world.col(idx) = R_cam2world * T_cam + photocenter_world;
            yaw[idx] = std::atan2(-toward_world.y(), -toward_world.x());
            reproj[idx] = reprojErr[s];
        }

        bool is_in_range = !(
            center_world.col(0).z() > range.max_high || center_world.col(0).z() < range.min_high ||
            center_world.col(1).z() > range.max_high || center_world.col(1).z() < range.min_high ||
            center_world.col(0).norm() > range.max_distence || center_world.col(1).norm() > range.max_distence ||
            (reproj[0] > reproj_threshold && reproj[1] > reproj_threshold)
        );

        armor = ArmorPosi(center_world, center_cam, photocenter_world, yaw, reproj, is_in_range);
        return true;
    };

    for (const auto& yolo_armor : armors) {
        ArmorPosi::Type target_type = ArmorPosi::Type::Unknow;
        bool is_big = false;
        if (!decode_yolo_type(yolo_armor.class_id, target_type, is_big)) {
            continue;
        }

        const auto& object_points = is_big ? this->objectBigArmorP : this->objectSmallArmorP;

        ArmorPosi armor;
        if (!solve_one(object_points, yolo_armor.keypoints, armor)) {
            continue;
        }

        if (!armor.IsInRange) {
            continue;
        }

        armor.type = target_type;
        armor.confidence = yolo_armor.conf;
        results.push_back(armor);
    }

    return results;
}
